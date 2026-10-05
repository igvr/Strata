"""Fresh-process A/B for STRATA_CPU_PIPELINE, using actual --serve token IDs.

Requires an installed model and the server's Python dependencies. No HTTP server
is started. Each temporary engine is closed; the supplied config is not edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.server import StrataEngine
from serve.frontend import ChatTemplate
from strata_tokenizer import Tokenizer

CASES = [
    ("code", "Write a Python function that merges overlapping intervals. Explain the invariants and give three examples.", {}),
    ("math", "Explain how to estimate the square root of 19 using Newton’s method. Work through the arithmetic.", {}),
    ("prose", "Write a short story about a radio operator receiving an unexpected message from an abandoned station.", {}),
    ("chinese", "用中文介绍哈希表扩容的实现方法，并说明如何保持查询性能。", {}),
    ("sample-code", "Explain when to use a mutex, a semaphore, and a condition variable. Include practical examples.",
     {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "seed": 84719}),
    ("sample-prose", "Describe a small town preparing for its first winter festival, including a surprising complication.",
     {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "seed": 29387}),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--reference-exe", type=Path, help="optional separately built upstream control")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expert-cache", type=int, default=5120, help="fixed cache MiB; adjust to fit the machine")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--repetitions", type=int, default=3, help="includes one warmup per prompt and process")
    a = ap.parse_args()
    if a.repetitions < 2 or a.tokens < 1:
        ap.error("at least two repetitions and one token are required")
    a.exe = a.exe.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    args = cfg["args"][:]
    for flag, value in {"--expert-cache": a.expert_cache, "--prefill": 1024,
                        "--adapt-every": 0, "--pcie-frac": 0}.items():
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args.extend([flag, str(value)])
    cwd = Path(cfg.get("cwd", a.config.resolve().parent)).resolve()
    pack = Path(args[args.index("--pack") + 1])
    native = Path(args[args.index("--native") + 1])
    if not pack.is_absolute(): pack = cwd / pack
    if not native.is_absolute(): native = cwd / native
    tok = Tokenizer.from_gguf(str(native))
    template = ChatTemplate(pack / "tokenizer/chat_template.jinja")
    # All experimental overrides are cleared. Only the schedule changes between arms.
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env["CUDA_MODULE_LOADING"] = "EAGER"
    env["STRATA_DECODE_TIMING"] = "1"
    binary_hash = hashlib.sha256(a.exe.read_bytes()).hexdigest()
    arms = [("baseline-before", 0, a.exe), ("pipeline", 1, a.exe), ("baseline-after", 0, a.exe)]
    if a.reference_exe:
        arms.insert(0, ("upstream-reference", 0, a.reference_exe.resolve()))
    settings = dict(exe=str(a.exe), args=args, cwd=str(cwd), binary_sha256=binary_hash,
                    repetitions=a.repetitions, tokens=a.tokens, cases=CASES,
                    order=[arm for arm, _, _ in arms],
                    binaries={arm: {"path": str(exe), "sha256": hashlib.sha256(exe.read_bytes()).hexdigest()} for arm, _, exe in arms},
                    note="No experimental head; fixed expert placement and PCIe fraction; warmup excluded per prompt.")
    (a.output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    results = {}
    StrataEngine.silence_s = 60
    for arm, enabled, executable in arms:
        if hashlib.sha256(a.exe.read_bytes()).hexdigest() != binary_hash:
            raise RuntimeError("engine binary changed during the comparison")
        engine = None
        rows = []
        try:
            engine = StrataEngine(str(executable), args, cwd=str(cwd), log=str(a.output / (arm + ".log")),
                                  env=dict(env, STRATA_CPU_PIPELINE=str(enabled)))
            for rep in range(a.repetitions):
                # Rotate the order identically across arms to distribute request/cache position.
                for name, prompt, sampling in CASES[rep % len(CASES):] + CASES[:rep % len(CASES)]:
                    rendered = template.render([{"role": "user", "content": prompt}], enable_thinking=False)
                    ids = tok.encode(rendered, parse_special=True)
                    start = time.monotonic()
                    tokens = [t for t in engine.generate(ids, a.tokens, sampling, threading.Event()) if t is not None]
                    row = dict(case=name, repetition=rep, token_ids=tokens, stats=dict(engine.last),
                               wall_ms=1000 * (time.monotonic() - start),
                               sha256=hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest())
                    rows.append(row)
                    (a.output / (arm + ".json")).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
                    print(json.dumps(dict(arm=arm, case=name, repetition=rep, **row["stats"])), flush=True)
        finally:
            if engine is not None:
                engine.close()
        results[arm] = rows
    summary = dict(binary_sha256=binary_hash, all_token_ids_equal=True, cases={})
    for name, _, _ in CASES:
        selected = {arm: [r for r in rows if r["case"] == name] for arm, rows in results.items()}
        reference = selected["baseline-before"][0]["token_ids"]
        equal = all(r["token_ids"] == reference for rows in selected.values() for r in rows)
        summary["all_token_ids_equal"] &= equal
        measurements = {}
        for arm, rows in selected.items():
            measured = [r for r in rows if r["repetition"] > 0]
            times = [r["stats"]["decode_ms"] for r in measured]
            measurements[arm] = dict(mean_decode_ms=statistics.mean(times),
                sd_decode_ms=statistics.stdev(times) if len(times) > 1 else None,
                tokens_per_second=sum(len(r["token_ids"]) for r in measured) * 1000 / sum(times),
                acceptance=sorted(set((r["stats"]["drafts_accepted"], r["stats"]["drafts_offered"]) for r in rows)))
        baseline = (measurements["baseline-before"]["mean_decode_ms"] +
                    measurements["baseline-after"]["mean_decode_ms"]) / 2
        summary["cases"][name] = dict(tokens=len(reference), token_ids_equal=equal, arms=measurements,
            speedup_percent=100 * (baseline / measurements["pipeline"]["mean_decode_ms"] - 1) if equal else None,
            baseline_drift_percent=100 * (measurements["baseline-after"]["mean_decode_ms"] /
                                          measurements["baseline-before"]["mean_decode_ms"] - 1))
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["all_token_ids_equal"]:
        raise SystemExit("Token divergence: timing ratios are not comparable for the divergent cases")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
