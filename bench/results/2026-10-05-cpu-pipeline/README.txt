Native CPU expert pipeline (opt-in)
=================================

Scope
-----
STRATA_CPU_PIPELINE=1 changes the scheduler in ExpertPool::run_split_multi_native.
Unset or 0 keeps the existing scheduler. There are no RT, routing, quantization,
model-weight, or default-setting changes in this branch. The existing row kernels
and quantizers are shared by both schedules. No token-path allocations are added.
The pool gains 96 cache-line-separated readiness records (6 KiB).

The old schedule completes every gate/up task, parks the pool, quantizes every
expert's intermediate vectors on the host, then starts a second down-projection
phase. The new schedule lets the last gate/up producer of each expert quantize
that expert immediately. Down tasks can start when their own expert is ready,
overlapping quantization and down work with the tail of other gate/up tasks.
There is one final pool drain rather than two separate global phases.

Synchronization argument
------------------------
1. Gate/up tasks write disjoint rows. Each ends with an acquire/release increment
   of its expert's producer count. The final producer therefore sees all rows.
2. That producer quantizes the complete vectors and release-stores readiness.
   Every down task acquire-loads readiness before reading those vectors.
3. All producer tasks precede all consumer tasks in the claim queue. A waiting
   consumer cannot strand an unclaimed producer; its unfinished dependencies
   already belong to running producers.
4. The existing epoch-tagged claim, completion and worker-park protocol protects
   reuse between batches. Readiness waits have the same stall bound as pool waits.

This preserves the numerical operations and their per-row reduction order. It is
not a proof that every full-engine configuration emits identical text: adaptive
CPU/GPU expert placement and latency-based draft policy are separate mechanisms.
It is also not a guarantee of speedup. Small batches, few cores, other formats,
memory bandwidth saturation and a GPU-bound workload can change the tradeoff.

Tests (no model or GPU required)
-------------------------------
Configure with STRATA_BUILD_TESTS=ON and STRATA_NATIVE_EXPERTS=ON. The GGML source
is the project's usual dependency; STRATA_GGML_DIR may point to a local checkout.

  cmake -S . -B build-cpu -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF
  cmake --build build-cpu --target native_pool_pipeline_test
  ctest --test-dir build-cpu -R native_pool_pipeline --output-on-failure

The numerical matrix uses GGML-quantized nonzero synthetic weights for six format
pairs: IQ2_S/Q2_0, IQ2_XXS/Q2_0, IQ1_M/Q2_0, Q2_0/Q2_0, Q4_K/Q5_1 and
IQ4_XS/IQ4_NL. It compares all output bits, unused rows and row-end guards, with
1/3/15 workers, host participation on/off, 1..8-token expert groups, empty/tiny/
large batches, the 96-expert scratch boundary and sleeping workers. The AVX2 test
forces Strata's dispatch on this AVX-512 host; it does not simulate an older CPU's
performance. The scheduler stress mode adds Q8_0/Q8_0 and many smaller batches.

ThreadSanitizer uses a CPU-only build with -fsanitize=thread in C and C++ compile
flags and the executable link flags, then runs native_pool_pipeline_stress. The
full numerical matrix under TSan exceeded its 300-second limit after one format;
that is not counted as a sanitizer pass. See validation.json for the actual runs.

Inference reproduction
----------------------
  python tools/bench_cpu_pipeline.py --config YOUR-CONFIG.json \
    --exe build/strata --output NEW-DIRECTORY --expert-cache 5120 \
    --tokens 256 --repetitions 3

Adjust the fixed cache size to fit the machine. The helper does not change the
config or installed engine. It starts temporary --serve processes without HTTP,
uses the normal tokenizer/template and native token IDs, closes every process,
clears experimental overrides, fixes expert residency and PCIe offload, and
measures baseline-before / pipeline / baseline-after using the same binary.
An optional --reference-exe adds a separately built upstream/installed control.
Each process gets a warmup for each prompt. Model loading and prompt handling
are excluded from decode_ms; request wall time is retained separately.

The six cases cover code, mathematics, prose, Chinese and two seeded sampled
requests. Full token IDs and acceptance counts are retained. Timings for a case
are comparable only when its outputs match; the helper reports divergence and
withholds that case's speedup ratio. Run tests/sanitizers separately from timings.

Limitations
-----------
The inference evidence is for one model and workstation. Windows, AMD/Intel CPUs
other than the tested Ryzen, other model quantizations, long-context workloads,
adaptive expert-cache policies and different CPU/GPU residency ratios need their
own end-to-end qualification before considering a default-on change.

Measured inference result
-------------------------
Same-binary pipeline on versus the mean of the two off controls:

Case            Off tok/s   On tok/s    Throughput gain
code                93.18      99.10             6.35%
math                95.14     101.33             6.51%
prose               75.10      79.33             5.64%
chinese             94.40      99.73             5.65%
sample-code        106.03     110.65             4.36%
sample-prose        73.54      77.70             5.66%

All 72 requests, including warmups and the separately built installed reference,
matched complete token sequences within each case. Residency settings were fixed;
acceptance counts are retained per arm in summary.json (some differ across runs).
The same-binary comparison is the primary
performance evidence; the installed reference is an additional control, not a
claim that its historical build settings were reproduced exactly.

The last test-only edit added independent oracle scratch and changing inputs.
The full release matrix and focused TSan results above predate that edit; its
short release stress result is recorded separately in validation.json.
