// Native CPU expert scheduling parity. No model, CUDA runtime, or downloaded fixture is needed.
// Nonzero GGML-quantized rows exercise both GU branches and both down-quantization branches.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace {
constexpr int kExperts = cpu::ExpertPool::kMaxSplitMulti + 1;
constexpr int kActivations = 61; // with two weight blobs, every tested expert has a distinct input pair
constexpr int kStride = cpu::H + 16; // detect writes past an output row
constexpr float kSentinel = -1.2345e33f;

std::vector<uint8_t> make_blob(const cpu::NativeFmt& f, int seed) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.025f);
    auto fill = [&](int type, int rows, int cols, size_t row_bytes, uint8_t* dst) {
        // Sixteen independently quantized nonzero rows, rotated across the full matrix.
        // This keeps the test small while retaining real codebook, sign, scale and min terms.
        constexpr int patterns = 16;
        std::vector<float> weights((size_t) patterns * cols), imatrix((size_t) cols, 1.0f);
        for (auto& v : weights) v = normal(rng);
        std::vector<uint8_t> packed((size_t) patterns * row_bytes);
        const auto ty = (ggml_type) type;
        ggml_quantize_chunk(ty, weights.data(), packed.data(), 0, patterns, cols,
                            ggml_quantize_requires_imatrix(ty) ? imatrix.data() : nullptr);
        for (int r = 0; r < rows; ++r)
            std::memcpy(dst + (size_t) r * row_bytes,
                        packed.data() + (size_t) ((r * 7 + r / patterns) % patterns) * row_bytes, row_bytes);
    };
    fill(f.gu_type, cpu::FF, cpu::H, f.gu_row, blob.data());
    fill(f.gu_type, cpu::FF, cpu::H, f.gu_row, blob.data() + f.up_off);
    fill(f.d_type, cpu::H, cpu::FF, f.d_row, blob.data() + f.down_off);
    return blob;
}
} // namespace

int main(int argc, char** argv) try {
    const bool stress = argc == 2 && std::strcmp(argv[1], "--stress") == 0;
    if (argc > 1 && !stress) throw std::runtime_error("usage: native_pool_pipeline_test [--stress]");
    const int experts = stress ? 8 : kExperts;
    setvbuf(stdout, nullptr, _IONBF, 0);
    // IQ gate/up + Q2_0 down; Q2_0 gate/up; and GGML-quantized down activations.
    std::vector<std::array<int, 2>> formats = {{22, 42}, {16, 42}, {29, 42}, {42, 42}, {12, 7}, {23, 20}};
    if (stress) formats = {{42, 42}, {8, 8}}; // both intermediate quantizers, affordable under ThreadSanitizer
    const std::vector<int> batch_sizes = stress ? std::vector<int>{0, 1, 2, 7}
                                                : std::vector<int>{0, 1, 2, 7, 17, 48, kExperts};
    uint64_t compared = 0;
    for (const auto& pair : formats) {
        cpu::NativeFmt f;
        std::string error;
        if (!cpu::native_fmt(pair[0], pair[1], cpu::H, cpu::FF, f, error))
            throw std::runtime_error(error);
        const std::vector<std::vector<uint8_t>> blobs = {make_blob(f, 331 + pair[0]), make_blob(f, 971 + pair[0])};
        std::mt19937 rng(881 + pair[0]);
        std::normal_distribution<float> normal(0.0f, 0.7f);
        std::vector<float> x((size_t) kActivations * cpu::H);
        for (auto& v : x) v = normal(rng);
        std::vector<uint8_t> nact((size_t) kActivations * cpu::kNativeActBytes);
        cpu::ActQ act[kActivations];
        for (int t = 0; t < kActivations; ++t) {
            cpu::native_quant_act(f, x.data() + (size_t) t * cpu::H,
                                 nact.data() + (size_t) t * cpu::kNativeActBytes);
            cpu::act_quant_any(x.data() + (size_t) t * cpu::H, cpu::H, act[t]);
        }
        std::vector<cpu::ExpertJobMulti> jobs(experts);
        for (int e = 0; e < experts; ++e) {
            jobs[e].blob = blobs[e % 2].data();
            jobs[e].nt = 1 + (e * 3) % cpu::MAXT;
        }
        auto inputs = [&](int epoch) {
            for (int e = 0; e < experts; ++e)
                for (int t = 0; t < jobs[e].nt; ++t) {
                    const int at = (e + 7 * t + 11 * epoch) % kActivations;
                    jobs[e].act[t] = &act[at];
                    jobs[e].nact[t] = nact.data() + (size_t) at * cpu::kNativeActBytes;
                }
        };
        std::vector<float> reference((size_t) experts * cpu::MAXT * kStride), output(reference.size());
        auto bind = [&](std::vector<float>& buffer) {
            std::fill(buffer.begin(), buffer.end(), kSentinel);
            for (int e = 0; e < experts; ++e)
                for (int t = 0; t < jobs[e].nt; ++t)
                    jobs[e].out[t] = buffer.data() + ((size_t) e * cpu::MAXT + t) * kStride;
        };
        // Separate scratch is essential: the oracle must not leave correct intermediates in
        // the pipeline's buffers and conceal an early publication or a skipped producer.
        cpu::ExpertPool oracle(1, false, true);
        oracle.set_native_pipeline(false);
        int epoch = 0;
        for (int workers : {1, 3, 15}) for (bool host : {false, true}) {
            cpu::ExpertPool pool(workers, false, host);
            pool.set_native_pipeline(true);
            for (int n : batch_sizes) {
                for (int repeat = 0; repeat < (stress ? 4 : 3); ++repeat) {
                    if (repeat == 1) std::this_thread::sleep_for(std::chrono::milliseconds(23));
                    inputs(++epoch); // every batch needs fresh intermediate results
                    bind(reference);
                    oracle.run_split_multi_native(f, jobs.data(), n);
                    bool nonzero = false;
                    for (int e = 0; e < n; ++e)
                        for (int t = 0; t < jobs[e].nt; ++t)
                            for (int r = 0; r < cpu::H; ++r) {
                                const float v = jobs[e].out[t][r];
                                if (!std::isfinite(v) || v == kSentinel) throw std::runtime_error("invalid reference output");
                                nonzero |= v != 0.0f;
                            }
                    if (n > 0 && !nonzero) throw std::runtime_error("degenerate all-zero reference");
                    bind(output);
                    pool.run_split_multi_native(f, jobs.data(), n);
                    if (std::memcmp(reference.data(), output.data(), output.size() * sizeof(float)) != 0) {
                        std::fprintf(stderr, "mismatch: format %d/%d workers=%d host=%d experts=%d repeat=%d\n",
                                     pair[0], pair[1], workers, host, n, repeat);
                        return 1;
                    }
                    for (int e = 0; e < n; ++e) compared += (uint64_t) jobs[e].nt * cpu::H;
                }
            }
        }
        std::printf("format %d/%d: all schedules bit-identical\n", pair[0], pair[1]);
    }
    std::printf("PASS: %llu output floats plus unused rows and row guards\n", (unsigned long long) compared);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "native pool pipeline: %s\n", e.what());
    return 1;
}
