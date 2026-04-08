# CPU & Hybrid Optimization Guide

This guide is the fast path to better Synapse performance on CPU-only systems and hybrid CPU + Rockchip NPU setups. Because Synapse inherits its core CPU runtime from `ik_llama.cpp`, many of the same tuning strategies apply here as well.

In practice, a tuned CPU run can outperform an untuned NPU or hybrid run, so it is worth getting the CPU baseline right before layering in NPU routing.

## Need Help Choosing a Path?

Synapse supports multiple valid optimization paths depending on your model, quantization, memory limits, and hardware. If you are not sure where to start, use the **Ask DeepWiki** badge in the repository header for a recommended starting configuration, then validate it with your own benchmarks.

## Recommended CPU Flags

When running `llama-server` on CPU, these flags are the best place to start:

| Flag | What it does | Why it helps |
| --- | --- | --- |
| `-fa` | Enables Flash Attention | Improves throughput and reduces memory pressure |
| `-fmoe` | Enables fused MoE kernels | Speeds up supported Mixture-of-Experts models |
| `-ctk q8_0 -ctv q8_0` | Quantizes the KV cache | Reduces bandwidth and improves long-context performance |
| `-b <N> -ub <N>` | Raises logical and physical batch sizes | Improves prompt-processing throughput |
| `-rtr` | Repackages tensors at runtime when interleaved variants are available | Can unlock better CPU performance on some quantizations |
| `-mla 3` | Enables MLA support where available | Useful for compatible model families |

## Example: Strong CPU Baseline

```bash
./build/bin/llama-server \
  --model /path/to/model.gguf \
  --ctx-size 4096 \
  -t "$(nproc)" \
  -fa \
  -fmoe \
  -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 2048 \
  -rtr \
  -mla 3
```

Adjust `-t`, `-b`, and `-ub` to fit your hardware and memory budget. If you have enough RAM, larger batches are often worth testing.

## ARM-Specific Build Notes

On ARM systems such as RK3588 and RK3576, `GGML_NATIVE=ON` is a good default:

```bash
cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

If you want to force architecture flags explicitly, this is a useful starting point:

```bash
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_ARCH_FLAGS="-march=armv8.2-a+dotprod+fp16" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

If your CPU supports newer extensions, you can experiment with variants such as `-march=armv8.7-a+dotprod+fp16`.

## Quantization Guidance

For CPU inference, the `K` family is usually the best place to start:

- `IQ3_K` and `IQ4_K` are strong general-purpose choices
- `IQ3_K_R4` and `IQ4_K_R4` can deliver better CPU throughput when repacking is available
- `IQ2_KS` is useful when memory is extremely constrained

As a rule of thumb:

- Prefer `K` formats for CPU and ARM-focused runs
- Be cautious with `KT` formats on CPU, where they may trade too much speed for quality
- Re-test on your actual model family, because MoE and dense models respond differently

## Hybrid CPU + NPU Tips

Hybrid mode lets Synapse split work between the CPU and Rockchip NPU. To get the best results:

- Start with a known-good manifest from [examples/hybrid-manifests](../examples/hybrid-manifests)
- Keep CPU-side optimizations enabled even in hybrid mode
- Test both conservative and aggressive routing profiles instead of assuming more NPU use is always faster
- Measure end-to-end throughput, not just startup behavior

Example hybrid run:

```bash
./build/bin/llama-server \
  --model /path/to/model.gguf \
  --hybrid-manifest examples/hybrid-manifests/dense-balanced.json \
  --ctx-size 4096 \
  -fa -fmoe -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -rtr
```

If you are tuning manifests and want failures to surface immediately, add `--hybrid-strict`.

## NPU Workflow Notes

To execute work on the Rockchip NPU, you still need the Rockchip runtime and a model workflow that matches your target backend:

- Install the `rknpu2` runtime on the device
- Build Synapse with `-DGGML_RKNPU2=ON`
- Validate your chosen manifest against the actual model and hardware

For backend-specific details, see [docs/backend/RKNPU2.md](backend/RKNPU2.md).

## Benchmarking Checklist

When comparing configurations, keep the process consistent:

- Use the same model, context size, and prompt shape
- Warm up before recording throughput
- Record both prompt-processing and token-generation rates
- Track memory pressure and thermals on embedded boards
- Change one variable at a time when tuning flags or manifests

If you are working on RK3588-specific tuning, the workflows in [docs/prompts/rock5/README.md](prompts/rock5/README.md) are a good companion.

## Suggested Tuning Order

1. Get a clean CPU baseline working first
2. Add `-fa`, KV quantization, and larger batch sizes
3. Test `-fmoe` and `-rtr` on the specific model you care about
4. Introduce a hybrid manifest from `examples/hybrid-manifests`
5. Compare throughput across CPU-only, hybrid, and NPU-heavier profiles

That process usually surfaces the real bottleneck faster than trying to jump straight to a highly customized hybrid configuration.
