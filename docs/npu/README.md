# Rockchip NPU Documentation

This directory contains documentation for the RKNPU2 backend that provides neural network acceleration on Rockchip NPUs.

## Contents

- [Getting Started](GETTING_STARTED.md) - Quick start guide for building and running
- [Environment Variables](ENVIRONMENT.md) - Configuration via environment variables
- [Performance Tuning](PERFORMANCE.md) - Optimization and benchmarking
- [Hybrid Runtime Validation](HYBRID_RUNTIME_VALIDATION.md) - ROCK 5 phase-3 benchmark workflow
- [Testing Guide](TESTING.md) - Comprehensive test plan for verification

## Overview

The RKNPU2 backend enables llama.cpp models to run on Rockchip's Neural Processing Units (NPUs), providing significant speedups for matrix operations in transformer models.

### Key Features

- **Multi-core support**: Distribute workload across NPU cores
- **DMA-Heap allocation**: Zero-copy buffer sharing with NPU
- **Hybrid quantization**: Mix NPU and CPU processing by layer
- **Large model support**: Split factor for models exceeding IOVA space

### Supported Hardware

- Rockchip RK3588 (3 NPU cores, 6 TOPS INT8)
- Rockchip RK3588S (2 NPU cores, 4 TOPS INT8)
- Rockchip RK3576 (3 NPU cores, 6 TOPS INT8)

### Quick Links

- [Backend Documentation](../backend/RKNPU2.md) - Detailed backend information
- [Build Instructions](../build.md) - CMake and Makefile build options
- [Parameters Documentation](../parameters.md) - llama-cli parameters
