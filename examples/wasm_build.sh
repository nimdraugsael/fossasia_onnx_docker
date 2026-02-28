#!/bin/bash
# Custom ONNX Runtime WASM build -- only operators used by our model
# Result: 2.4 MB vs ~12 MB standard build (80% smaller)

# 19 operators included (vs 150+ in standard build):
# Conv, ConvTranspose, BatchNormalization, Relu, Sigmoid,
# Add, Mul, ArgMax, Reshape, Concat, Cast, ...

python tools/ci_build/build.py \
  --build_dir build_wasm \
  --config Release \
  --build_wasm \
  --enable_wasm_simd \
  --minimal_build \
  --include_ops_by_config required_operators.config \
  --disable_exceptions \
  --disable_rtti \
  --skip_tests \
  --parallel

# Key flags:
#   --minimal_build              Only essential code
#   --include_ops_by_config      Only operators our model uses (19 ops)
#   --disable_exceptions         Reduce size (no C++ exceptions)
#   --disable_rtti               Reduce size (no runtime type info)
#   --enable_wasm_simd           2-4x speedup with SIMD instructions

# Output:
#   ort-wasm-simd.wasm  ~2.4 MB  (vs ~12 MB standard)
#   ort-wasm-simd.mjs   ~11 KB
