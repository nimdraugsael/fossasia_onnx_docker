#!/bin/bash
set -euo pipefail

cd /ort
source cmake/external/emsdk/emsdk_env.sh

echo "=== ORT version: $(cat VERSION_NUMBER) ==="
echo "=== emsdk version: $(em++ --version | head -1) ==="
echo "=== Operator config: ==="
cat /ort/required_operators.config
echo ""

echo "=== Building non-threaded WASM (SIMD only) ==="
python3 tools/ci_build/build.py \
  --build_dir build_wasm \
  --config Release \
  --build_wasm \
  --enable_wasm_simd \
  --minimal_build \
  --include_ops_by_config /ort/required_operators.config \
  --disable_exceptions \
  --disable_rtti \
  --skip_tests \
  --parallel \
  --allow_running_as_root

echo "=== Building threaded WASM (SIMD + Threads) ==="
python3 tools/ci_build/build.py \
  --build_dir build_wasm_threaded \
  --config Release \
  --build_wasm \
  --enable_wasm_simd \
  --enable_wasm_threads \
  --minimal_build \
  --include_ops_by_config /ort/required_operators.config \
  --disable_exceptions \
  --disable_rtti \
  --skip_tests \
  --parallel \
  --allow_running_as_root

echo "=== Collecting artifacts ==="
mkdir -p /artifacts
cp build_wasm/Release/ort-wasm-simd.wasm /artifacts/
cp build_wasm/Release/ort-wasm-simd.mjs /artifacts/
cp build_wasm_threaded/Release/ort-wasm-simd-threaded.wasm /artifacts/
cp build_wasm_threaded/Release/ort-wasm-simd-threaded.mjs /artifacts/

echo "=== Build complete ==="
ls -lh /artifacts/
