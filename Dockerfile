# Reproducible ONNX Runtime WASM build for BlazeDOC
# Produces custom minimal WASM with only 19 operators (~2.4 MB vs ~12 MB standard)
#
# Build the image (~30-40 min, needs ~8 GB RAM):
#   docker build -t ort-wasm-blazedoc .
#
# Extract WASM artifacts:
#   docker run --rm -v $(pwd)/output:/output ort-wasm-blazedoc
#
# Or pull pre-built from GHCR:
#   docker pull ghcr.io/nimdraugsael/fossasia_onnx_docker:latest
#   docker run --rm -v $(pwd)/output:/output ghcr.io/nimdraugsael/fossasia_onnx_docker:latest
#
# Output files in ./output/:
#   ort-wasm-simd.wasm          (~2.4 MB, SIMD-only, no SharedArrayBuffer needed)
#   ort-wasm-simd.mjs           (~11 KB, JS loader)
#   ort-wasm-simd-threaded.wasm (~2.4 MB, SIMD + threads, needs COOP/COEP headers)
#   ort-wasm-simd-threaded.mjs  (~18 KB, JS loader)

# --- Stage 1: Build environment ---
FROM ubuntu:22.04@sha256:3ba65aa20f86a0fad9df2b2c259c613df006b2e6d0bfcc8a146afb8c525a9751 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3=3.10.6-1~22.04.1 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pin ORT to exact commit that produced the working 2.4 MB WASM.
# This is main branch between v1.23.2 and v1.24.1 (VERSION_NUMBER=1.24.0).
ARG ORT_COMMIT=34b7558efde040e1b7cd017a9429dffa70ddae5f
ARG EMSDK_VERSION=4.0.11

# Clone ORT at exact commit, then init submodules
RUN git clone https://github.com/microsoft/onnxruntime.git /ort \
    && cd /ort \
    && git checkout ${ORT_COMMIT} \
    && git submodule update --init --recursive

WORKDIR /ort

# Install emsdk (pinned version, matches the real build)
RUN cd cmake/external/emsdk \
    && ./emsdk install ${EMSDK_VERSION} \
    && ./emsdk activate ${EMSDK_VERSION}

# These exact versions were discovered through mass trial-and-error.
# ORT doesn't document compatible build dependencies — you get to play
# dependency combinatorics: N packages × M versions × P Python versions.
# This specific combination is known to produce a working WASM build.
# Change any one of them and you get to mass-retry. Have fun.
#
# CMake via pip because ORT 1.24+ requires cmake >= 3.28 (ubuntu 22.04 only has 3.22)
RUN pip3 install --no-cache-dir \
    "cmake==4.1.2" \
    "numpy==2.2.6" \
    "flatbuffers==25.9.23" \
    "protobuf==5.29.4"

# Copy operator config (19 ops for BlazeDOC document corner detection model)
COPY required_operators.config /ort/required_operators.config

# Copy and run build script — artifacts baked into image
COPY build.sh /ort/build.sh
RUN chmod +x /ort/build.sh
RUN /ort/build.sh

# --- Stage 2: Tiny output image with just the artifacts ---
FROM alpine:3.19 AS output

COPY --from=builder /artifacts/ /artifacts/

# When run: copy artifacts to mounted /output volume
CMD ["sh", "-c", "cp -v /artifacts/* /output/ && echo '=== Done ===' && ls -lh /output/"]
