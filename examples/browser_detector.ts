// Browser-side document corner detection with ONNX Runtime Web

import * as ort from "onnxruntime-web/wasm";

// Configure IMMEDIATELY at module level -- before any inference session
ort.env.wasm.numThreads = 1;                    // single-thread for iframe compat
ort.env.wasm.simd = true;                       // SIMD for performance
ort.env.wasm.wasmPaths = "/docaligner-ort/";    // custom-built ORT runtime

export const createDocAlignerDetector = (modelPath: string) => {
  let session: ort.InferenceSession | null = null;

  const initialize = async () => {
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  };

  const detect = async (image: HTMLCanvasElement) => {
    // Resize to 256x256, convert RGBA -> NCHW float32, normalize to [0,1]
    const input = preprocessImage(image);

    const feeds = {
      img: new ort.Tensor("float32", input, [1, 3, 256, 256]),
    };

    const start = performance.now();
    const outputs = await session!.run(feeds);
    const inferenceTime = performance.now() - start;

    // Model outputs normalized [0,1] coordinates directly
    const points = outputs["points"].data as Float32Array;
    const confidence = (outputs["has_obj"].data as Float32Array)[0];

    const corners = ["TL", "TR", "BR", "BL"].map((label, i) => ({
      x: points[i * 2],       // normalized x [0,1]
      y: points[i * 2 + 1],   // normalized y [0,1]
      label,
      score: confidence,
    }));

    return { corners, inferenceTime };
  };

  return { detect, initialize: initialize() };
};
