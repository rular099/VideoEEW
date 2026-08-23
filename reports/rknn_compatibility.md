# RKNN compatibility and partition plan

Status: fixed-shape encoder ONNX exported and PC numerical validation passed;
actual RKNN conversion is blocked because neither the matching RKNN Toolkit2
nor an RK3588 board/runtime is accessible.

## Verified feature-encoder boundary

- ONNX opset: 17;
- input/output: `[1,3,384,512]` -> `[1,128,96,128]`;
- ONNX size: 10,578,669 bytes; SHA-256
  `688ec661990154748b3388597cf2eef0bbe4f2b9ee3eeb14b2956fedfa4958c1`;
- ONNX checker: passed;
- ONNXRuntime 1.16.3 versus PyTorch: max absolute error `1.997e-6`,
  mean absolute error `1.772e-7`, mean relative error with `1e-6` floor
  `1.861e-5`.

The encoder graph contains Conv 22, InstanceNormalization 21, Relu 26, Resize
4, plus normalization/shape operators. It contains no GridSample or Einsum.
ONNX shape inference still represents parts of the Resize and normalization
path symbolically despite a fixed graph input/output; this must be tested by
the actual RKNN converter. PyTorch emitted an InstanceNorm training-mode export
warning even though the module was in eval mode. The very small ONNXRuntime
error is evidence of numerical equivalence on the deterministic test input,
but the warning remains a conversion-review item. Full details are in
`reports/onnx_ops.csv` and
`runs/20260823-deployment/feature_encoder_onnx_metrics.json`.

## Confirmed high-risk call chain

The pinned local CoTracker commit contains both:

- `torch.nn.functional.grid_sample` through `bilinear_sampler` for feature and
  correlation sampling;
- `torch.einsum("btnhwc,bnijc->btnhwij", ...)` in online correlation.

The project specification's RKNN Toolkit2 2.3.2 reference marks GridSample and
Einsum unsupported. This status must be rechecked against the converter version
installed for the eventual board; the report does not generalize it to future
versions.

## Planned split

1. Export fixed-shape feature encoder including input and L2 normalization.
2. Convert only that encoder to RKNN and compare feature tensors.
3. Keep GridSample on CPU NEON or Mali/OpenCL initially.
4. Replace the explicit correlation Einsum with the tested reshape + batched
   MatMul candidate only after full-track numerical regression.
5. Attempt UpdateFormer export only after encoder acceleration is measured.

At the production-like tensor shapes `[1,16,32,9,9,128]` and
`[1,32,7,7,128]`, the pure PyTorch batched-MatMul candidate matched the
reference Einsum exactly in the frozen CPU FP32 run (max/mean absolute and
relative error all 0). Its observed mean was 5.10 ms versus 5.78 ms for Einsum.
These host timings are informational, not an RK3588 estimate. The candidate has
not yet been substituted into the full tracker, so track-output error remains
explicitly unmeasured. Raw metrics are in
`runs/20260823-deployment/correlation_rewrite_metrics.json`.

## Required board evidence

- converter build log and exact toolkit/runtime/driver versions;
- `reports/onnx_ops.csv` from actual exported graphs;
- feature max/mean/relative error for FP32/FP16/INT8 as applicable;
- end-to-end track mean/p95/max pixel error and visibility disagreement;
- NPU/CPU/GPU timing, transfer time, RSS, temperature and throttling;
- 10- and 30-minute queue/memory stability.
