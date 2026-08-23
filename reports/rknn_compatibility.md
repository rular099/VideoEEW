# RKNN compatibility and partition plan

Status: PC source audit implemented; actual RKNN conversion blocked because no
RK3588 board or installed matching RKNN Toolkit2/Runtime is available.

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

The pure PyTorch candidate currently matches the reference Einsum within the
unit-test tolerance. This is not yet an ONNX, RKNN or track-output equivalence
claim.

## Required board evidence

- converter build log and exact toolkit/runtime/driver versions;
- `reports/onnx_ops.csv` from actual exported graphs;
- feature max/mean/relative error for FP32/FP16/INT8 as applicable;
- end-to-end track mean/p95/max pixel error and visibility disagreement;
- NPU/CPU/GPU timing, transfer time, RSS, temperature and throttling;
- 10- and 30-minute queue/memory stability.

