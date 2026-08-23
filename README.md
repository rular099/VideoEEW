# VideoEEW

VideoEEW is an auditable CoTracker3 wrapper for extracting common image-plane
motion and local residual motion from video, deriving timestamp-aware signal
features, and evaluating a quality-gated empirical coarse-PGA mapping.

The implementation deliberately separates four claims:

1. CoTracker recovers sparse image-plane tracks.
2. Robust geometry separates common image motion from local residuals.
3. Timestamp-aware filtering and derivatives produce pixel-domain features.
4. A separately cross-validated empirical model maps features to instrument
   PGA truth.

No geometric scale is currently available for the supplied videos. Therefore
all motion remains in pixels and the runtime rejects physical PGA output with
`scale_invalid_and_model_not_trained`. Offline PGA cross-validation is marked
research-only. No RK3588 performance claim is made without an accessible board.

## Current verified baseline

- External CoTracker commit: `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`.
- Scaled-online checkpoint SHA-256:
  `205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218`.
- Tracker input: 16 frames; finalized step: 8 frames; 32 points; 6 iterations;
  model input 384 x 512.
- Pure-translation rendered tracking RMSE: 0.0942 px.
- Translation/rotation/local rendered tracking RMSE: 0.4368 px; local residual
  RMSE: 0.3103 px.
- V100S rendered warm p95: 141.8 ms/block and 128.5 ms/block, both below the
  30 FPS eight-frame arrival budget of 266.7 ms.
- Ten-minute bounded-buffer fast-forward: 18,000 frames, final buffer 16,
  maximum queue depth 2, RSS slope 0.00185 MB/min, and every rejected enqueue
  explicitly counted.

These values are frozen in `runs/` and guarded by regression tests. See
`reports/IMPLEMENTATION_STATUS.md` for scope and caveats.

## Setup

The existing CoTracker checkout remains outside this repository and is never
modified by VideoEEW. Point the CLI at it explicitly.

```bash
python -m pip install -e '.[dev,pc,onnx]'
```

`requirements/pc.txt` is the install constraint set;
`requirements/environment-lock.yaml` freezes the exact local/server versions
used for the recorded evidence, including the read-only ONNXRuntime ABI
workaround.

Configuration is versioned under `configs/`. Checkpoints, private videos,
strong-motion records, ONNX/RKNN binaries and large NPZ files stay external.
Every full run records config/input/checkpoint identity, Git state, environment,
timing, memory, queue state, tracks, motion quality, features and rejection
reason.

## PC offline pipeline

```bash
python -m seismic_motion.cli.run \
  --config configs/pc_baseline.yaml \
  --video /path/to/video.avi \
  --cotracker-root /path/to/co-tracker \
  --checkpoint /path/to/scaled_online.pth \
  --device cuda \
  --output runs/my-pc-run

python scripts/plot_run.py --run runs/my-pc-run
```

The decoder is incremental. Tracker windows, history reseeding and runtime
queues are bounded. A partial final window is padded only for inference and is
cropped back to the valid frame count. Offline zero-phase filtering is clearly
labelled and is not represented as realtime filtering.

## Synthetic benchmark

```bash
python -m seismic_motion.cli.benchmark_synthetic \
  --config configs/synthetic.yaml
```

The frozen suite covers translation, rotation, local vibration, subpixel
amplitudes, several frequencies/FPS values, occlusion, blur and illumination
change. Exact-track oracle results validate geometry/signal code; separate
rendered CoTracker cases validate the model and are never conflated with the
oracle.

## Real video and strong-motion evaluation

Build a pairing manifest without copying private data:

```bash
python scripts/build_pga_dataset_manifest.py \
  --data-root '/path/to/视频数据/数据' \
  --output runs/pga-dataset/dataset_manifest.csv \
  --summary runs/pga-dataset/dataset_summary.json \
  --redact-paths
```

Extract one feature row per selected record:

```bash
python scripts/run_pga_feature_batch.py \
  --data-root '/path/to/视频数据/数据' \
  --record-ids 6 16 26 48 61 65 69 71 84 85 \
  --config configs/real_video_eval.yaml \
  --cotracker-root /path/to/co-tracker \
  --checkpoint /path/to/scaled_online.pth \
  --device cuda \
  --output-root runs/pga-real
```

The comparison searches both acceleration-proxy and displacement domains. A
full-record search is labelled an exploratory candidate because choosing the
maximum across many offsets can inflate correlation. It is not treated as
synchronization proof.

```bash
python scripts/recompute_alignments.py \
  --data-root '/path/to/视频数据/数据' \
  --run-root runs/pga-real

python -m seismic_motion.cli.evaluate_pga \
  --dataset runs/pga-real/pga_features.csv \
  --config configs/pga_train.yaml \
  --output-dir runs/pga-evaluation
```

Evaluation uses record/event groups, not frame-random splits. It reports simple
single-coefficient, Ridge and Huber baselines; absolute and multiplicative
errors; factor-of-1.5/factor-of-2 rates; Pearson/Spearman correlation; PGA-bin
confusion matrices; and available group breakdowns. Because camera/site IDs and
scale are missing, cross-camera/site generalization and deployment PGA remain
unvalidated.

## Realtime runner

```bash
python -m seismic_motion.cli.realtime \
  --config configs/rk3588_realtime.yaml \
  --camera /dev/video0 \
  --cotracker-root /path/to/co-tracker \
  --checkpoint /path/to/scaled_online.pth \
  --device cuda \
  --duration-s 600 \
  --output runs/realtime-test
```

Capture and output queues have fixed capacity. On overload, processing stops
and writes an explicit event instead of silently dropping frames. The run emits
standard timing, memory and queue CSVs plus a rejection-gated summary. The
RK3588 configuration is ready, but board acceptance requires real 10- and
30-minute measurements including temperature and throttling.

## ONNX/RKNN work

The first deployment boundary is the fixed-shape feature encoder:

```bash
python scripts/export_feature_encoder.py \
  --cotracker-root /path/to/co-tracker \
  --checkpoint /path/to/scaled_online.pth \
  --output artifacts/cotracker_encoder.onnx \
  --device cuda

python scripts/audit_operators.py \
  --cotracker-root /path/to/co-tracker \
  --onnx artifacts/cotracker_encoder.onnx \
  --output reports/onnx_ops.csv
```

The full CoTracker call chain contains GridSample and Einsum, both high-risk
for the referenced RKNN Toolkit2 2.3.2 support table. The planned partition
keeps coordinate sampling on CPU/Mali initially and uses a numerically tested
batched-MatMul correlation candidate. Actual RKNN conversion and feature/track
equivalence require the matching toolkit and a real board. The fixed encoder
ONNX has already passed ONNX checker and ONNXRuntime comparison with maximum
absolute feature error `2.0e-6`; see `reports/rknn_compatibility.md`.

## Audit bundle and tests

```bash
python scripts/plot_run.py --run runs/RUN_ID
python scripts/build_audit_bundle.py \
  --run runs/RUN_ID \
  --audit-root audit \
  --zip

PYTHONPATH=. python -m unittest discover -s tests -v
```

The compact audit bundle excludes raw videos, checkpoints and large binary
tracks but includes provenance, metrics, latency/memory/queue summaries, motion
quality, a PGA sample and plots. Missing evidence remains visibly missing; the
builder never invents measurements.
