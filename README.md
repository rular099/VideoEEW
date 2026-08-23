# VideoEEW

VideoEEW is an auditable engineering pipeline for extracting common image
motion and local residual motion from video with CoTracker3, then building a
quality-gated empirical coarse-PGA estimator.

The project deliberately separates four claims:

1. CoTracker recovers sparse image-plane tracks.
2. Robust geometry separates common image motion from local residuals.
3. Timestamp-aware signal processing derives motion features.
4. A separately evaluated empirical model maps those features to PGA truth.

Until a geometric scale is supplied, motion remains in pixels and physical
PGA output is rejected as uncalibrated. RK3588 measurements are also withheld
until real board access is available.

## Development setup

The existing CoTracker checkout remains outside this repository. Point the
adapter at it explicitly:

```bash
export COTRACKER_ROOT=/path/to/co-tracker
python -m pip install -e '.[dev,pc]'
```

Configuration is versioned under `configs/`. Runtime inputs, checkpoints and
large binary outputs are not committed. Every run writes enough metadata to
identify those external inputs by path and SHA-256.

## Intended commands

```bash
python -m seismic_motion.cli.run --config configs/pc_baseline.yaml --video VIDEO
python -m seismic_motion.cli.benchmark_synthetic --config configs/synthetic.yaml
python -m seismic_motion.cli.evaluate_pga --dataset configs/pga_dataset.example.yaml
python scripts/build_audit_bundle.py --run runs/RUN_ID
```

See `reports/IMPLEMENTATION_STATUS.md` for completed and blocked milestones.

