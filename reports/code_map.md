# Review code map

The implementation follows the paths requested by the execution plan:

```text
seismic_motion/tracking/cotracker_adapter.py
seismic_motion/tracking/online_buffer.py
seismic_motion/motion/global_motion.py
seismic_motion/motion/residual_motion.py
seismic_motion/motion/quality.py
seismic_motion/signal/filtering.py
seismic_motion/signal/derivatives.py
seismic_motion/signal/features.py
seismic_motion/pga/model.py
seismic_motion/runtime/pipeline.py
seismic_motion/runtime/metrics.py
scripts/build_audit_bundle.py
```

Additional deployment and dataset entry points are:

```text
seismic_motion/pga/records.py
seismic_motion/pga/alignment.py
seismic_motion/deployment/feature_encoder.py
seismic_motion/deployment/correlation.py
seismic_motion/deployment/operator_audit.py
seismic_motion/runtime/realtime.py
scripts/run_pga_feature_batch.py
scripts/recompute_alignments.py
scripts/export_feature_encoder.py
scripts/validate_feature_encoder_onnx.py
scripts/audit_operators.py
scripts/plot_run.py
```

Phase-by-phase evidence and blocked items are recorded in
`reports/IMPLEMENTATION_STATUS.md`.
