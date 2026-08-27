# Audit summary: 20260827-next-stage-audit-local-partial

## Deployment status

- PC 30 FPS realtime: NOT_TESTED
- 50 FPS realtime: NOT_TESTED
- RK3588 realtime: BLOCKED
- Causal PGA: PASS
- Signal/PGA zero-lookahead causality: PASS
- Tracker source-timestamp causality: FAIL_FUTURE_CONTEXT
- End-to-end source-timestamp causality: FAIL
- PGA scientific validity: RESEARCH_ONLY
- Geometric scale: UNCALIBRATED
- Strict-causality interpretation: online availability alone is not treated as source-timestamp causality; see the manifest tracker future-context range.

## Required review questions (A-L)

- A. Strict realtime causality: `FAIL`; signal/PGA `PASS`, tracker `FAIL_FUTURE_CONTEXT`.
- B. PC 30 FPS without backlog: `NOT_TESTED`.
- C. PC 50 FPS realtime: `NOT_TESTED`.
- D. Silent frame drop: `NO_SILENT_DROP_EXPLICIT_REJECTION_ACCEPTANCE_FAIL`; frames/blocks `1` / `0`.
- E. Reseed fake peak: `NOT_EVALUABLE_SINGLE_EVENT_BELOW_THRESHOLD`; analyzed events `1`, acceleration-spike p95 ratio `1.2166072453659869`.
- F. Strong-motion data used for ALL/VIDEO-QUALITY selection: `NO`.
- G. ALL primary PGA metrics: `NOT_EVALUABLE`.
- H. VIDEO-QUALITY primary PGA metrics: `NOT_EVALUABLE`.
- I. Offline versus causal difference: see `offline_vs_causal_pga.png` and the causal signal benchmark; `NOT_EVALUABLE` if absent.
- J. CoTracker common/local/rotation stress evidence: `NOT_MEASURED`.
- K. RK3588 measured: `NO_BLOCKED_NO_DEVICE`.
- L. Evidence commit/config: `1548c049940cbfb104d963a56a9dd2b9c48439b4` / `effective_config.yaml`.

## Scope and provenance

- Git commit: `1548c049940cbfb104d963a56a9dd2b9c48439b4`; dirty: `False`.
- Input: `{'alignment': 'NOT_MEASURED', 'pga': 'runs/20260827-pga-evaluation-local-record48', 'reseed': 'runs/20260827-pga-causal-local-cpu/reseed-record-48', 'runtime': 'runs/20260825-realtime-local-cpu-smoke', 'stress': 'NOT_MEASURED'}`.
- Device: `cpu`.
- Checkpoint SHA-256: `205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218`.
- Tracker parameters: `{"backend": "cotracker3_online", "causality_semantics": "delayed_block_with_future_source_frames", "checkpoint": null, "cotracker_root": null, "iters": 6, "max_blocks_before_reseed": 64, "model_resolution": [384, 512], "num_points": 32, "point_mode": "corners", "source_timestamp_future_context_frames": [8, 15], "step": 8, "visibility_threshold": 0.6, "window_len": 16}`.
- Motion parameters: `{"global_model": "similarity", "max_fit_rmse_px": 2.0, "min_inlier_ratio": 0.6, "min_spatial_coverage": 0.08, "min_valid_tracks": 10, "quality_gate": true, "ransac_threshold_px": 1.5, "use_ransac": true}`.
- Signal parameters: `{"bandpass_hz": [0.3, 8.0], "causal": true, "derivative_method": "causal_polynomial", "detrend": false, "online": {"effective_lookahead_samples": 0, "filter_order": 4, "polynomial_order": 3, "startup_behavior": "emit_nan_derivatives_until_ready", "window_length": 9}, "timestamp_aware": true}`.
- PGA model: `videoeew-pga-v2-research`.
- Change summary: `strict causal online signal, unbiased subsets, realtime audit, reseed and stress`.
- Working-tree patch: `0 files, +0/-0 lines`.
- Changed paths: `[]`.

## Accuracy and quality

- Metrics: `{"causal_pga_status": "PASS", "end_to_end_source_timestamp_causality": "FAIL", "geometric_scale": "UNCALIBRATED", "pc_30_fps_realtime": "NOT_TESTED", "pc_50_fps_realtime": "NOT_TESTED", "pga": {"causal_input_values": ["1"], "causal_pga_status": "PASS", "deployment_prediction_allowed": false, "feature_version": "videoeew-motion-v1", "geometric_scale": "UNCALIBRATED", "group_cv_status": {"camera_group_cv": "NOT_EVALUABLE_METADATA_UNKNOWN", "event_group_cv": "NOT_EVALUABLE_METADATA_UNKNOWN", "record_group_cv": "PROVISIONAL_UNKNOWN_RECORD_RELATIONSHIPS", "site_group_cv": "NOT_EVALUABLE_METADATA_UNKNOWN"}, "input_feature_versions": ["videoeew-motion-v1"], "primary_algorithm": "single_coefficient", "scientific_validity": "RESEARCH_ONLY", "subsets": {"all": {"algorithms": {}, "included_rows": 1, "input_rows": 1, "interpretation": "PRIMARY_TRUTH_BLIND_SELECTION", "not_evaluable_reason": "fewer_than_two_included_records", "selection_uses_strong_motion": false}, "posthoc_aligned": {"algorithms": {}, "included_rows": 1, "input_rows": 1, "interpretation": "RESEARCH_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE", "not_evaluable_reason": "fewer_than_two_included_records", "selection_uses_strong_motion": true}, "video_quality": {"algorithms": {}, "included_rows": 1, "input_rows": 1, "interpretation": "PRIMARY_TRUTH_BLIND_SELECTION", "not_evaluable_reason": "fewer_than_two_included_records", "selection_uses_strong_motion": false}}, "target": "pga_horizontal_vector_gal"}, "reseed": {"acceleration_spike_ratio_p50": 1.2166072453659869, "acceleration_spike_ratio_p95": 1.2166072453659869, "mask_recommendation": "REVIEW_REQUIRED_IF_RATIO_P95_EXCEEDS_3", "plot_status": "GENERATED", "reseed_events_analyzed": 1, "reseed_events_in_log": 1}, "rk3588_realtime": "BLOCKED", "runtime": {"captured_frames": 32, "causality_note": "Signal filtering, derivatives, features, and running PGA use no future samples. CoTracker finalized blocks use future video frames relative to their source-frame timestamps; results are available online but do not satisfy strict source-timestamp causality.", "dropped_blocks": 0, "dropped_frames": 1, "end_to_end_source_timestamp_causality": "FAIL", "events": 4, "frames_written": 0, "observed_duration_s": 0.0, "pga_est": null, "pga_rejection_reason": "deployment_output_rejected_without_geometric_scale", "pga_research_output": "see running_pga.csv", "queue_slopes_per_block": {"block_queue_depth": NaN, "frame_queue_depth": NaN, "writer_queue_depth": NaN}, "queues": {"capture": {"capacity": 16, "depth": 16, "max_observed_depth": 16, "name": "capture", "rejected_items": 1}, "output": {"capacity": 2, "depth": 1, "max_observed_depth": 1, "name": "output", "rejected_items": 0}}, "realtime_acceptance": "NOT_TESTED", "realtime_acceptance_scope": "requires_at_least_590s_observed", "signal_pga_causality": "PASS", "stopped_for_overload": true, "target_fps": 30.0, "timing": {}, "tracker_source_timestamp_causality": "FAIL_FUTURE_CONTEXT", "tracker_source_timestamp_future_context_frames": [8, 15]}, "scientific_validity": "RESEARCH_ONLY", "signal_pga_causality": "PASS", "stress": {}, "tracker_source_timestamp_causality": "FAIL_FUTURE_CONTEXT"}`.
- Motion quality counts: `{}`.
- PGA rows included: `1`.
- Scale state: `uncalibrated`.

## Runtime behavior

- Tracker latency summary: `{}`.
- Total latency summary: `{}`.
- Queue depth slopes per row: `{}`.
- Dropped frames / blocks: `1` / `0`.
- Overload states: `['OVERLOAD']`.
- Peak RSS: `2425.68359375 MB`.
- Event counts: `{"overload": 1, "playlist_source_end": 1, "playlist_source_start": 1, "tracker_block": 1}`.

## Baseline comparison and known limitations

- Baseline metrics: `not supplied`.
- Large videos, checkpoints, raw tracks and binary signals are intentionally excluded from this compact bundle.
- Any missing field is visible as missing; the builder does not invent measurements.
- Failed/degraded samples are represented by motion-quality counts and events; raw large artifacts remain in the source run.
