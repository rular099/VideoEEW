# Audit summary: 20260826-next-stage-audit-partial

## Deployment status

- PC 30 FPS realtime: NOT_TESTED
- 50 FPS realtime: NOT_TESTED
- RK3588 realtime: BLOCKED
- Causal PGA: FAIL
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
- E. Reseed fake peak: `NOT_EVALUABLE`; acceleration-spike p95 ratio `None`.
- F. Strong-motion data used for ALL/VIDEO-QUALITY selection: `NO`.
- G. ALL primary PGA metrics: `{"fraction_within_factor_1p5": 0.3, "fraction_within_factor_2": 0.6, "log_pga_mae": 0.9009784310940608, "mae": 112.63645160056869, "median_multiplicative_error": 1.8629259240028397, "pearson": 0.5057293857426376, "rmse": 133.73348559525422, "spearman": 0.6}`.
- H. VIDEO-QUALITY primary PGA metrics: `{"fraction_within_factor_1p5": 0.3, "fraction_within_factor_2": 0.6, "log_pga_mae": 0.9009784310940608, "mae": 112.63645160056869, "median_multiplicative_error": 1.8629259240028397, "pearson": 0.5057293857426376, "rmse": 133.73348559525422, "spearman": 0.6}`.
- I. Offline versus causal difference: see `offline_vs_causal_pga.png` and the causal signal benchmark; `NOT_EVALUABLE` if absent.
- J. CoTracker common/local/rotation stress evidence: `NOT_MEASURED`.
- K. RK3588 measured: `NO_BLOCKED_NO_DEVICE`.
- L. Evidence commit/config: `3697ca58afea8d90fa79bc3282c2ef20ef48e1d4` / `effective_config.yaml`.

## Scope and provenance

- Git commit: `3697ca58afea8d90fa79bc3282c2ef20ef48e1d4`; dirty: `False`.
- Input: `{'alignment': 'NOT_MEASURED', 'pga': '/home/zhangb/work/people/zhangbei/cotracker_rk3588/VideoEEW/runs/20260823-pga-evaluation-v2-offline-input', 'reseed': 'NOT_MEASURED', 'runtime': '/home/zhangb/work/people/zhangbei/cotracker_rk3588/VideoEEW/runs/20260825-realtime-local-cpu-smoke', 'stress': 'NOT_MEASURED'}`.
- Device: `server-242-or-NOT_MEASURED`.
- Checkpoint SHA-256: `unknown`.
- Tracker parameters: `{"backend": "cotracker3_online", "causality_semantics": "delayed_block_with_future_source_frames", "checkpoint": null, "cotracker_root": null, "iters": 6, "max_blocks_before_reseed": 64, "model_resolution": [384, 512], "num_points": 32, "point_mode": "corners", "source_timestamp_future_context_frames": [8, 15], "step": 8, "visibility_threshold": 0.6, "window_len": 16}`.
- Motion parameters: `{"global_model": "similarity", "max_fit_rmse_px": 2.0, "min_inlier_ratio": 0.6, "min_spatial_coverage": 0.08, "min_valid_tracks": 10, "quality_gate": true, "ransac_threshold_px": 1.5, "use_ransac": true}`.
- Signal parameters: `{"bandpass_hz": [0.3, 8.0], "causal": true, "derivative_method": "causal_polynomial", "detrend": false, "online": {"effective_lookahead_samples": 0, "filter_order": 4, "polynomial_order": 3, "startup_behavior": "emit_nan_derivatives_until_ready", "window_length": 9}, "timestamp_aware": true}`.
- PGA model: `videoeew-pga-v2-research`.
- Change summary: `strict causal online signal, unbiased subsets, realtime audit, reseed and stress`.
- Working-tree patch: `0 files, +0/-0 lines`.
- Changed paths: `[]`.

## Accuracy and quality

- Metrics: `{"causal_pga_status": "FAIL", "end_to_end_source_timestamp_causality": "FAIL", "geometric_scale": "UNCALIBRATED", "pc_30_fps_realtime": "NOT_TESTED", "pc_50_fps_realtime": "NOT_TESTED", "pga": {"causal_input_values": ["0"], "causal_pga_status": "FAIL", "deployment_prediction_allowed": false, "geometric_scale": "UNCALIBRATED", "group_cv_status": {"camera_group_cv": "NOT_EVALUABLE_METADATA_UNKNOWN", "event_group_cv": "NOT_EVALUABLE_METADATA_UNKNOWN", "record_group_cv": "PROVISIONAL_UNKNOWN_RECORD_RELATIONSHIPS", "site_group_cv": "NOT_EVALUABLE_METADATA_UNKNOWN"}, "primary_algorithm": "single_coefficient", "scientific_validity": "RESEARCH_ONLY", "subsets": {"all": {"algorithms": {"huber": {"fraction_within_factor_1p5": 0.4, "fraction_within_factor_2": 0.5, "log_pga_mae": 1.6126157752754182, "mae": 506778.4402237637, "median_multiplicative_error": 1.99920895046741, "pearson": -0.4834925572848771, "rmse": 1602287.8724885928, "spearman": -0.01818181818181818}, "log_linear": {"fraction_within_factor_1p5": 0.1, "fraction_within_factor_2": 0.2, "log_pga_mae": 10.168637124935236, "mae": 2.953506522288751e+36, "median_multiplicative_error": 9.2371445102764, "pearson": -0.4835263271494577, "rmse": 9.339807694595319e+36, "spearman": -0.24848484848484845}, "median": {"fraction_within_factor_1p5": 0.5, "fraction_within_factor_2": 0.7, "log_pga_mae": 0.48346306573347775, "mae": 96.82382347682722, "median_multiplicative_error": 1.4771348357164042, "pearson": -0.33573106763000254, "rmse": 119.14514227910954, "spearman": -0.3302891295379082}, "ridge": {"fraction_within_factor_1p5": 0.4, "fraction_within_factor_2": 0.7, "log_pga_mae": 2.008299452117024, "mae": 37533821.71347526, "median_multiplicative_error": 1.9177660408974986, "pearson": -0.4835258474343586, "rmse": 118692098.48409678, "spearman": 0.0303030303030303}, "single_coefficient": {"fraction_within_factor_1p5": 0.3, "fraction_within_factor_2": 0.6, "log_pga_mae": 0.9009784310940608, "mae": 112.63645160056869, "median_multiplicative_error": 1.8629259240028397, "pearson": 0.5057293857426376, "rmse": 133.73348559525422, "spearman": 0.6}}, "included_rows": 10, "input_rows": 10, "interpretation": "PRIMARY_TRUTH_BLIND_SELECTION", "selection_uses_strong_motion": false}, "posthoc_aligned": {"algorithms": {"huber": {"fraction_within_factor_1p5": 0.42857142857142855, "fraction_within_factor_2": 0.7142857142857143, "log_pga_mae": 0.4963443532794786, "mae": 105.16428714773504, "median_multiplicative_error": 1.5589573723716226, "pearson": 0.04117158473377689, "rmse": 117.85604278380897, "spearman": 0.07142857142857144}, "log_linear": {"fraction_within_factor_1p5": 0.42857142857142855, "fraction_within_factor_2": 0.7142857142857143, "log_pga_mae": 0.5700125754182731, "mae": 119.8998668308702, "median_multiplicative_error": 1.6002217622157346, "pearson": 0.2701303110728397, "rmse": 137.22104648072974, "spearman": 0.3214285714285715}, "median": {"fraction_within_factor_1p5": 0.7142857142857143, "fraction_within_factor_2": 0.8571428571428571, "log_pga_mae": 0.3391926708700708, "mae": 77.34740327079278, "median_multiplicative_error": 1.2333708826233318, "pearson": -0.7623452191858969, "rmse": 99.8363466098214, "spearman": -0.7525786756697113}, "ridge": {"fraction_within_factor_1p5": 0.42857142857142855, "fraction_within_factor_2": 0.7142857142857143, "log_pga_mae": 0.527634906536033, "mae": 112.41906504091149, "median_multiplicative_error": 1.5589573723716226, "pearson": -0.05264063946405329, "rmse": 126.66941154463095, "spearman": 0.07142857142857144}, "single_coefficient": {"fraction_within_factor_1p5": 0.5714285714285714, "fraction_within_factor_2": 0.7142857142857143, "log_pga_mae": 0.49309646743161134, "mae": 94.4184449581068, "median_multiplicative_error": 1.3657508510146175, "pearson": 0.2576483839125422, "rmse": 118.08073313865391, "spearman": 0.4642857142857144}}, "included_rows": 7, "input_rows": 10, "interpretation": "RESEARCH_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE", "selection_uses_strong_motion": true}, "video_quality": {"algorithms": {"huber": {"fraction_within_factor_1p5": 0.4, "fraction_within_factor_2": 0.5, "log_pga_mae": 1.6126157752754182, "mae": 506778.4402237637, "median_multiplicative_error": 1.99920895046741, "pearson": -0.4834925572848771, "rmse": 1602287.8724885928, "spearman": -0.01818181818181818}, "log_linear": {"fraction_within_factor_1p5": 0.1, "fraction_within_factor_2": 0.2, "log_pga_mae": 10.168637124935236, "mae": 2.953506522288751e+36, "median_multiplicative_error": 9.2371445102764, "pearson": -0.4835263271494577, "rmse": 9.339807694595319e+36, "spearman": -0.24848484848484845}, "median": {"fraction_within_factor_1p5": 0.5, "fraction_within_factor_2": 0.7, "log_pga_mae": 0.48346306573347775, "mae": 96.82382347682722, "median_multiplicative_error": 1.4771348357164042, "pearson": -0.33573106763000254, "rmse": 119.14514227910954, "spearman": -0.3302891295379082}, "ridge": {"fraction_within_factor_1p5": 0.4, "fraction_within_factor_2": 0.7, "log_pga_mae": 2.008299452117024, "mae": 37533821.71347526, "median_multiplicative_error": 1.9177660408974986, "pearson": -0.4835258474343586, "rmse": 118692098.48409678, "spearman": 0.0303030303030303}, "single_coefficient": {"fraction_within_factor_1p5": 0.3, "fraction_within_factor_2": 0.6, "log_pga_mae": 0.9009784310940608, "mae": 112.63645160056869, "median_multiplicative_error": 1.8629259240028397, "pearson": 0.5057293857426376, "rmse": 133.73348559525422, "spearman": 0.6}}, "included_rows": 10, "input_rows": 10, "interpretation": "PRIMARY_TRUTH_BLIND_SELECTION", "selection_uses_strong_motion": false}}, "target": "pga_horizontal_vector_gal"}, "reseed": {}, "rk3588_realtime": "BLOCKED", "runtime": {"captured_frames": 32, "causality_note": "Signal filtering, derivatives, features, and running PGA use no future samples. CoTracker finalized blocks use future video frames relative to their source-frame timestamps; results are available online but do not satisfy strict source-timestamp causality.", "dropped_blocks": 0, "dropped_frames": 1, "end_to_end_source_timestamp_causality": "FAIL", "events": 4, "frames_written": 0, "observed_duration_s": 0.0, "pga_est": null, "pga_rejection_reason": "deployment_output_rejected_without_geometric_scale", "pga_research_output": "see running_pga.csv", "queue_slopes_per_block": {"block_queue_depth": NaN, "frame_queue_depth": NaN, "writer_queue_depth": NaN}, "queues": {"capture": {"capacity": 16, "depth": 16, "max_observed_depth": 16, "name": "capture", "rejected_items": 1}, "output": {"capacity": 2, "depth": 1, "max_observed_depth": 1, "name": "output", "rejected_items": 0}}, "realtime_acceptance": "NOT_TESTED", "realtime_acceptance_scope": "requires_at_least_590s_observed", "signal_pga_causality": "PASS", "stopped_for_overload": true, "target_fps": 30.0, "timing": {}, "tracker_source_timestamp_causality": "FAIL_FUTURE_CONTEXT", "tracker_source_timestamp_future_context_frames": [8, 15]}, "scientific_validity": "RESEARCH_ONLY", "signal_pga_causality": "PASS", "stress": {}, "tracker_source_timestamp_causality": "FAIL_FUTURE_CONTEXT"}`.
- Motion quality counts: `{}`.
- PGA rows included: `0`.
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
