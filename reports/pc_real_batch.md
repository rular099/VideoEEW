# Real-video PC batch and PGA research baseline

## Data and scope

The local manifest contains 94 record IDs: 79 directly paired video/sensor
records, 2 paired records with split sensor files, and 13 videos without a
strong-motion file. Ten records spanning approximately 68–412 gal were chosen
for the first full CoTracker pass: 6, 16, 26, 48, 61, 65, 69, 71, 84 and 85.
The strong-motion horizontal-vector peak is the PGA truth.

No geometric scale, camera identity, mounting metadata or validated site label
is available. All rows are `UNCALIBRATED`; deployment PGA is rejected.

## Tracking and common/local motion

The 242 V100S batch decoded 13,718 frames and produced 1,709 tracker blocks.
All frames passed the current motion quality gate. Across the batch, tracker
latency was mean 180.5 ms, p50 160.2 ms, p95 247.5 ms, p99 445.5 ms and maximum
6.015 s. A live process snapshot during record 85 showed peak RSS 1,267,432 kB,
current RSS 1,129,916 kB and GPU memory 3,906 MiB.

The server load average was about 98 during this experiment. Therefore the
p95 happens to fit the 266.7 ms 30 FPS block-arrival budget, but p99/max do not,
and these numbers are not a controlled realtime acceptance result.

## Video/sensor candidate alignment

The video signal was compared against every fully containing sensor segment in
both acceleration-proxy and band-limited displacement domains. Displacement
was preferred for 9/10 records. Correlation was at least 0.4 for 7 records;
records 6, 16 and 85 were rejected from model training. The best correlation
ranged from 0.186 to 0.736. Because each result is the maximum over many tested
offsets, it is an exploratory candidate and not proof of clock synchronization.

## Group-cross-validated PGA baseline

The configured quality filter retained 7 records and used five-fold splitting
by record/event group. The predeclared primary model is the single-coefficient
visual-acceleration baseline. Its out-of-fold results were:

- MAE 94.4 gal; RMSE 118.1 gal;
- log-PGA MAE 0.493;
- median multiplicative error 1.366;
- within factor 1.5: 57.1%; within factor 2: 71.4%;
- Pearson 0.258; Spearman 0.464.

Ridge and Huber were also reported, but neither demonstrated meaningful
correlation on seven rows. Results are offline, same-setup, research-only and
insufficient for a deployable PGA model. The saved model has
`requires_scale=true`; runtime use without a valid scale raises an error.

Compact evidence is under `runs/20260823-pga-real/` and
`runs/20260823-pga-evaluation/`.
