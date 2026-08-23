# PC bounded-stream stability

Status: `PASS_FOR_BUFFER_AND_QUEUE_SCOPE`

The deterministic fast-forward test in
`runs/20260823-streaming-stability/metrics.json` advanced 18,000 frames, equal
to 10 minutes at 30 FPS, through the fixed 16-frame ring buffer and a bounded
two-block queue.

## Measured result

- final frame-buffer size: 16 frames;
- queue capacity / maximum observed depth: 2 / 2 blocks;
- rejected enqueue attempts: 72, all explicitly counted;
- initial / final RSS: 43.535 / 44.125 MB;
- post-warm-up RSS slope: 0.00185 MB/min;
- silent frame or block drops: none in this test harness.

This is a fast-forward data-structure test, not a wall-clock real-time tracker
test. CoTracker tensor history is bounded separately by the logged periodic
reseed (528-frame maximum with the baseline configuration). The V100S rendered
benchmarks satisfy the 266.7 ms eight-frame arrival budget, but the current
real-video batch ran on a highly saturated shared CPU and therefore is not a
controlled real-time acceptance measurement.

## Not yet accepted

- 30-minute wall-clock PC camera/file feed;
- RK3588 10- and 30-minute tests;
- board temperature, throttling, NPU use and system-memory traces.

These remain blocked for the board until an RK3588 device is accessible. No
board result is inferred from the PC fast-forward test.
