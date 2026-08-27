# 242 execution recovery

## Last confirmed state

- Host: `10.2.102.242`; project root:
  `/public/home/zhangbei/work_dir/zhangbei/cotracker_rk3588/VideoEEW`.
- Long run tmux session: `videoeew_pga_causal`.
- Output: `runs/20260825-pga-causal-v2`.
- Planned paired records: 81.
- Record 0 probe passed with causal polynomial features; the resumable all-paired
  run reused that cache.
- Last copied partial table contained at least 3 feature rows with
  `feature_version=videoeew-motion-v1` and `causal=1`. No failure row had been
  observed at that time.
- From 2026-08-26 through 2026-08-27, TCP port 22 accepted connections but did
  not emit an SSH banner. Therefore the final process/table state is unknown.

## Safe recovery sequence

Do not launch another batch before running these read-only checks:

```bash
tmux has-session -t videoeew_pga_causal
tail -n 80 runs/20260825-pga-causal-v2/batch.log
wc -l runs/20260825-pga-causal-v2/pga_features.csv
cat runs/20260825-pga-causal-v2/failed_records.csv
find runs/20260825-pga-causal-v2 -maxdepth 1 -type d -name 'record-*' | wc -l
```

If the session ended normally, first copy the compact CSV/log/JSON summaries
back to the Git checkout and run PGA evaluation v2. If it stopped, rerun the
same `--all-paired` command with the same output root; completed records are
loaded from cache only when their causal flag matches the config.

Before a new GPU job, synchronize the server code to the current GitHub
`main`. Keep the existing run directory; do not overwrite or delete it.

## Work still dependent on 242

1. Retrieve the complete causal feature/failure tables.
2. Run PGA evaluation v2 and compare causal versus frozen offline results.
3. Analyze reseed boundaries on completed record runs.
4. Run the >=1000-surrogate alignment null audit on the declared record set.
5. Run the real-CoTracker stress matrix.
6. Under controlled node load, run 30 FPS wall-clock replay for 10 minutes,
   then 30 minutes; characterize 50 FPS separately.

None of these is marked passed in the current partial audit bundle.
