# Experiment placement and storage (2026-11-28)

After the current fixed-budget confirmation wave, the project may use **at most four A800s (physical 0, 1, 2, 3)** and the two H800s on n164 (physical 0, 4). Existing bounded jobs may finish on their original devices. A800s are for experiments only. Do not launch a holder or idle reservation on an A800. Do not extend a wave beyond its registered cutoff. On H800 use the existing PID-checked holder handoff on the assigned card only.

Only weights, source and required code dependencies belong on persistent A800 storage. Do not redirect stdout/stderr to a file on the A800 or retain result directories, figures, traces, build caches or temporary copies there. Stream logs/results to n164. For a legacy evaluator requiring seekable files, use a per-run private directory in `/dev/shm` with a size limit/budget and transfer immediately to n164; release it after a successful receipt. Do not start an evaluator with hard-coded persistent output paths. A shared system directory, another user's files or a model checkpoint is never cleanup scope.

The durable historical root is **n164**:

```text
/data1/project_data/gx-erqw-20261123/archive/a800-gs-20261128/snapshot
```

For instance, a report that originally cites `/gs/results/single512-stop-1128` now refers to `snapshot/results/single512-stop-1128` under this archive. Pass that path to the audit tool, not the old A800 path. The raw data are not stored on the Mac or in Git. The sibling `cold-destination.ndjson` and `cold-receipt.json` record the destination file hashes and the first verified removal. Later sweeps retain separate receipts. A row may be removed at source only after the matching n164 copy has been read and hashed; active inputs, open files, and changed files are excluded.
