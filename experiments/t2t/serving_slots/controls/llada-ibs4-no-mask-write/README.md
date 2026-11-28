# Native IBS4: writes on paid debt when no selected MASK can sample

Keep native T2T. On six frozen pairs, blocking this class of write changed
one trajectory, cost 88 more decoder forwards and 339 more selected rows,
and changed no binary scores. All differences in these totals come from
one MATH500 case, not from immediate projection savings or six independent
wins. Attribution only; not a 90-case estimate or a switching qualification.

## Design

The previous mixed-cold T2T study required a selected cold MASK; its
eligibility condition is disjoint from this one. MASK-order changes selected
rows. Neither is this experiment. `registration.json` was deposited before
running or reading these paired outcomes. Cases were chosen by index,
shard, cohort and dataset, excluding #1 and #67, not by length, correctness
or expected edit. Each pair used the same physical H4, prompt tokens, seed,
weights and frozen scorer, cap 2048 and batch one; arm order alternated.

Both arms execute the pinned hard-capped IBS4 implementation, not the shared
planner. The measured native top-1 proposal and confidence are computed
for the original selected rows. In `block` only, immediately before the
canvas write, remove the native edit at a projected incoming-debt position
if **none of the selected rows was an unresolved MASK**. MASKs may exist
elsewhere in the block. No new sampler call or logit read, no change to the
row planner, payments, other eligible writes, prefix update or termination
rules. On matched states the native predicate is unchanged (`confidence >
0.5`, old token differs). The existing writer sees the final edit mask, so
a withheld write creates no fictional debt; every actual write still does.

## Paired results

Each cell with two values is **keep / block**. Eligibility counts below
are row-visits, not unique tokens. The unexposed cases had eligible
positions but no native write meeting the veto predicate.

| Index | Correct | N | F | Decoder Q | Eligible visits | Native targets | First differing generation tick |
|---:|---:|---:|---:|---:|---:|---:|---|
| 7 | 1 / 1 | 212 / 212 | 135 / 135 | 523 / 523 | 85 / 85 | 0 / 0 | none |
| 13 | 1 / 1 | 358 / 376 | 230 / 318 | 885 / 1224 | 185 / 105 | 1 / 2 | 28 |
| 25 | 1 / 1 | 211 / 211 | 135 / 135 | 528 / 528 | 145 / 145 | 0 / 0 | none |
| 33 | 1 / 1 | 236 / 236 | 149 / 149 | 576 / 576 | 129 / 129 | 0 / 0 | none |
| 55 | 0 / 0 | 248 / 248 | 217 / 217 | 837 / 837 | 75 / 75 | 0 / 0 | none |
| 71 | 1 / 1 | 97 / 97 | 60 / 60 | 233 / 233 | 77 / 77 | 0 / 0 | none |

| Frozen stratum | Correct | N | F | Decoder Q | Macro TPF | Pooled TE |
|---|---:|---:|---:|---:|---:|---:|
| Original30: 3 pairs | 3 / 3 | 781 / 799 | 500 / 588 | 1936 / 2275 | 1.5633 / 1.4386 | 40.34% / 35.12% |
| New60: 3 pairs | 2 / 2 | 581 / 581 | 426 / 426 | 1646 / 1646 | 1.4478 / 1.4478 | 35.30% / 35.30% |
| All six | 5 / 5 | 1362 / 1380 | 926 / 1014 | 3582 / 3921 | 1.5055 / 1.4432 | 38.02% / 35.20% |

TPF is the mean of per-request N/F; TE is sum(N)/sum(history-reading
decoder Q), excluding prefill. There is no encoder here. All 12 requests
ended at EOS without failure or outstanding debt. All six keeps matched
archived prompt/output IDs, every planner/write event, progress status,
physical bills and score. Both arms checked actual Q on all 20 layers,
visibility, the planner's original selected rows and the incoming/new debt.

**First write difference (#13, `test/geometry/456.json`, seed 17013):**
28 earlier generation calls plus prefill matched. At tick 28 (block 2,
step 18), both saw the same rows and incoming debt `[93,94,95]`, KV
end 96, same physical inputs, MASK flags, proposals and confidence. No
other native edit was eligible on this call. On row 93, old ID `268`,
native ID `1435`, confidence `0.6817777753`: keep wrote it; block kept
`268`. Both paid Q=3, one physical forward that tick. Block later encountered
a second veto on its own changed trajectory; it is **not** a second
matched intervention. Both answers scored correct; delivered IDs differ.
The remaining five pairs matched in full observed inputs, proposals, events
and delivered IDs.

Across six keeps: 232 no-selected-MASK forwards; 212 had eligible debt
(696 row-visits); 1 target edit was made out of 5 total T2T writes. Across
six blocks: 222 such forwards; 202 had eligible debt (616 visits);
2 native targets were vetoed, with 5 actual T2T writes elsewhere. The
historical 342 counts edits on the original frozen90 trajectories, not
independent requests, or opportunities guaranteed on this six-case panel.

## Reproduce without a GPU

Raw paired receipts stayed in `n164:/data1/project_data/gx-erqw-20261123/results/llada-ibs4-no-mask-paid-write-h4-v1`.
`code.sha256` names files as deployed in that directory. Only compact
receipts, registration, hashes and resource handoff are archived here.

```bash
set -euo pipefail
R=/data1/project_data/gx-erqw-20261123
D=$R/results/llada-ibs4-no-mask-paid-write-h4-v1
(cd "$D" && sha256sum --check code.sha256)
cd "$R/runtime/formal90-7a38ffc/src"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD:$R/runtime/fixed-baseline-deps" \
    "$R/runtime/venv/bin/python" -B "$D/llada_ibs4_no_mask_write.py" report | cmp - "$D/compact.ndjson"
```

Rebuilt compact SHA-256: `3b0c61852d780d82e341b095fdab22c34f71578bfe4acdc236a910a1b6e2daeb`.
Worker PID 1589711 exited; the H4 guard verified the restored holder
PID 1608071, start ticks 2040465339, as the sole process on the registered GPU.
