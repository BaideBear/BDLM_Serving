# LLaDA Q4: first observed divergence on two H4 losses

2026-11-28, physical n164:H4. Diagnostic replay, not an ablation or a
quality estimate. Both cases were deliberately selected from H4 formal losses:
index 1 has a partial prompt block, index 67 has an aligned prompt. Same H4,
frozen input IDs, seed, scorer, Q4 physical limit and 2048 cap within each
pair. No other GPU was used for this diagnostic.

| Frozen index / task / seed | IBS vs shared prefill Q | First different MASK / actual write tick | Delivered EOS offset (IBS / shared) | N,F,Q (IBS / shared) |
|---|---:|---:|---:|---|
| 1, `gsm8k/251`, 17001 | 64 / 65 | 0 / 1 | 81 / 132 | 81,58,225 / 132,122,485 |
| 67, `test/precalculus/323.json`, 17067 | 128 / 128 | 3 / 3 | 1387 / none; cap 2048 | 1387,799,3105 / 2048,1888,7543 |

Ticks, absolute row positions and delivered offsets are zero-based. A sampled
MASK becoming a token is not the same as block retirement or EOS delivery.
Both IBS answers scored correct; both registered shared answers scored wrong.

- **#1:** IBS builds prompt row 64 on generation tick 0 and samples MASK
  rows 65-67. Shared already prefilled row 64 and samples 65-68. Both write
  `65:10840`. At tick 1 both pay row 65. IBS samples 66-68 and writes
  `66:4772, 67:4057`; shared samples cold rows 69-71 and writes `69:21`.
  First delivered ID difference is offset 2 (`4057` vs `1028`).
  First measured EOS write (tick:absolute row) is `50:146` vs `107:198`;
  final drain ticks are 57 and 121. The first shared EOS write is not the EOS
  ultimately delivered: EOS eligibility, its first write and delivery differ.
- **#67:** Prefill, first projected rows 128-131 and all four initial writes
  match. At tick 1 both pay exactly these four rows. IBS tests those decoded
  rows for an edit, shared drains only; neither writes. Tick 2 again matches
  on rows, MASKs and writes. At tick 3 both pay `132,133,134`, leaving one
  seat. IBS revisits the first unresolved MASK, resident 135, writing
  `135:9129`. Shared projects fresh MASK 136, writing `136:25487`, leaving
  135 unresolved. No prefix seat or successor row was involved in this
  fork. First delivered ID difference is offset 11 (`25` vs `1307`).
  IBS writes EOS ID `156892` at tick 787, row 1515, and finishes at tick 798.
  Shared never writes EOS and reaches the cap on tick 1887, last ID `92`.

Of the 90 original pairs, 88 have a partial prompt block and 47 of those
are IBS-correct/shared-incorrect; both aligned cases are also losses. This
is a descriptive split, not a prompt-prefill intervention. The two
instrumented cases are not two new data points. They identify the first
different actions, not which action caused an incorrect answer hundreds of
calls later. IBS also permits T2T on selected decoded generation rows,
including paid debt; row eligibility differs from actual writes. This test
does not isolate the effects of visible KV, editing, or EOS policy.

## Validation and readback

Original-IBS equality checks include full output IDs, every selected-row,
MASK, write and retirement/prefix event, EOS, charge and reference score.
Both H4 shared replays matched all fields used in the original formal
receipt check: status, input-ID digest, score, N/F/Q, physical peak and role
counts. Formal receipts did not contain a complete prior shared token tape;
shared IDs here are from the instrumented same-board replay, not a claimed
hash match to unavailable historical IDs. The IBS write ID shown is read
from the very next physical projection that pays that position's debt. The
shared write ID is the measured proposal actually selected for writing.

N164 root: `/data1/project_data/gx-erqw-20261123`. Raw trajectories remain in
`results/llada-q4-first-fork-7a38ffc-H4`. `compact.ndjson` SHA-256:
`d78034bba0e82553f0b522b5071d2ad50e4b741b8d80592aaf1a35713b09f9a1`.
Two pairs, four requests, one checkpoint load, zero missing. The release is
`7a38ffc`; base registration `a18f108aef1dbc607f3b21b2a4d169b04f5ae5dab01be400d015d07249d406f3`.
The following command reproduces the compact report without using the GPU:

```bash
R=/data1/project_data/gx-erqw-20261123
D="$R/results/llada-q4-first-fork-7a38ffc-H4"
cd "$R/runtime/formal90-7a38ffc/src"
set -o pipefail
PYTHONPATH="$PWD:$R/runtime/fixed-baseline-deps" \
  "$R/runtime/venv/bin/python" -B "$D/llada_q4_first_fork.py" report "$D" | cmp - "$D/compact.ndjson"
```

Next discriminating test, not launched: start at #67 tick 3 with identical
state and three paid writes. Exchange only cold 136 for resident 135 on that
one call; keep edits, thresholds, KV visibility policy, terminal handling
and later planner code fixed. Audit the common prefix through tick 2, the
measured write at the fork, survival of that write, then output and bills.
If selecting 135 does not restore its immediate IBS admission, ranking
alone is insufficient on that state. If it does but the score remains wrong,
one locally restored admission was insufficient to rescue the request.
