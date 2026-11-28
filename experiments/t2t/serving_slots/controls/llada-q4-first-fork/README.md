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

## Follow-up: existing single-factor evidence; no new H4 inference

Row 135 is a diagnostic example, not a policy. The registered [MASK-order
probe](../shared-request/mask-order18.json) already tests earliest unresolved
row first, irrespective of residency, at unchanged within-block counts and
unchanged edit eligibility. [H0 Q4](../shared-request/mask-order-h0.json):
2/3 to 3/3 correct, macro TPF 1.4895 to 1.6036, F 464 to 527. This is not
a guaranteed saving. A different [one-seat Q4 intervention on 90
pairs](../llada-slot-allocation/c4-frontier-frozen90.json) improved 18 to
47 correct, but requires at least two selected cold MASKs: it does **not**
fire at the one-spare-seat fork in #67.

The other candidate already has actual A800 pairs: on a current-head call
already sampling a cold MASK with all debt payable, enable native T2T only on
paid, resident, decoded dirty rows. Keep the same selected rows and tick at
the intervention, use real current logits, and bill writes as new debt.
The following are separate historical head-only controls, not the H4 unified
policy or replications on held-out inputs. Both arms passed physical widths
at 20 layers, the write/debt ledger and the matched-state first-exposure checks.

| Historical pairs | Exposed calls / eligible row visits | Actual edits / revokes there | Equal output IDs | Correct off/on | Macro TPF off/on | Total F off/on | Decoder Q off/on |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4, 0/10/20 | 236 / 276 | 0 / 0 | 3/3 | 0/0 out of 3 | 1.14346 / 1.14346 | 782 / 782 | 2946 / 2946 |
| Q8, original30 | 2164 / 2681 | 7 / 0 | 26/30 | 11/11 out of 30 | 1.63460 / 1.63413 | 16220 / 16267 | 113469 / 113797 |

At Q8 there was one rescue and one loss. No request failed in either pair
set. Q4 used zero prefix share and Q8 used 1/8. Neither tested T2T during
a pure drain or on warm-MASK-only calls. In the independent, capped IBS4
frozen90 trace, 441 of 455 edits touched paid debt. Of those 441, **66**
occurred with a cold MASK, **33** with warm MASKs only, and **342** with
no selected MASK. Only **17** of the latter 342 were on calls where every
physical row was debt; the other 325 also projected non-debt rows. Counts
include completed calls for the six failed IBS requests. These are facts
about where IBS edits, not evidence that applying them rescues the unified
output. The no-MASK calls are not all interchangeable with unified drains:
the latter do not advance the sampler clock. No drain intervention is claimed.

Readback on n164, stdlib only (no inference or raw transfer):

```bash
ssh n164 /data1/project_data/gx-erqw-20261123/runtime/venv/bin/python -B - \
  < tools/llada_q4_paid_edit_readback.py
```

Archived source: `archive/a800-gs-20261128/snapshot/results/` under the
n164 root above. Paired row-set SHA-256 (hash of concatenated per-file digests
in index order): Q4 `4b39818bdae84f77646014354faed945c56d867517fcafab5e456929fbd660df`;
Q8 `6f9bf699a8c0e194b6dddd3a785e5ba0e9c5f0a1992c43e402748f4ca1e2a9da`.
