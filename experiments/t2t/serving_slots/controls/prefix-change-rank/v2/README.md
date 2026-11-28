# Q8 spare prefix projections: six paired requests, sealed

Strict LLaDA 2.1 hard-capped IBS8 on physical n164 H4. Same six frozen inputs,
seeds, prompt, original sampling, and paid frozen-EOS control. Right and change
spend the same number of spare seats **on the same state**; after divergence
there is no matched-trajectory count requirement. All injected rows are real
KV projections, excluded from acceptance and native T2T. No 90 requested or run.

## Recovery

v1 remains intact. Its saved `07-none.json` was reused by SHA-256
`4c9aaab631241e268aa5b1e0b5e8f90af603da24f8967b18f5df2bdd82ac613a`.
At zero-based ordinary tick 2 the paid rows were 71..78. The real helper
returned empty candidate/selection vectors at zero slack; the old auditor still
enumerated 68..70 and their scores. Both sets of picks were empty and the
paid inputs, clock and sampling mask matched. See `diagnosis.json`.

Before releasing the v1 holder, `preflight.py` used the original #07 bytes,
five independent shape fixtures on the real selector, four positive-slack
corruption checks, and the original frozen-EOS predicate on all 12 historical
stock/right trajectories. Recomputed exposure bytes matched the sealed v1 hash.
All 89 ordinary #07 none calls matched history; first certified EOS 273,
no terminal debt, N/F/Q 205/89/661, prefill 64 separately. Stock/right
historical archives precede paid EOS: their original full-block end bills were
**not** substituted for the paid stop. Ordinary events, input IDs, actual rows,
output IDs and any suffix payment were reconciled to the historical witnesses.

Study registration, sampler/adapter, ranker, exposure, fixed source and board
unchanged. New worker 2281804 executed 17 requests. The final CPU report matched
the live compact bytes; `seal.py` separately reconciled each write to the real
input ID at payment, original certificates and 20-layer physical widths.
Only the genuine restored H4 holder remained: PID 2302548, start ticks
2041372273, UUID `GPU-d3bdc2bc-d19d-e98f-ac79-ab9c6e887e0e`.

## Results

| Arm | N | Decoder F | Selected decoder Q | Prefix Q | Macro TPF | Pooled TE | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| None | 1346 | 531 | 3943 | 0 | 2.62169 | .34136 | 6/6 |
| Rightmost | 1326 | 541 | 4171 | 152 | 2.53509 | .31791 | 6/6 |
| Dependency change | 1477 | 610 | 4646 | 203 | 2.54426 | .31791 | 6/6 |

N excludes special tokens. Macro TPF is the mean request N/F. Pooled TE is
sum(N)/sum(selected history-reading decoder Q), including final paid decoder
calls, excluding prefill. All generation calls are at most Q8.
Original three macro TPF (none/right/change): 2.679/2.567/2.555; added three:
2.565/2.503/2.533. All groups were 3/3 on these development samples.

| Case | First differing tick (right/change) | Change minus right N | Delta F | Delta Q | Same output IDs? |
|---:|---:|---:|---:|---:|---|
| 7 | 14 | -9 | -6 | -46 | no |
| 13 | 14 | +141 | +76 | +561 | no |
| 25 | 7 | 0 | -3 | -21 | yes |
| 33 | 7 | -5 | 0 | -8 | no |
| 55 | 9 | +24 | +2 | -11 | no |
| 71 | 14 | 0 | 0 | 0 | yes |

The mechanism was exposed on 6/6; e.g. #07 used 5 original + 2 prefix rows
in both arms, right [89,90] vs change [70,78]. Four final ID streams changed
but no score flipped. #13 accounts for more than the net +69 F vs right.
#33 none required a real final forward on [308,309,310,311]. A spare seat
does not make its projection free and the common-state count does not guarantee
future F or Q savings. No promotion or expansion is supported by this screen.

## Recheck without a model

On n164 the two sibling result directories are
`/data1/project_data/gx-erqw-20261123/results/llada-ibs-q8-prefix-change-rank-h4-six-v1`
and `.../llada-ibs-q8-prefix-change-rank-h4-six-v2`. Raw files stay there.
The deployed v2 folder includes exact copies of the v1 `rank.py`, `adapter.py`,
`test_rank.py`, `registration.json` and `exposure.json`, plus this folder's
overrides. `study.py report` reads the one retained v1 receipt and 17 fresh v2
receipts and never loads the checkpoint. In the deployed v2 directory:

```bash
set -euo pipefail
T=/data1/project_data/gx-erqw-20261123/runtime
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$T/formal90-7a38ffc/src:$T/fixed-baseline-deps"
sha256sum -c source.sha256
sha256sum -c report.verified
"$T/venv/bin/python" -B preflight.py | cmp - preflight.json
"$T/venv/bin/python" -B diagnosis.py | cmp - diagnosis.json
"$T/venv/bin/python" -B study.py report | cmp - compact.ndjson
"$T/venv/bin/python" -B seal.py | cmp - summary.json
```
