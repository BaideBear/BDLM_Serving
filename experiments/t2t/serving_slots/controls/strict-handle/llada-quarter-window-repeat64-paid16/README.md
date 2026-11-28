# LLaDA `quarter_window_repeat64_paid16` qualification

Opt-in switchable profile built on `quarter_window_repeat64`. It keeps the Q64
paid-repeat / warm-head behaviour and adds the already-measured Q16 paid
waiting warm-MASK sampling on mixed cold calls. The Q16 addition uses only rows
already projected in the same call: no extra Q, no clean prefix, no head dirty
row and no new sampling tick. Q4/Q8 are unchanged.

## Functional GPU qualification (A800 GPU4)

- profile: `quarter_window_repeat64_paid16`
- 15 fixed cases (offers 4/8/16/32/64 on frozen 0/10/20): **15/15 exact
  replays** against the appropriate fixed controls (`fixed16` includes the
  paid Q16 arm).
- `fixed16` paid exposure: `paid_mask_rows=273`, `window_F=121` over the three
  requests.
- `switch25`: all **25 directed pairs** exercised; 101 downshifts with debt and
  50 parked rows paid; `paid_mask_rows=5`, `discard_on_downshift=114`,
  `repeat_Q=17`.
- `park` (Q64→Q4→Q64): `64>4` and `4>64` observed; 3 debt downshifts and 6
  parked rows paid; `parked_dirty_Q=10`.
- `park16` (Q64→Q16→Q64): `64>16` and `16>64` observed; 3 debt downshifts and
  22 parked rows paid; `parked_dirty_Q=22`. This particular parked boundary did
  not hit a cold Q16 call with a waiting warm MASK, so its paid-mask exposure is
  zero; the real mixed paid boundary is exercised in `switch25`.
- Q64 path: `fixed64` exact replay 3/3 with `repeat_Q=64` and
  `warm_sample_rows=88`; the new class does not alter the Q64 plan.

## Paths

- valid run: `/gs/results/qualify-llada-paid16-api-1128-v2` (24 compact
  receipts; source data stays on the cluster)
- aggregate: `report.json` here; per-request receipts in `receipts/`
- excluded first attempt: `/gs/results/qualify-llada-paid16-api-1128`
  (all 24 receipts were generated, but the report rejected it because the
  `park16` suite did not increment `parked_dirty_Q`; this was a counter bug,
  not a model or qualification failure)
- code: `bdlm_serving/methods/t2t/serving_profiles.py`,
  `experiments/t2t/serving_slots/llada_window_sample.py`
  (`switchable_paid_class`), `strict_handle_qualify.py`,
  `strict_handle_report.py`, `tools/qualify_llada_quarter_a800.sh`
- tests: `tests/test_llada_paid16_profile.py`,
  `tests/test_strict_handle_report.py`

The compact contains no output IDs; only output digests, bills, directed pairs
and policy counters.

The reporter requires the `park16` suite for this profile (even if every
`park16` receipt is deleted) and rejects `park16` on every other profile. It
also requires a genuine `paid_mask_rows` exposure in `switch25`; the parked
Q64→Q16 boundary itself may legitimately have zero paid rows.

Rebuild the aggregate from the archived receipts without inference:

```sh
PYTHONPATH=. python -m experiments.t2t.serving_slots.strict_handle_report \
  llada experiments/t2t/serving_slots/controls/strict-handle/llada-quarter-window-repeat64-paid16/receipts \
  benchmarks/frozen90/manifest.json
```
