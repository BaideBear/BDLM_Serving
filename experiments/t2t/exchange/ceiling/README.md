# Physical exchange ceiling on native LLaDA 2.1

CPU replay of the 30 verified frozen native B32 traces, on n164. Only the
1,170 post-bootstrap denoise states with a remaining editable MASK are scored.
Prompt positions, the first denoise of each block, full verification, and
commit are outside the experiment. An edit is a changed non-MASK to a different
non-MASK. The 191 targets are strict edits on non-fresh positions; another 339
strict edits were on mandatory fresh positions and are common to all arms.

The zero-reserve selector fixes the physical width at each state, including
fresh overflow. All fresh positions and the first originally selected MASK
(if any) are protected. Only non-fresh resolved positions outside the base can
enter. A lost admission is an evicted **base** MASK that the native transition
actually resolves. A different evicted MASK is a real displacement but costs
zero immediate admissions. This does not preserve native admissions that were
outside the base to start with (the base covers 610 / 1,616 / 3,098 of 4,776 at
Q4 / Q8 / Q16).

## Exact frontier

An entering miss can be undone without losing any hits. A swap evicting a base
hit can likewise be undone without reducing coverage. The oracle therefore
keeps all base hits and matches outside hits to eligible victims in order:
old misses, non-admitting MASKs, admitting MASKs. It stops at the cap (1/1/2
new seats per state, or none), available hits/victims, or admission budget.
The ordering minimizes admissions lost, then swaps, then raw MASK displacement
at any given optimal hit count.

The two endpoints encode the whole integer Pareto frontier for each request,
cap and Q. If `H0` and `Hmax` are the zero-loss and unconstrained-loss hit totals,
the maximum total hits at an admission budget of `b` for that request are
`min(Hmax, H0 + b)`: every hit beyond `H0` adds one admission lost, and states
are independent under teacher forcing. Subtract `base_old_hits` to get net
new coverage; divide that by `residual_old_targets` when it is nonzero to get
residual coverage fraction.

## Measured endpoints

Each cell is **total old-edit hits / lost admissions / all displaced MASK rows**.

| Q | Zero reserve | Frozen edit4 | Capped zero loss | Capped max hits | Uncapped zero loss | Uncapped max hits |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0/0/0 | 3/3/9 | 54/0/41 | 67/13/54 | 60/0/45 | 77/17/62 |
| 8 | 0/0/0 | 14/13/77 | 100/0/65 | 112/12/77 | 139/0/91 | 159/20/111 |
| 16 | 7/0/0 | 41/24/208 | 155/0/43 | 160/5/48 | 184/0/53 | 189/5/58 |

Equal physical rows across arms: 9,285 / 12,243 / 19,800 (including 4,605 /
2,893 / 1,136 fresh overflow rows). Q16 edit4 adds 37 hits but drops 3 base
hits, for 34 net new; its 208 displaced MASKs include 184 non-admitting ones.
The capped zero-loss oracle beats frozen edit4 in old hits on 11 / 15 / 17
requests, and never gets fewer hits on these traces.

At the current caps, the larger opportunity is identifying good entrants
and low-cost victims: allowing admission loss adds only 13 / 12 / 5 hits to
the capped zero-loss ceiling. Relaxing the cap at zero loss adds 6 / 39 / 29;
Q8 has the largest absolute cap headroom. A useful next step is to evaluate
admission protection and edit ranking at the current caps before raising them.
Zero loss does not mean harmless: delaying a MASK that does not resolve on this
step can affect later steps. The labels are hindsight on the full native path,
not events guaranteed to reproduce with sparse Q. No rollout, latency, output
quality, or generalization claim follows from these results.

## Reproduce

From the n164 repo at `dad7d559e6ede420fcaaa9f5796c9ccdf5bca929`, with the
existing sibling trace directory and checked-in frozen exchange rates:

```sh
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -B -m tools.exchange_ceiling \
  --dir ../t2t-native-1125/llada21 --rates experiments/t2t/exchange/rates.json \
  --out experiments/t2t/exchange/ceiling
/usr/bin/python3 -B -m unittest discover -s tests -p 'test_exchange*.py'
```

[`requests.csv`](requests.csv) has the 18 rows per request;
[`summary.csv`](summary.csv) sums them. Frozen edit4 hit, admission, and Q
counts were checked against the existing native replay totals.

## Fixed gate: which decision is limiting?

Hold the frozen 4x gate, the number of swaps **in each state**, and the
number of old/MASK victims fixed. The candidate-only oracle changes only which
outside old rows enter. The victim-only oracle changes only which eligible rows
of each original victim type leave; it still protects mandatory fresh and the
first selected MASK. Ties prefer the original choice. The `locked_mask_rate`
alternative is logit-free: within the same gate, take the least likely-to-admit
removable MASK using the existing frozen 24-request rate table. It is a native
replay ablation, not a new sparse model run.

| Q | Frozen 4x | Candidate oracle only | Victim oracle only | Both, gate fixed | Cheapest predicted MASK victim |
|---:|---:|---:|---:|---:|---:|
| 4 | 3 / 3 | 17 / 3 | 3 / 3 | 17 / 3 | 3 / 4 |
| 8 | 14 / 13 | 64 / 13 | 14 / 7 | 64 / 7 | 14 / 14 |
| 16 | 41 / 24 | 138 / 24 | 44 / 14 | 141 / 14 | 41 / 24 |

Cells are **total old-edit hits / native base admissions lost**. Learning who
enters buys the largest direct edit opportunity at all three widths. The fixed
gate still misses useful times to exchange (e.g. Q4 capped zero-loss 54, vs
17 when both choices are optimal but the old gate is fixed). Simply replacing
the last eligible MASK with the lowest score in the existing table did not
reduce lost admissions at any budget. These are immediate native transitions;
no end-to-end gains are implied. Including the four extra locked variants,
`requests.csv` has 30 configurations per request.
