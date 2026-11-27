# Fixed-budget evaluation: choosing a real batch

This opt-in calibration is separate from the running batch-1 ablation. Never
replace sealed cases with pilot outputs. Each resident request keeps its own
offer, RNG, attention mask, KV, termination and edit state. There is no
serving scheduler or mixed-budget policy in this experiment.

Group decoder calls by actual selected width, not by nominal offer. One physical
call on `n` lanes of padded width `w` costs `n*w` projected Q rows; log both this
number and the sum of valid widths. Aim near 512 rows without unnecessarily
padding dissimilar widths. An indivisible oversize call runs alone. Gemma's
native encoder calls run separately. A batch is a cohort, so a long last
request can lower the mean rows far below 512.

`packed_pilot.py MODEL CARD NEW_DIR` uses the same cluster GPU handoff as the
formal run. Compare warm timings, peak reserved CUDA memory and estimated
device headroom. Only admit a candidate that reproduces the sealed output token
IDs, per-step selected positions, per-request forwards and total useful Q
for every calibration case. Stop growing on OOM or low headroom (less than
8% of device memory or 4 GiB, whichever is larger). Never replay partially
modified KV; any smaller pilot must start with fresh requests.

For the first Gemma Q128 arm 00 trial (four identical tasks per batch setting):

| Resident requests | Decoder rows, mean / max | Wall seconds | Peak reserved GiB | Exact trajectories |
|---:|---:|---:|---:|---:|
| 1 | 105.7 / 128 | 19.57 | 49.40 | 4/4 |
| 2 | 159.1 / 256 | 26.94 | 50.56 | 0/4 |
| 4 | 207.5 / 512 | 29.18 | 52.87 | 0/4 |

No OOM. One of four final token sequences matched at each larger setting,
but no entire trajectory matched. These runs do not establish a quality
difference, numerical root cause, or speed at equal work. Keep batch 1 for
this setting pending further qualification. Do not infer other offers or
models from this pilot.

The formal CSV `experiments/t2t/serving_slots/controls/block-ablation-gemma-v1.csv`
is unrelated to these timings. Regenerate it with:

```sh
python summarize_sealed_ablation.py gemma "$D/block-ablation-gemma-v1/complete.json" "$D/native-gemma.csv"
```

For each frozen cohort, the main TPF column (`macro_tpf`) is the unweighted
mean of per-request TPF, as in the native reference. `pooled_tpf` is
`sum(N)/sum(F)`. Token efficiency is `sum(N)/sum(Q)`.
