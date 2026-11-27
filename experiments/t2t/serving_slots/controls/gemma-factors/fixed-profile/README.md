# Fixed-budget charts

`verified-eos-*` is the current qualified opt-in profile. The Q32/64/128
terminal shortcut changes only the final ordered sweep, with exact output
parity on all 90 at each offer. Q256/512 reuse the previous fixed estimates
after exact three-case replay. Run the plotting module with `--eos-only` for
these three charts and the numeric comparison table without rewriting history.

`request-*` compares one qualified `dense_stop_joint_single512` request
profile at all five fixed offers to same-cohort controls. `previous-joint-*`
is its `dense_stop_joint_verify` predecessor; only Q512 differs. `metrics.csv`
also retains the individual ablations. `fixed-candidate-*` is the original
pre-API-qualification snapshot, not a separate experiment.

Regenerate current and predecessor charts with
`python -m experiments.t2t.serving_slots.plot_fixed_gemma_profile`.
Use `original30`, `new60`, or `frozen90` consistently across methods; the
table below each plot prints the numeric values. The native line represents
a single native Q256 run, not five native experiments. A missing Flow TE
means no audited historical-KV denominator was available.
