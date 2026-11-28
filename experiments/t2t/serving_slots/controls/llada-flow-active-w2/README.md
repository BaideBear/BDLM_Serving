# LLaDA 2.1: dense vs active Flow W2 on H4

Single-request reference experiment, not a unified planner, policy promotion,
or official fused-backend speed benchmark. B=32, offer=64 throughout. No
changes to the default FlowBlock control or public endpoint.

## Sealed result

All 90 fresh dense generations exactly reproduced the frozen references.
Both arms were freshly run on H4: 180 requests, 90 paired identities, no
missing results. The six qualification pairs are included just once.
Independent raw-to-compact recomputation, including scoring, was byte-identical
(`report.verified`). Local files are compact only.

| Cases | Correct, dense / compact | Generation F | Generation Q | Macro TPF | Pooled TE | Equal output IDs |
|---|---:|---:|---:|---:|---:|---:|
| Original 30 | 23 / 27 | 1,768 / 1,704 | 113,152 / 75,872 | 9.721 / 9.815 | .1181 / .1684 | 6/30 |
| Additional 60 | 51 / 51 | 3,826 / 3,854 | 244,864 / 166,080 | 10.149 / 10.075 | .1151 / .1668 | 13/60 |
| Frozen 90 | 74 / 78 | 5,594 / 5,558 | 358,016 / 241,952 | 10.006 / 9.988 | .1161 / .1673 | 19/90 |

85/90 requests spent less Q. Of the **116,064** net saved decoder rows
(32.42%), 113,760 come from the 3,555 measured single-slot compact calls;
2,304 are attributable to 36 fewer calls on the changed trajectories.
N fell from 41,553 to 40,479. Eight scores improved and four worsened;
the additional 60 had three wins and three losses. This is an efficient
reference point, not a token-equivalent optimization or evidence of higher
TPF. The trajectories were completely decision-identical on only 11/90.

`audit_raw.py` independently reconstructed all 180 responses, including 88
partial prompts per arm. Dense/compact had 84/86 in-order EOS commits and
78/92 locally finished but still projected slot occurrences. There were
3,337/3,429 actual T2T token writes. Different totals are trajectory effects,
not removed obligations. `summary.json` gives all split denominators;
`compact.ndjson` gives one record per pair.

## Contrast

The frozen dense control is the extracted official FlowBlock packed loop
(`8f730a2173140792a4324736efdcba27a2bdee75`), native LLaDA 2.1 greedy
sampler, mask/edit `.5/.5`, post-edit `16`, and `acceptance_ratio >= .8`
to open the successor. The archive contained dense W1/W2; the LLaDA
progress80, prefix-cache/retirement and fixed-admission experiments were
not this pair. Gemma active W2 is the analogous question, not a control
for LLaDA.

Both arms run that loop on the same board. An instrumentation callback
records its real commit/right pointers. For one open block only, the opt-in
adapter passes its 32 rows to the transformer and writes those 32 real KV
rows. For two open blocks it uses the stock 64-row backend, including
locally finished but not yet committed slots. The virtual 64-position
gather/embedding lookup is retained. The unprojected synthetic key columns
are masked padding, not cached logits or features for a future block.

For slot `s`, the upstream mask exposes only the committed prefix
`[0,head)` and fresh tail `[tail,tail+32*(s+1))`. It excludes the persistent
future and all later scratch slots. A block can enter the prefix only
through the open window. An opened successor has its own measured KV and
logits in the same forward, before its first sample. These are statements
about dataflow, not bitwise equivalence of different tensor shapes. The
experiment does not isolate a numerical kernel responsible for divergence.

## Registration and audit

`registration.json` was placed on n164 before inference. Six fixed indices
`7,13,25,33,55,71` cover all three datasets in each cohort; the remaining
84 run once in ascending order only after qualification. Arms alternate
execution order. Inputs, generation cap 2048, weights, backend, scorer,
physical H800 GPU 4, decode parameters and source per arm are fixed.
`source.sha256` seals the GPU scripts with their n164 deployment basenames.

The physical bill is measured independently at **all 20 fused QKV
projections** and cross-checked with every batch and backend event. Prefill
F=1/request and its Q are separate; LLaDA has no encoder. N excludes special
tokens, F is the count of generation forwards, TPF is mean per-request N/F,
pooled TE is total N / total **history-reading decoder Q**. Token IDs include
the first generated EOS. Full keys are retained on both arms; Q savings
are not claims of equal percentage wall time or FLOPs.

Remote evidence directory: `n164:/data1/project_data/gx-erqw-20261123/results/llada-flow-active-w2-h4-v1/`.
Only compact outputs belong in this control. Recompute locally without a GPU
with `python3 summarize.py` from this directory. `audit_raw.py DIRECTORY` runs
on n164: it reconstructs opening, in-order commit, immutable prompt pieces,
actual M2T/T2T writes, sticky finished slots, and capped/EOS-trimmed output
from each arm's slot decisions. `llada_flow_active_w2_study.py report`
re-scores the n164 raw files and re-derives the compact pairs; it does not
load the model. A first fork's `tick` and `slot` are zero-based; a
`first_token` entry is `[within_slot_offset, dense_token, compact_token]`.

Signed savings split exactly into `32 * compact_single_ticks` on the
visited trajectory plus `64 * (dense_F - compact_F)` from any change in its
length. The second term can be negative. Neither correctness nor a shorter
trajectory follows from the first term.
