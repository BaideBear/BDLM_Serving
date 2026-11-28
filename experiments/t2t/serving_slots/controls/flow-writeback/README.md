# What overlaps at a physical block boundary?

Readback of fixed-budget GPU runs on the original 30 plus additional 60 tasks. This is an execution audit, **not a new decoding score**. It tests whether a sparse window and a two-block window can use the same rule: a projected row pays for the *input version it actually reads*, and sampling that row can immediately create another outstanding version. A model-specific block closure need not be a separate forward.

## LLaDA Q64: write, project and commit

We replayed the open slots, absolute positions and sampled IDs recorded on n164 H4, including partial prompt blocks. Each actual ID change has exactly one destination: read as input on the next active projection, committed without such a read, or discarded when an earlier block ends the request. The total F/Q are independently reconciled against the 180 measured bills.

| Frozen 90, B32 | Original W2 | Compact W2 |
|---|---:|---:|
| M2T writes: reprojected / discarded at earlier EOS | 43,845 / 1,204 | 42,490 / 1,633 |
| T2T writes: reprojected / discarded at earlier EOS | 3,336 / 1 | 3,429 / 0 |
| Committed blocks containing any unpaid ID change | 0 / 1,389 | 0 / 1,357 |
| Forwards committing at least one block | 1,312 | 1,292 |
| Also wrote a different block that remained live past this forward | 1,077 | 1,075 |
| Also wrote a different block discarded on this forward | 38 | 52 |
| Commit forward with no token write in any slot | 197 | 165 |

Even a forward with no token changes has paid projection and informs stopping. The traces do not establish that one can skip it. “Remained live” means the subsequent decoder consumed the changed version, **not** that this ID survived every edit or appeared in the final answer. Nor does input-ID equality imply that a successor's entire multilayer KV equals a fresh recomputation on a new encoder context.

[Counters by cohort and source digests](../llada-flow-active-w2/payment.json) · [Row-version replay](../llada-flow-active-w2/audit_payment.py) · [Measured quality, TPF and TE](../llada-flow-active-w2/README.md).

## Can a smaller Q use genuine spare seats?

[What one free successor call per boundary could buy, and the next matched screen](boundary-sensitivity.md).

For each physical block on strict IBS, count its **first** ordinary call satisfying all these pre-forward conditions: no unresolved valid head row, head is resident or will be resident using its original query set, `0 < head_Q < q`, and there is generation room beyond the block. We then classify what actually happened in the original tape. The following columns require both a later promoted successor and **no subsequent head ID edit from the triggering call through promotion**. The runtime may use only the current canvas and must check for generated EOS; it cannot use the reference outcome or edit history. A block with an EOS or a head revision must be handled even when that speculation is unprofitable.

| Model | q | Candidate blocks before outcome check | Subsequently promoted without head edit | Distinct requests | Spare Q on these blocks |
|---|---:|---:|---:|---:|---:|
| LLaDA | 4 | 982 | 933 | 90 | 2,163 |
| LLaDA | 8 | 1,120 | 1,062 | 90 | 5,294 |
| LLaDA | 16 | 1,183 | 1,122 | 89 | 12,077 |
| LLaDA | 32 | 1,175 | 1,100 | 89 | 24,779 |
| Gemma | 32 | 175 | 92 | 53 | 2,106 |
| Gemma | 64 | 175 | 89 | 58 | 4,200 |
| Gemma | 128 | 164 | 78 | 57 | 7,708 |
| Gemma | 256 | 165 | 82 | 58 | 15,073 |

At all eight offers those seats correspond to a **proper subset** of the successor's first actual IBS row set. Not one entire next call is prepaid. For Gemma in particular, an encoder intervenes on ordinary promotion, so this counts row addresses only; it is **not evidence of reusable KV**. The Q8 LLaDA tape had 14 other promoted blocks with a subsequent head edit. A new executor cannot silently reuse successor state after a dependency changes.

Next single factor: compare stock strict IBS to overlapping the head's measured spare seats with the successor on fixed LLaDA Q8. Leave head rows and their ability to edit alone, use a physical block-causal mask, allow paid successor proposals to sample, track writes and invalidate on a head change. Use the same card for an unchanged arm, observation-only arm and treatment. Six frozen screening indices were specified before GPU treatment; all six have geometric exposure. Only real, paired requests can determine quality, TPF or TE. Do not extrapolate to an unmeasured offer or model.

[Eight tiers, cohorts and raw digests](close-overlap-exposure.json) · [Readback script](../../../../../tools/ibs_close_overlap.py). Reproduce on n164: `python3 tools/ibs_close_overlap.py PREFIX90`; `python3 experiments/t2t/serving_slots/controls/llada-flow-active-w2/audit_payment.py W2_RAW_DIR`. Neither loads a model. Generated answers and raw traces stay on n164.

The [optional shared decision](../../../../../bdlm_serving/methods/t2t/close_overlap.py) was checked separately at every recorded decoder call in these 720 requests. `PYTHONPATH=. python3 tools/ibs_close_overlap.py PREFIX90 --verify-rule` asserted identical row sets, including empty sets; the report matches the independent census byte-for-byte (`dd2275d4ed030a6cb097fdb75e6cd551bab1f951ecd4d25015ccdad184d17c6a`). Because partial-step EOS values are missing from the old tapes this comparison deliberately disables the online EOS veto in the *replay*, not in the runtime. It does not qualify continuation logits, sampler output, or cross-encoder reuse. The separate promotion predicate fails closed when the context version is absent or changed, the request ended, the head did not retire, or the head changed.
