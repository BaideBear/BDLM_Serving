# Gemma paid IBS Q64: no same-boundary neighbor seat

**Prelaunch no-go.** The original paid IBS Q64 planner yielded no eligible
one-seat exchange on the A0 frozen90 trace. Protecting mandatory writes, the
first MASK and the rightmost selected row leaves no legal exchange. A3 was
not started. There are no observer or treatment scores, changed-ID records,
or first divergences. No sampler, window, KV boundary or runtime was changed.

| Reconstructed on the paid arm | Frozen90 | Fixed six (7, 13, 25, 33, 55, 71) |
|---|---:|---:|
| Ordinary decoder calls | 2,765 | 155 |
| Ordinary calls at physical Q64 | 2,252 | 122 |
| Separate paid EOS calls | 70 | 4 |
| Upstream list longer than 64, ordinary calls / distinct requests | 91 / 58 | 4 / 3 |
| Outer hardcap dropped a MASK, calls / distinct requests | 7 / 7 | 1 / 1 |
| Dropped MASK had previously been sampled | 0 | 0 |
| Clean exchange before age or neighbor filtering | 0 | 0 |
| Previously sampled omitted MASK strictly inside the visible boundary, even ignoring upstream-list membership | 0 | 0 |

The last line tests **all** ordinary full-width calls, not just the outer
cap's trimming events. It is a larger candidate set than the requested
rule; it still has no recipient. For Q64, upstream retains at most
`max(1, 64 - non_mask_count)` MASKs in ascending position order. Therefore
an ordinary proposed list longer than 64 contains at most one MASK, the
first. The outer cap preserves it unless dirty rows fill the offer. All seven
observed omissions were this all-dirty case (fixed-six: #13); none permitted
a donor, even if a recipient had passed every other check.

Different estimand from the prior studies:

- [Warm fewer fail](../warm-fewer-fail-q32-a1/README.md) ranks failures in
  Q32 Live verified-EOS, not paid IBS.
- [Waiting seat](../waiting-seat-q64/README.md) selects an older warm MASK
  in a different Q64 Live planner with independent prefix allocation.
- [LLaDA IBS4 frontier swap](../../llada-ibs4-frontier-swap-partial19/README.md)
  gives a seat to the first MASK on an otherwise no-MASK call. This Gemma
  rule protects the first MASK so that test cannot supply its exposure.

## Evidence and reproduction

[Compact receipt](receipt.json) and [CPU replay](audit.py). The replay uses
the byte-pinned upstream NumPy planner path and capped-row function. It
reconstructs the ordered upstream list, the cap's output, the exact recorded
selected positions and decoder position IDs, paid EOS slices, visibility
and holes, admissions, dirty ledger and frontier. It distinguishes
previously sampled MASKs from rows never projected. The raw tape includes
sample positions and proposals, but not full per-row logits, confidence or
counterfactual output. None of those quantities is inferred. The zero
recipient result precedes any choice of age score, vocabulary filter or
neighborhood rank. This is an exposure audit, not a measured treatment.

Historical A0 paid baseline only, shown to identify the tapes: frozen90
76/90, `N=32782, D=2835, E=85, F=2920, Q=160484`; fixed six 6/6,
`N=1439, D=159, E=3, F=162, Q=8900`. `F` excludes prompt prefill and includes
generation encoder; `Q` counts decoder rows. These are not A3 controls.

On n164 (NumPy, no model or GPU):

```sh
root=/data1/project_data/gx-erqw-20261123
run="$root/results/gemma-ibs-certified-eos-a0-q64-20261128T110532Z"
python3 audit.py "$run/raw.log" "$run/summary.json" \
  "$root/archive/a800-gs-20261128/snapshot/work/gemma-ibs512-1127"
```
