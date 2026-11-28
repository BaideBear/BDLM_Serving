# Can the observed progress of the head justify opening the successor one call earlier?

This is a retrospective trigger audit, not a GPU intervention. Source: the
[sealed compact LLaDA Flow W2 Q64 control](../llada-flow-active-w2/README.md).
Only when one unfinished block has a legal successor, and there were two
consecutive single-open samples of this same head, consider
`f < 0.8 <= f + max(0, f - previous_f)` after the ordinary update. `f`
is the upstream measured acceptance ratio; no second forward or unselected
logits are used. The original `f >= 0.8` path is unchanged. Clear the
previous observation across changed heads or multi-block calls. EOS is a
diagnostic, not an extra treatment gate.

| Control cohort | Requests hitting rule | First hits on distinct heads | Next ordinary forward reaches original gate | Head finally terminates at EOS |
|---|---:|---:|---:|---:|
| Original 30 | 20/30 | 70 | 42 | 1 |
| Added 60 | 30/60 | 124 | 61 | 2 |
| Frozen 90 | 50/90 | 194 | 103 | 3 |

There were 216 positive ticks, including repeat hits on those same heads.
191 of the 194 first hits both had a nonterminal head and ultimately executed
a successor on the control. These are not 191 saved forwards. Earlier
successor calculations can be stale or harmful even if that successor
eventually contributes to output.

Exploratory check after establishing exposure: among ticks matched on the
**exact current progress** (151 flagged ticks in overlap), the next-tick
gate rate was 58.9% when flagged versus 46.2% in unflagged ticks weighted
to the flagged current-progress distribution. Both cohorts show the same
direction. This is a repeated-tick association, not a causal advantage over
a relaxed static gate. If the GPU treatment beats the original, compare a
static opening rule matched on opening frequency before crediting the trend.

[Compact data and registration hashes](exposure.json). Reproduce on n164
with `python3 tools/flow_admission_lead_exposure.py PATH_TO_SEALED_W2`.
Only paired fixed-budget GPU rollouts can establish actual F, Q, final
output quality or TPF. Keep the mandated-write accounting and existing
physical causal mask intact.
