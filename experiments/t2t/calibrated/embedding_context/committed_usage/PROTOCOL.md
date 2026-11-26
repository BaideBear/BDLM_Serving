# Does local usage in committed output predict old-row strict edits?

LLaDA B32; freeze before evaluation. Start with the previous 28-dimensional position +
input-embedding context ranker. Add exactly six inputs: log1p observed counts (capped
at 16) for the current accepted token and for *any alternative* token matching
its immediate left neighbor, right neighbor, or both. Count only the most recent
256 already committed generated tokens of the same request, never the current block
or raw prompt as training text. The current row's flanking IDs come from the
visible canvas (including visible prompt suffix if present); masks and absent
neighbors give no evidence. Do not use a draft, logits, KV, external corpus, task
label or a known future output. Rebuild the bounded hash tables only at commit.

One unchanged logistic fit (L2=2) on the independent calibration24; one frozen30
native replay. Original comparator frozen; verify its hit/seat/target counts
against archive. Primary: Q4/E1, Q8/E1, Q16/E2, number of reserved spots hitting
a strict edit. A follow-up sparse run is justified if at least two gain and
none lose relative to the embedding ranker. Top-1/4/8/16 are descriptive only.
Never tune the lookback, templates or stopping criterion on frozen30. Measure
incremental CPU feature cost separately; none of these native measurements
establish online quality, TPF or same-path sparse outcomes.
