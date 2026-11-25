# Frozen Q8 physical exchange: separate timing from placement

Original frozen30 identity, seeds, 2,048 generation cap, native B32 on LLaDA 2.1.
Both job shards must reproduce the full-width reference 30/30. All four arms
use the already-frozen input-context ranker and identical nominal Q8, at most
one explicit old seat. No inference from sampled successes or changes to prompts.

Start from zero reserve (every fresh, including previous T2T; earliest MASKs,
then earliest neutral old rows). Preserve its complete physical width, including
fresh overflow. If no seat remains, do not displace fresh. Slack-only can only
replace the last neutral old row. The gate arm does the same if there is slack;
otherwise it can replace the last selected MASK only when at least two fit and
the maximum old score reaches `log(.05/.95)`. Five percent is inherited from
the earlier positional study, not a threshold retuned on frozen30. It is a
model score cutoff, not assumed to be a calibrated live edit probability.

Rank every eligible old row, including neutral backfill. If the best is already
selected, make no exchange in either paired arm. Otherwise the ranked arm
swaps in the best row, and the random arm uses the same opportunity and victim
on its own current state, drawing uniformly from old rows outside the base.
Random seed is `context-swap:{request_seed}:{block}:{hard_clock}`. No global RNG,
logits, live KV or extra forward for selection. A closed slack-only gate needs
no scoring. The observation is recorded on every actually selected step.

Primary feasibility for each ranked rule: correctness strictly above zero
reserve and unrounded macro TPF at least as high. Attribution is a separate
comparison against its own randomized placement (same criteria); also report
fixed context, original IBS/OBS, efficiency, physical Q/F, paired uncertainty,
swap types, admissions cost and strict hits. Report wins and losses even when
totals tie. No strategy substitution or picking a better seed after scoring.
Native replay supplies independent same-state coverage and immediate admission
loss, not counterfactual sparse answers. Distinguish a frozen30 pass from
confirmation on an independent sample or seed.
