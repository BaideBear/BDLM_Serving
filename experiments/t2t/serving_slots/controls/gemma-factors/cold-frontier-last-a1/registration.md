# Cold frontier last, A800 physical A1 only

Evidence before launch: verified-EOS control Q32/Q64 frozen90 paid
6,874/3,949 F. Matched pre-EOS phase receipts show 170/172 heads,
2,550/1,204 cold F, 81,600/77,056 cold Q, zero idle cold seats, and
3,001/1,967 warm F. A perfect merger of every remaining separate closing
drain could save at most 165/149 F on these fixed historical trajectories;
that is an upper bound, not a proposed legal merger. Previously tested
half-pacing takes more cold calls per head and changes the cold clock count.

One factor: defer local [0,Q) first visits until all later tiles have been
visited. Fill every residual cold seat, changing neither first-visit count
nor capacity; retain baseline order within each cohort and physical call.
All debt still has first priority. The regular sampler gets only fresh
cold logits in its one existing call; pure drains do not consume a clock.
Warm scheduler, seed, entropy bound, re-noise, T2T, 1/8 clean prefix,
verified EOS, ordered terminal sweep and necessary encoder are unchanged.
No successor. Test whether enabling the leftmost tile to see more resident
in-block keys reduces outstanding frontier masks and warm F per head. It
could instead withhold useful prefix context from other cold rows. Changed
positions change RNG assignment and trajectories despite the matched seed.

Compare independent fixed Q32 and Q64 arms. For each, start with 0/10/20,
then complete original30 and the other 60; no tuning or post-anchor score
selection. Both arms on A1; alternate order. A1 off must match the complete
historical A7 verified-EOS-on receipt per case. Fail closed on replay or
counter mismatch, starvation, absence of actual first-call exposure, foreign
processes or hard cutoff 1795843443 (2026-11-28 13:24:03 +08). Register
one non-reusable model process identity (PID, start ticks, boot ID), policy,
source, checkpoint and inputs before inference. No holder or queued second
worker. Frozen gates are technical, not score-based.

For each tier: compare 90, original30, new60 separately; give score
rescues/losses, non-special N, all post-prefill decoder+encoder F, selected
history-reading decoder Q, canonical TE=sum(N)/sum(decoder Q), macro
TPF=mean(N/F), width and phase bills, heads and warm F/head. Full IDs/raw
tapes stay on A800. Recommendation requires higher macro TPF and canonical
TE on the union with no net correct loss in either subcohort. Even if that
passes, do not infer population noninferiority or a latency win.
