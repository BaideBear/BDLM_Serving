# Verified-EOS Live Gemma Q64: right-edge vs stratified-prefix identities

Fixed six 7,13,25,33,55,71 on physical A0 only. Baseline is the sealed verified-EOS
`on` arm at Q64 (`Law(1,8)`, early=True, fused first verify, native Live executor).
Control on A0 must replay that sealed arm field-for-field: delivered IDs, physical
tape digest, hit, encoder digest, N, score, F and all route F/Q.

Treatment changes only the identity of the clean-prefix seats selected after the
unchanged `choose` dose. `prefix_coverage.spread_prefix` chooses equal-mass bin
endpoints from the same strict eligible pool returned by `_clean_prefix`
(initialized, non-dirty, before the first unresolved row; no prompt, unpaid row,
successor, debt or extra seat). Offer remains 64, quota remains 1/8 of remaining
seats, and the last eligible row remains the frontier. No T2T rule, RNG stream,
sampling-clock rule, admission threshold, verifier, EOS stop, encoder or scoring
change. Prefix rows remain subject to the existing native Live prefix semantics.

For each case run right-edge control first and require exact native replay;
only then run spread. Record the first trace fork with both selected prefix
identities and the strict spread pool. At that fork require same tick, Q64,
sparse route, sampled arity and prefix dose; the spread selected set must lie in
the logged eligible pool and preserve its frontier. Any replay/physical/fork
mismatch writes the partial raw case and stops. No blind retries. Raw case
files stream/copy to n164; A0 `/dev/shm` is removed only after n164 hashes match.
No expansion beyond the six and no default/profile recommendation follows.
