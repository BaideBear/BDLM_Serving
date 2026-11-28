# Q256 request-private continuous RNG on verified visibility

Fixed cohort, order 0/10/20, rest of original30, new60. A1 only, both arms
alternating within case, cut off at 1795843443. Previous visibility A1
frozen90 triplets are immutable references: same-stack native for comparison,
visibility-on as the replay control. Validate all 90 source row hashes before
loading the model. No new native run, other card, holder, switch or public API.

Hypothesis: after an actually verified and encoded block, the next random
initial canvas diverges because strict seeds a new generator. Continue the
one request-private generator across sequential block openings; do not borrow
global RNG. No draw, clock, acceptance, SC, write, projection, encoder, stop
or verify action is added or suppressed. Off/on both use verified visibility.
Do not generalize to interleaved or speculative blocks.

Abort on any mismatch of the *entire* off receipt to saved visibility-on
(including all output IDs, bills, scores, layer/tick hashes when present).
Require all first-block step ledgers, RNG states, canvases and delivered first
blocks to agree between arms, every unexposed request to agree throughout,
every continued stream to start at the previous generator end, one real draw
per opening, and one measured Q256 decoder projection per step across all 85
hooks. Anchors 0/10 must both change the second initial canvas from different
to identical to native at the same positions; 20 must have zero exposure.
Only then expand to 30/90 without quality-based stopping. No incomplete
cohort interpolation. Count actual exposure by blocks and requests.

Primary semantics endpoint: whole delivered IDs equal to native, with
exact first block preserved. Report paired transitions, total N, total F,
decoder/encoder bill, case-macro N/F, canonical sum(N)/sum(history decoder Q),
quality and paired uncertainty for original30/new60/frozen90. Prefer for
semantic alignment only if exact native matches improve without first-head
regression. No throughput-default endorsement on a directional point estimate.
