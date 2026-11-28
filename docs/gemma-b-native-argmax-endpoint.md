# Gemma B native-argmax endpoint protocol

Scope: fixed Q256/B, one sampled head, no parked tail, no rebuild, and no
Q32-IBS dependency.  This is not a serving default or a sparse-budget rule.

Read order: `native-visibility-a1` establishes the representation boundary,
`continuous-rng-a1` establishes next-block RNG continuity, and
`predicate-only-a1` isolates the real native stop.  The offline audit below is
sufficient for the endpoint decision, so no GPU acceptance runner was started.
It is **not** proof that switching nonterminal context to native argmax is a
safe deployment path; that would require a separately coordinated runner.

## Decision from archived receipts

Existing evidence is sufficient for the narrow endpoint question.  The
predicate-only A1 run has 90 complete requests.  Its on arm made 1,260
measured full-head denoising forwards, closed all 168 terminal blocks with an
actual native stop, and had zero step-limit-only closes.  Native generate's
output boundary is the last same-turn argmax after a hit; it performs no extra
forward over the updated canvas.  Therefore that same-turn native argmax is a
valid model-defined output for this qualified endpoint.

This is **not** whole-request native parity and not permission to use sparse or
cached logits.  The same receipts show ordered strict verification changed the
native-stopped draft in 20/168 blocks.  In 19/90 requests, the first mismatch
from same-stack native occurs exactly at verify.  Five of the 78 nonterminal
blocks sent to an encoder would therefore have a different strict context.

## State and boundary contract

- **Terminal block:** the denoising forward that produced the hit and the
  same-turn argmax is real and already paid.  Under the native endpoint, that
  argmax is the block output and native performs no further forward over the
  updated canvas.  The archived strict arm nevertheless paid ordered
  verification after the hit so that the draft/verified difference could be
  measured.  Under the old strict contract, that verification argmax is the
  output; a draft EOS is not sufficient.  Never use sparse or cached logits as
  the same-turn observation.
- **Nonterminal encoder input:** the current sealed candidate encodes the
  verified block; all 78 encoder calls are real.  A native-argmax endpoint must
  instead encode the chosen native argmax block.  It may differ on five blocks;
  it still costs an encoder call and 256 Q, so no virtual F saving may be
  claimed.
- **Next block context:** continuity is defined by the chosen endpoint.  Strict
  continues from verified delivery; native-argmax continues from native argmax
  delivery.  Do not mix one as output and the other as next context.  The
  continuous-RNG experiment already establishes that the next canvas must
  continue the request RNG, not reseed by ordinal.
- **Previous-step debt:** unchanged.  Dirty writes, acceptance edits, rebuilds,
  and required encoder work are paid before any endpoint choice or savings
  claim.  Ordered verification remains mandatory while executing the archived
  strict arm; omitting it is part of the proposed native endpoint, not an
  established efficiency result.  A native stop only ends further denoising of
  that head.

## Difference from old strict contract

Old `dense_stop_joint_verified_eos` treats the ordered paid verification argmax
as delivered state and stops the sparse terminal sweep only at a measured EOS
in that verified prefix.  The proposed endpoint treats the native-stopped
same-turn argmax as delivery and, if nonterminal, as encoder/next context.
They are different model-defined endpoints, not equivalent implementations.

## Limits

The archived on arm paid strict verification and encoded the verified block.
Thus the receipts can compare the native-stopped draft against what strict
actually delivered/encoded, but cannot prove the downstream behavior of a new
serving path that also changes every nonterminal context to native argmax.
Report no virtual F saving: the observed candidate paid 78 real encoder calls;
same-stack native paid 81.  A missing encoder call is not inferred from a draft
that would have differed.

If such a path is later needed, first coordinate a free A800 card (not H0/H4,
A2/A3, or another owner's card), then run same-card native versus instrumented
native with observation-only hooks.  The hooks must record real criterion and
projection events without changing logits, RNG, row order, debt, or stop.

## Offline audit

```sh
python3 -m tools.audit_gemma_native_argmax_endpoint
```

The tool reads the sealed compact native-visibility, continuous-RNG, and
predicate-only CSVs, checks their 90-row provenance links, and reports first
block alignment, later request-RNG carry, terminal blocks, native stops,
actual encoder calls, and the five strict/native context divergences.  No
model or device is loaded.
