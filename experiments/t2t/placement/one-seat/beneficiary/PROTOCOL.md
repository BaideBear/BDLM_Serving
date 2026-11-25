# Does an exchanged seat benefit the other MASKs?

Freeze before reading the new runs. Replay the previously registered first single-seat
intervention on all identical 30 requests per model. LLaDA 2.1 Q4 and DiffusionGemma
Q32: zero-reserve base, targeted exchange, random exchange. Keep original selection,
sampling, seeds, physical Q accounting and all no-opportunity requests. Continue with
zero reserve after the intervention. Verify outputs, forwards, Q and historical state
on each arm before interpreting the new observer.

Record only after each denoise forward, including verification, from the anchor
through the end of its block. Follow previously unresolved positions in the base
selection, separating masks common to all three anchor selections from the displaced
MASK (if any). Store accepted IDs only, never draft IDs. The reference is the
base arm's final delivered token at the same generated-output offset, not ground
truth. Report out-of-delivery offsets separately; no imputation of reference IDs.

Primary horizon H=4 denoise events (anchor=0). After block exit carry forward
its last accepted state for the remaining horizon, without adding actual forwards.
First match and first sustained match through the end of the four-step window:
0,1,2,3; code unobserved match as 4. This is not lifelong token stability.
Compare paired branches at common rows and the displaced MASK, report endpoints,
coverage and denominators, stratified by model and pre-intervention victim type
(old refill vs MASK). Strata are observational, not randomized. Keep request-level
results to avoid overweighting requests with many masks. No hyperparameter changes,
sample replacements or arm exclusions based on quality. Single-intervention effects
are descriptive; report the real 30-request quality, TPF and physical token efficiency.
Publish only aggregates and token-free request summaries, never raw IDs.
