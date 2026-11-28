# A3 Q128 first-payment recheck, frozen six

Six fixed indices `7,13,25,33,55,71`. Use three arms in alternating order:
original repeat, measurement-only repeat, and repeat plus correction.
All retain the registered capped IBS planner, repeat admissions, seed,
and paid EOS. The reference is n164 `results/gemma-payment-observer-a3-v1`.
The repeat must match every available registered apply field; the observer
must reproduce both the full repeat trajectory and the archived payment tape.

Only a repeat-extra position on its first real dirty-ID payment can change.
Use the current raw FP32 softmax, a different non-special top-1, and strict
`p > .9`. Publish after the unchanged same-step repeat sampler and noise
draw. Mark the replacement dirty with its position still resolved; it
cannot be frozen or encoded until a later real projection reads the new ID.
Replacements are not eligible for another correction. A terminal suffix
projection with a new replacement must be followed by one true suffix-only
repayment within Q128 (no new sampling). A broken boundary or missing
payment fails the run rather than disabling a qualifying edit.

Record stage, first difference, same-step decisions and RNG, physical input
IDs for both payments, frontier, full Q bill, complete delivered IDs, and
encoder inputs. Count every paid forward and row. No answer-based rule and
no extension to 90. This is opt-in experimental T2T, not native Gemma IBS.
