# Gemma hardcap IBS: frozen-prefix EOS

Q32, B256, single physical block, batch-one ordinary Optimus IBS, generation
cap 2048. Pinned DiffusionGemma bf16 / SDPA / grouped_mm on A800 physical 0
only. Same prompt, seed, model, unmasker, decoder and GPU in all four arms.
Qualification: frozen90 indices 0, 10, 20, 30, 40, 50 in that order. Only once
all six pass, continue the identical four-arm test on the remaining 84 in index
order. Alternate forward and reverse arm order. No replacement questions.

- **stock**: unmodified IBS, observation only.
- **control**: same EOS predicates at the same post-update boundary; continue
  stock to natural termination to measure the counterfactual bill. The control
  records both first certificates. A same-stop baseline is the control at the
  same boundary (and with the same paid forward for the paid rule), not an
  alternate row allocator.
- **strict**: after a real decoder invocation and upstream `update_from_mask`,
  deliver iff the first EOS is covered by the current head's frozen prefix,
  that head is not yet retired, and the entire baseline dirty array is empty.
- **paid**: apply the same clean rule, or at the first qualifying frozen EOS
  with nonempty dirty, require all global dirty to be within this physical
  block and strictly after EOS, and fit in this invocation's fixed offer.
  Execute exactly one real suffix-only decoder forward on precisely that debt,
  no new sampling, no renoise, no encoder. Check physical writes, cleared debt
  and unchanged delivery. Do not inspect future work or suppress a zero-gain
  payment. Charge this decoder forward and every projected row to this arm.

Gate: stock/control match for the entire request; treatments match stock
through each trigger (inclusive), or for their entire request if unexposed.
Compare real decoder positions/inputs, all sparse sampler argmax proposals
and admissions, RNG, frontier, complete delivered IDs, encoder inputs,
score, reconciled debt and projector widths for every Q/K/V module. Require
no pending writes at delivery. D/E are decoder/continuation-encoder forward
counts; F=D+E, Q=actual decoder history-read rows; encoder rows reported
separately. Prompt prefill is observed, excluded from F/Q. No simulated
forward is included, no latency or native T2T claim, and no improvement
attributed to row selection. Strict and paid are separate stopping factors
relative to stock; reuse the same stopping rules for any subsequent baseline.
