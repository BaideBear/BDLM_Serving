# Single-request serving interface

The outer scheduler owns the offer **every decoder step**. It does not have to follow a recommended cycle. Keep the same handle across offers, including transitions between one and two physical blocks. The handle is a serial batch-one reference implementation, not a new continuous-batching server. Use one owning thread and no overlapping forwards on a shared model.

```python
from bdlm_serving.methods.t2t import serving_profiles as profiles

def drive(handle, choose):
    # choose(menu) is supplied by the caller's load policy; it returns an offer.
    # Keep prompt-only work separate from generation F and Q.
    prefill_q, paid_q, paid_f = handle.opened_prefill_q, 0, 0
    try:
        while handle.status == "ready":
            menu = handle.menu()     # does not run a decoder forward
            offer = choose(menu)    # e.g. lambda menu: 64 for a fixed-slot control
            step = handle.advance(menu[offer])
            paid_q += step.paid_q
            paid_f += step.paid_f
            prefill_q += step.prefill_q
        return handle.result(), prefill_q, paid_q, paid_f
    finally:
        handle.close()
```

**LLaDA2.1, B32.** Load the pinned BF16 model and tokenizer, call `install_llada_rectangular_mask(model, allow_selected_cache=True)` from `bdlm_serving.runtime.serial` once at setup, and pass chat-templated input IDs of shape `(1, prompt_len)` on the model device. The frozen protocol uses:

```python
config = dict(confidence_threshold=.5, editing_threshold=.5, max_post_steps=16)
handle = profiles.open_llada(model, input_ids, config, gen_length=2048, seed=seed,
    mask_id=tokenizer.convert_tokens_to_ids("<|mask|>"), eos_ids=[tokenizer.eos_token_id],
    profile="progress80_candidate")
```

Offers: **4, 8, 16, 32, 64**. Both profiles keep cross-block causal visibility even after a downshift. `economy` opens the next block eagerly on an eligible Q64 offer. `progress80_candidate` can defer that opening until the last observed head step removed at least 80% of its then-remaining MASKs; it requires no extra forward or logits from unselected positions. Both use zero extra optional prefix rows. The gate is a candidate, not a GPU90 quality guarantee. An offer does not promise that the full nominal Q will be used.

**DiffusionGemma, B256.** Load the pinned text model and its processor; `data` is the native processor's single text-only chat input with all prompt rows valid, on the model device. Initialize a handle with:

```python
handle = profiles.open_gemma(model, data, dict(block_size=256, cap=2048),
    seed=seed, profile="economy")
```

Offers: **32, 64, 128, 256, 512**. `economy` retains causal visibility, admits a second canvas when the head progress threshold is met, and inserts no optional prefix. `q128_prefix_candidate` fills genuine spare Q128 positions with up to 32 random positions from the current block's contiguous decoded prefix. Use it as a fixed-slot comparison, not as the mixed-path default: [paired transfer results](../experiments/t2t/serving_slots/controls/gemma-unified-profile-transfer.csv).

**What a quote buys.** `price.main_q` is the exact projected width of the next decoder; `mandatory_q` is already included in that width. `price.post_q_max` bounds conditional paid encoder work. Mandatory cache writeback can exceed a small nominal offer; warm-up, verification, and commits can also have different widths from the offer. If a current hard limit cannot accommodate any quote, the outer scheduler can defer this request. Do not silently clip rows or count nominal Q as actual usage. Use `step.paid_q` / `step.paid_f` for the actual bill, not the quote maximum. `result()` returns token IDs; the frozen scorer counts non-special output tokens for N. TPF is `mean(N/F)` and efficiency is `sum(N)/sum(paid_Q)` across requests.

**Reproduce and inspect.** [Frozen data and environment](reproduce.md), [slot results, baselines, and limits](../experiments/t2t/serving_slots/EXPANDED.md), [directional mask and paired results](../experiments/t2t/serving_slots/mask-evidence.svg), [tests of the public entry points](../tests/test_request_profiles.py). The single-request wrappers expect this checkout, including `experiments/t2t/serving_slots`, on the Python path.
