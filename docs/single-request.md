# Single-request execution at a given Q budget

The kernel takes one request and a supplied budget. At each fixed budget the two objectives are frozen-task accuracy and tokens per forward (TPF). Do not trade away one in a hidden scalar score: report the observed non-dominated methods when the two objectives disagree. Only compare strict-budget candidates after checking every generation decoder forward, including initialization, verification and commit. Gemma's post-commit encoder is a separate, explicitly billed cost; a decoder cap is not an end-to-end per-forward cap on that encoder. There is no request ordering, admission or load controller in this task.

The handle below is the existing serial reference. Its nominal offers are **not** a strict cap. Keep the same handle if the supplied budget changes; replaying a predetermined sequence tests cache continuity, not how to choose that sequence.

```python
from bdlm_serving.methods.t2t import serving_profiles as profiles

def drive(handle, q):
    # Run one request at the given fixed offer.
    # Keep prompt-only work separate from generation F and Q.
    prefill_q, paid_q, paid_f = handle.opened_prefill_q, 0, 0
    try:
        while handle.status == "ready":
            step = handle.advance(handle.menu()[q])
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

**Measured cost.** `price.main_q` is the projected width of the next decoder; `mandatory_q` is included. Mandatory cache writeback, warm-up, verification or commit can exceed a small nominal offer on this reference path. One such forward disqualifies a run from the strict tier even if mean Q is below the cap. Use `step.paid_q` / `step.paid_f` for actual generation costs, including conditional encoder work. `result()` returns token IDs; the frozen scorer counts non-special output tokens for N. TPF is `mean(N/F)` and efficiency is `sum(N)/sum(paid_Q)` across requests.

**Strict Gemma decoder (experimental).** To actually limit *every decoder projection* to Q128, including the first visit and the final argmax sweep, use the single-canvas request below. One call performs one measured decoder forward. For the frozen suite, charge post-commit encoder calls as shown; only pure prompt prefill is excluded.

```python
from experiments.t2t.serving_slots.gemma_strict_request import StrictRequest

request = StrictRequest(model, data, seed=seed, max_new_tokens=2048, prefix_quota=0)
f = physical_q = 0
while request.status == "ready":
    step = request.step(128)
    f += 1 + int(step["encoded"])
    physical_q += step["q"] + 256 * int(step["encoded"])
token_ids = request.tokens
```

Valid single-canvas caps are 32, 64, 128 and 256; do not use this class to represent Q512. An externally changed cap can be supplied to the same object. Token changes (including re-noising) and changes in acceptance require a refresh; a change in self-conditioning alone does not. Unselected rows keep their cached state. Right-prefix sampling can revise a token; this differs from the LLaDA IBS control which only refreshes KV in spare seats. This experimental local sampler, sparse initialization and ordered terminal sweep change the trajectory relative to native Gemma. The first real-weight paired case is a functional test, not a frozen90 quality result. [Physical projection tests](../tests/test_gemma_strict_request.py).

**Reproduce and inspect.** [Frozen data and environment](reproduce.md), [slot results, baselines, and limits](../experiments/t2t/serving_slots/EXPANDED.md), [directional mask and paired results](../experiments/t2t/serving_slots/mask-evidence.svg), [tests of the public entry points](../tests/test_request_profiles.py). The single-request wrappers expect this checkout, including `experiments/t2t/serving_slots`, on the Python path.

Two experimental actions implement the same `step(q)` API. Choose one intervention at a time:

* [PrefixActionRequest](../experiments/t2t/serving_slots/gemma_prefix_action.py) takes `prefix_quota=32, prefix_mode="right", prefix_action="kv_only"` or `"editable"`. In the former, the extra prefix rows are physically projected but excluded from the sampler. Their old token, acceptance and self-conditioning stay fixed; the ordinary sampled rows can still change. `step["prefix"]` is a subset of `step["rows"]` in both modes, not necessarily of `step["sampled"]`.
* [ClosingRequest](../experiments/t2t/serving_slots/gemma_strict_close.py) can instead project pending terminal writes as part of the argmax sweep. With `prefix_quota=0` at Q128 it reproduced three complete CPU outputs using 5 fewer decoder calls, not a general equivalence or latency guarantee. Both implementations retain the post-block encoder charge.
