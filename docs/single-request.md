# Single-request execution at a given Q budget

[How to spend a small, full-block or two-block offer](request-budget-decisions.md):
mandatory writes, optional prefix rows, when to open the successor, and
which proposed savings changed outputs in the fixed-budget experiments.

`open_strict_llada(..., profile="quarter_edit_warm")` uses the same live KV
bank and absolute dirty ledger at all five offers. Relative to `quarter_fullwrite`
only two conditions change. At B, a fully resident, mask-free head with all
outstanding writes in that head can sample its already-paid logits once before
retirement. Any resulting edit or revocation becomes debt. At 2B, when the
only cold positions are in the successor and all mandatory rows fit, sample
the already-projected warm head MASKs in the same call. The thresholds and the
quarter-MASK opening gate do not change. See the [switching evidence](../experiments/t2t/serving_slots/controls/strict-handle/llada-quarter-edit-warm.json)
and the [measured fixed-budget comparisons](request-budget-decisions.md).
This is an opt-in implementation, not a claim of dominance at low offers.

`open_strict_llada(..., profile="quarter_fullwrite")` accepts every registered
B32 offer on one persistent request. Q4/Q8 keep the head-only planner; Q16
may lend real slack to contiguous successor rows without sampling them; Q32
can spend a mask-free write on every valid head row, once, with **no** new
sampling. It does so only if the head is already initialized and all
outstanding writes are in this head. Q64 projects the full current head and
opens a successor once at most a quarter of valid head positions remain
masked. This profile does not enable the separate Q32 edit or Q64 warm-head
sampling factors. The [functional GPU qualification](../experiments/t2t/serving_slots/controls/strict-handle/llada-quarter-fullwrite.json)
includes 15 fixed-arm exact replays (all positions, IDs, F and Q) and all
25 directed offer transitions. The latter are state-machine tests; they do
not score a mixed policy. Omit the profile to retain legacy behavior.

`open_strict_gemma(..., profile="dense_stop_joint_single512")` opts into
the joint-verify profile below plus single-head stopping at Q512. A pair
requires consecutive `dense_single` samples of all 256 positions, exactly
one resident block, the same head and encoder epoch, **and the same offer**
(256 or 512). A parked successor, sampled rebuild, partial sample or changed
offer breaks the window. Each decision uses fresh processed logits. A hit
stops further sampling of that head, not writeback or final verification.
For example, `step(32)` after a Q512 latch pays the old writes and the normal
ordered terminal sweep in cap-sized calls; only fresh terminal logits supply
the output. The default profile and `dense_stop_joint_verify` are unchanged.
See the [GPU replay and switch qualification](../experiments/t2t/serving_slots/controls/strict-handle/gemma-single512-stop.json)
and the [current fixed five-tier chart](../experiments/t2t/serving_slots/controls/gemma-factors/fixed-profile/request-frozen90.png).

`open_strict_gemma(..., profile="dense_stop_joint_verify")` keeps the same
five offers and the previous guarded stop/promotion rules. Only an offer of
512 with a fully initialized terminal head, resident successor and outstanding
writes can combine the full head verification and the successor's required KV
writes. No tail sampling or encoder work is skipped. A smaller offer drains the
writes within its own physical limit, then verifies in chunks. See the
[fixed frozen90 comparison](../experiments/t2t/serving_slots/controls/gemma-factors/joint-verify/README.md),
[previous five-tier chart](../experiments/t2t/serving_slots/controls/gemma-factors/fixed-profile/previous-joint-frozen90.png)
and [public API GPU qualification](../experiments/t2t/serving_slots/controls/strict-handle/gemma-joint-verify.json).

`profile="dense_stop_promotion_fused"` additionally lets a fixed Q512 offer
consume the first ordinary sample on a promoted head during its complete
new-epoch rebuild. This requires B genuinely missing rows, no dirty rows,
one sampleable head, and no successor that should open this tick. A partial
rebuild, including after a downshift, never takes the shortcut. Q256 retains
the guarded dense stop; the three sparse tiers keep their own row choices.
The [same-device fixed Q512 comparison](../experiments/t2t/serving_slots/controls/gemma-factors/promotion/paired.csv)
preserved all final IDs and removed exactly 58 decoder forwards on frozen90.

For a switchable Gemma profile with the measured Q256 full-head stop, pass
`profile="dense_b_stop_fused"` to `open_strict_gemma`. The other fixed offers
keep their `fixed90_fused` paths. Only two consecutive real, complete,
single-canvas Q256 samples of the same head/encoder epoch can trigger the
native criterion. Any intervening ineligible tick resets the observation.
A completed decision survives a subsequent downshift; required writes,
ordered verification and encoder commits still run and are billed. This is
an opt-in fixed-budget candidate, not a policy for choosing offers.

The experimental Gemma profile `phase_high_fused` uses the same five external
offers and the same guarded fuse as `fixed90_fused`. Only at Q256 and with
at least 128 still-unaccepted head positions does it request the full head.
The executor still declines that path if it would displace outstanding writes
to a parked successor. No state is reset at this density threshold; it can
be crossed in either direction after token revisions. This is not a change
in the offered budget. See the [fixed-budget study](../experiments/t2t/serving_slots/controls/gemma-factors/README.md)
for the evidence; this experimental profile does not change the default.

Start with the [current fixed-budget evidence](../experiments/t2t/serving_slots/design_race/README.md). To drive the current request on **one serial model lane**, use the strict entry points below. They do not run a scheduler. The nominal-budget interfaces further down remain for historical reproduction.

```python
from bdlm_serving.methods.t2t import serving_profiles as profiles

# Call one constructor, with GPU-resident, registered model and inputs.
h = profiles.open_strict_gemma(model, data, seed=seed, initial_offer=32)
# Or, after installing the rectangular LLaDA mask on the loaded model:
# h = profiles.open_strict_llada(model, input_ids, seed=seed, mask_id=mask_id, eos_ids=eos_ids)
generation_f = history_q = physical_q = 0
try:
    while h.status == "ready":
        e = h.step(q)  # fixed q for evaluation; an external caller may change it every tick
        generation_f += e["generation_F"]
        history_q += e["history_Q"]
        physical_q += e["physical_generation_Q"]
    output_ids = h.result()
finally:
    h.close()
```

`initial_offer` on Gemma selects the initial setup, never an obligatory first
large decoder forward. Opening performs prompt-only prefill. The first and every
subsequent generation projection use the actual offer passed to `step`.
No clean-ups, warm-ups or second decoder forwards are hidden from the offer.
A Gemma encoder commit can add one F and 256 physical Q but does not add to
the selected decoder rows reading prefix KV. Use the count `N` of final
non-special IDs, excluding repeated edits: TPF is mean per-request `N/F`;
token efficiency is `sum(N)/sum(history_q)` across the registered cohort.
The profile does not manufacture a forecast for a different offer.

Gemma `fixed90_candidate` uses the right-edge rank and clean-prefix share
1/4 at Q128, reference 1/8 at the other sparse tiers. `reference` keeps 1/8
throughout. Both use the half gate at Q512, do not borrow at lower tiers, and
do not enable the experimental stop or terminal fusion. An offer with no
eligible prefix or a dense two-block operation need not use any quota. LLaDA
uses the same 1/8 law at all tiers pending its paired dose results; select
`coverage="sparse"` (default) or experimentally `"head"` at Q64, and specify
`top_gate` explicitly for a registered alternative. These are fixed-tier
candidates, not an assertion that either profile dominates IBS or FlowBlock.

`fixed90_fused` retains the candidate's row plan and combines the last writeback
with the first verification slice only when the current head alone is resident
and that slice contains all outstanding writes. Both Q128 and Q512 have now
shown exact final-ID parity on all 90 at their actual candidate configurations.
`fixed90_dense_fused` additionally spends a true Q256 on the whole head when
it fits together with outstanding writes. It cannot displace dirty successor
positions to do so. This last change can alter output IDs and is experimental;
the [fixed same-budget controls](../experiments/t2t/serving_slots/controls/gemma-factors/README.md)
keep it separate from terminal fusion.

The kernel takes one request and a supplied budget. At each fixed budget the two objectives are frozen-task accuracy and tokens per forward (TPF). Do not trade away one in a hidden scalar score: report the observed non-dominated methods when the two objectives disagree. Only compare strict-budget candidates after checking every generation decoder forward, including initialization, verification and commit. Gemma's post-commit encoder is a separate, explicitly billed cost; a decoder cap is not an end-to-end per-forward cap on that encoder. There is no request ordering, admission or load controller in this task.

The handle below is the historical serial reference. Its nominal offers are **not** a strict cap. Keep the same handle if the supplied budget changes; replaying a predetermined sequence tests cache continuity, not how to choose that sequence.

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

**Measured cost.** `price.main_q` is the projected width of the next decoder; `mandatory_q` is included. Mandatory cache writeback, warm-up, verification or commit can exceed a small nominal offer on this reference path. One such forward disqualifies a run from the strict tier even if mean Q is below the cap. Use `step.paid_q` / `step.paid_f` for actual generation costs, including conditional encoder work. `result()` returns token IDs; the frozen scorer counts non-special output tokens for N. TPF is `mean(N/F)`. Historical `sum(N)/sum(paid_Q)` is a physical-work ratio, **not** token efficiency when `paid_Q` includes encoder work. Publish token efficiency only with a verified prefix-KV row denominator.

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
