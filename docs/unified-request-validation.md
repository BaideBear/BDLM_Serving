# Unified request validation — preregistration and acceptance checklist

Scope: one shared planner for LLaDA and Gemma, with scalar hyperparameters
only. The two checkpoints can have independently tuned parameter values;
equal decisions on synthetic identical inputs are only an adapter unit test.
Adapters own model-specific execution; the planner sees a normalized
`Block` snapshot and returns a `RowDecision`. This aligns with
`docs/shared-request-planner.md` and `bdlm_serving/methods/t2t/unified_rows.py`.
It does not change shared code.

## 1. Actual planner contract

The only ordinary-selection entry point is
`unified_rows.plan_rows(q, b, head, next_block=None, *, opened=False,
params=Params())`.

- `Block(start, size, masks, dirty, resident)`:
  - `masks`: unresolved positions. LLaDA = actual MASK; Gemma = positions not
    accepted by the sampler, not a literal token-ID comparison.
  - `dirty`: published input changes not yet consumed by a decoder call.
  - `resident`: KV-resident positions.
- `Params(prefix_share=1/8, open_at=1/4)`; these are the only scalar
  hyperparameters.
- `RowDecision(rows, dirty, mask, prefix, cold, sampled, open_next)`.
- Planner inputs are only actual `Block`/`q`/`b`/`opened`/`Params`. There is no
  model name, dataset, logits, future budget, adapter-ranked candidate list,
  `max_steps`, confidence/edit threshold, coverage flag, raw token ID, or
  legacy `warm_head_sample` input. `warm_head_sample` is not active.

## 2. State audit: what is and is not equivalent

### 2.1 Mutable resolved state

- LLaDA: resolved means generated `x != mask_id`. There is no
  `rows.accepted` field. `x` is mutable through edits/revokes/renoise.
- Gemma: resolved means `StrictRows.accepted` (sampler-accepted positions).
  `finish` recomputes it from `accepted_after`, `input_changed` and `flips`;
  it is mutable, not an immutable history.
- The planner does not read `resolved` directly. Adapters map their native
  resolved state to `masks`; do not claim native equivalence.

### 2.2 Sample clock is not a planner input

LLaDA `sample_clock` is per request; Gemma `SelectedRowSampler.age` is per
canvas/head. They are not equivalent, and neither is a planner input. Native
sampling may use them, but the shared plan must not depend on them.

### 2.3 Dirty debt scope

Gemma `StrictRows.finish`:

```text
after = {r for r, ok in zip(plan.sampled, accepted_after) if ok}
flips = (accepted ∩ sampled) ^ after
dirty = (dirty - plan.rows) | flips | {r for r, new in zip(sampled, changed) if new}
accepted = (accepted - sampled) | after
```

In `gemma_fixed_dual._apply`, `input_changed` compares the previous and new
token IDs. An acceptance flip also incurs debt. A change to self-conditioning
alone, with unchanged ID and acceptance, does **not** incur debt. LLaDA `dirty` is
token-write based (`admitted`/`edited`/`revoked`) and has no `accepted` field.
The planner sees only the published `dirty`; adapters must document their
mapping and scope. Do not silently assume native equivalence.

Dirty is global across switch/promotion. A switch, tail park, head promotion
or encoder-epoch change may change `resident` but must not clear `dirty`; the
next plan pays `sorted(dirty)[:q]` first.

### 2.4 Unresident clean

`clean_unresident = resolved - dirty - resident` exists. It must not be used
as prefix/conditioning without a rebuild. The planner's prefix selection uses
`resident - dirty` before the first mask, so unresident clean is excluded;
adapters must not publish it as `resident`.

### 2.5 Bank/residency checks

LLaDA `_Bank.begin` checks that new rows extend a contiguous high-water from
`last_end` to `end` and that `end <= capacity`. `SelectedResidentCache.configure`
separately checks that every position below `end` is initialized. `_Bank.update`
uses absolute `index_copy_` and does not itself prove all prior positions are
initialized; do not claim "no holes" universally. The adapter must enforce
residency. Gemma `WaveKV.live` is the resident set; `drop_before`/rebuild may
change it, but `dirty` must survive.

## 3. Preregistration checklist

- [ ] Publish the normalized `Block` schema and adapter contract; no raw object
  access from the planner.
- [ ] Only scalar hyperparameters `prefix_share` and `open_at`; no
  model-specific branches in the planner.
- [ ] Both adapter owners sign off on the mapping: LLaDA `x != mask` → masks;
  Gemma `accepted` → masks; dirty scope documented.
- [ ] Plan invariants: `0 < width <= q`; dirty rows first; sampled ⊆ projected;
  no stale logits; `sampled = mask + prefix`; no extra Q; EOS only from native
  verification; clean unresident not used as prefix; switch/promotion preserves
  `dirty`.
- [ ] Fix cases before scores: 0/10/20 pilot, original30, new60; same inputs and
  seeds; alternate on/off; control must reproduce the sealed reference.
- [ ] Metrics: plan equality on normalized states, first-difference ledger,
  width/debt checks, N/F/Q, pooled TE, macro TPF, correct.
- [ ] Any invariant failure invalidates the pair; no post-hoc replacement.
- [ ] Archive compact: summary, `pairs.csv` with hashes, manifest; raw on
  cluster/n164; no local traces.

## 4. Evaluation acceptance checklist

- [ ] Both adapters emit identical normalized `Block` snapshots for the same
  logical state.
- [ ] The planner emits the same `RowDecision` for both adapters on the fixed
  state set.
- [ ] Width/debt invariants hold on every tick.
- [ ] No stale logits: sampled rows are a subset of the current projected rows;
  logits come from the current forward.
- [ ] EOS: only a measured verified EOS may terminate; no unverified shortcut.
- [ ] Dirty ledger: `dirty_after == (dirty_before - rows) | writes`; no debt lost
  across switch/promotion.
- [ ] `clean_unresident` is counted separately and never used as prefix without
  a rebuild.
- [ ] LLaDA `x != mask` and Gemma `accepted` mappings are documented; no
  `rows.accepted` assumption.
- [ ] `sample_clock`/`age` are not planner inputs; native sampling may use them.
- [ ] Same-device control reproduction: IDs, bills and seeds match the sealed
  reference before any treatment is read.
- [ ] First-difference ledger: first intervention matches pre-state, rows, q,
  dirty, prefix, mask, cold, resident, phase and native physical Q.
- [ ] Compact archive complete; raw retained on cluster/n164; no local traces.
- [ ] No shared planner code edits; adapters own execution.

## 5. Warm-coverage evidence links

Corrected historical run: n164 archive `snapshot/results/gemma-warm-coverage-v2-1128`
under the root in [cluster storage](cluster-storage.md). In 27 screened pairs
the corrected policy did not change the physical rows; these are not results
of the shared planner. The earlier v1 absolute/local coordinate mix-up
invalidates its contrasts.
