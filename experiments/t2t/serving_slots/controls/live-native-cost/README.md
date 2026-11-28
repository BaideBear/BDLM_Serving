# Live Gemma Q128: refresh and endpoint costs

**Decision: no new stop or fused Live comparator qualified.** Keep the existing
paid terminal EOS check. The attempted warm-refresh reuse rule found no eligible
adjacent calls on A800 physical 0 (A0). No treatment inference or 90-case
extension ran.

## Frozen90 cost

The 88/90, 13.047 macro TPF reference is the historical verified-EOS **on** arm,
not an IBS run or a new A0 result. `F` includes generation encoder forwards,
not prompt prefill. `Q` counts the physical decoder rows.

| Work item | Forward calls | Physical decoder rows |
|---|---:|---:|
| Sample plus any in-call refresh | 1,567 | 171,746 |
| Cold refresh only, during denoising | 164 | 20,992 |
| Warm refresh only, during denoising | 176 | 22,528 |
| Closing refresh only, before verification | 121 | 2,272 |
| Ordered terminal verification, including 43 already fused with refresh | 282 | 36,096 |
| Generation encoder commit | 74 | Not decoder Q |
| Total | 2,384 | 253,634 |

The largest separately identifiable refresh cost is repeated denoising refresh,
not a single endpoint call: 340 full-width drain calls and 43,520 decoder rows,
versus 121 closing drains and 2,272 decoder rows. Across pure drains **and**
sample calls, ordinary mandatory refresh accounts for 129,960 decoder rows
(51.24%). Another 487 dirty rows are already included in the 43 fused
verification calls, not counted twice. These are physical-work counts, not
measured latency or an estimate of how much can be skipped.

The audit joins 90 verified-EOS pairs, their fused-history references, and the
registered cold-pacing **control** receipts by frozen index. It reconciles
scores, IDs or ID digests, input hashes, physical bills and route counts before
using the control's phase split. The existing terminal EOS check removed
46 forwards / 5,888 decoder rows; in the remaining 44 requests the EOS was only
verified in the second chunk. All 90 still paid the entire EOS-containing chunk.

## Why further fusion did not qualify

- **No IBS prefix certificate.** Live Q128 uses a selected-row sampler with
  native entropy acceptance, edits, renoise and block RNG, not full-canvas
  `generate` parity. Its accepted prefix remains editable: the historical
  right-edge prefix seats recorded 24 edits and 41 revocations. Zero debt is
  not a freeze. The stop only becomes valid after `begin_close`, when sampling
  is off and the ordered terminal projection has emitted the EOS argmax.
- **Closing debt and commit.** The first verify chunk uses all 128 seats. Live
  fuses in the exact closing debt if that debt is wholly inside the first
  chunk; every remaining closing drain contains a row outside it, so the
  unchanged union exceeds Q128. Paying a later chunk first is not equivalent:
  same-block attention sees the resident KV written by earlier projections.
  Both verify chunks together are 256 rows. At a nonterminal commit, the
  encoder needs the complete verified IDs, and the next decoder needs the
  resulting new encoder cache. A stale-cache next call is not an exact fusion.
- **Denoising refresh.** The 164 cold drains already fill Q128; the next
  required cold rows are distinct. For warm drains the only rule probed was
  exact immediate reuse: an unchanged, full-width, ordered row set in the
  same encoder epoch, with identical selected IDs and self-conditioning and
  no intervening sampling. The control would verify bit-identical logits;
  a treatment would consume them through the unchanged sampler and verify the
  actual saved decoder call with the physical hooks. This rule was exposed on
  **0 of 9 warm drains across six A0 requests**. It did not execute a
  treatment, so neither equivalence nor a speedup was measured. The historical
  aggregate has 176 warm drains but lacks raw row sets to count this rule's
  exposure; this is not a claim of zero exposure on all 90.

## A0 receipt

The six fixed indices were `0,10,20,30,40,50`. Stock-only completed at 5/6,
with `N=1743`, `D=136`, `E=3`, `F=139`, and decoder `Q=14984`. There were 9
cold, 9 warm, and 6 closing drains. Input hashes match the archived cases,
but final ID digests differ in all six, with the first tape mismatch at tick
0 in each and one score flip (#20). This is an observed cross-run mismatch,
not evidence of a GPU-specific cause. Do not pool the pilot with the
historical 88/90 or use it to assert a treatment result.

Worker PID `789075`, launcher `789027`, GPU
`GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1`. Released on November 28, 2026 at
20:20:45 UTC+8 (`rc=0`, 0 MiB, no compute processes, locks closed).
A separate readback found all three locks free and no probe scratch directory.
Only source was staged on A800; the raw log is on n164:
`/data1/project_data/gx-erqw-20261123/results/gemma-live-native-cost-a0-20261128T1220Z/raw.log`.

[Compact receipt and raw hashes](receipt.json) | [Audit](audit.py) |
[A0 probe](probe.py) | [A0 launcher](launch.sh)

The probe used `/gs/runtime/verified-eos-1128/repo`, the byte-pinned historical
overlay based on `badf953ac0e51a0b3e3b1e5860041c6b5008ac54`, and the local
`diffusiongemma-cd40195` bf16 checkpoint with SDPA and `grouped_mm`. The launch
log records the exact source hashes. To reproduce the compact summary without
GPU inference, run `audit.py` on n164 with the following paths:

```sh
root=/data1/project_data/gx-erqw-20261123
python3 audit.py "$root/archive/a800-gs-20261128/snapshot/results" \
  --probe-log "$root/results/gemma-live-native-cost-a0-20261128T1220Z/raw.log"
```
