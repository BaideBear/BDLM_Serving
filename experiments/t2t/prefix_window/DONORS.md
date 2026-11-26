# Which existing query should give up its seat?

Registered before online donor-arm scores. Gemma B256, frozen90, original seeds, native scoring and full-width qualification. Q128/K32 random is the primary cell; Q32/K8 right is a secondary contrast. These cells were chosen using prior results on these 90 tasks: this is an exploratory mechanism test, not independent validation.

At each selected step, build the existing zero-add plan. Let P be accepted generation rows in the current physical block before the first unaccepted position, minus mandatory fresh rows. B denotes selected ordinary accepted backfill rows, outside P. M denotes selected unaccepted rows other than the protected first one. Strict T2T is not a selection feature. Unselected KV stays visible. No extra forwards, window changes or hard cap on mandatory work.

| Arm | Count on that state | Donor from the zero plan |
|---|---|---|
| Zero (archived, same nominal Q) | 0 | None |
| Backfill-only | min(K, \|P\|, \|B\|) | Rightmost B; preserve every selected MASK |
| Paired-backfill | min(K, \|P\|, \|B\|, \|M\|) | Rightmost B |
| Paired-MASK | Same paired count | Rightmost M |

The two paired arms use exactly the same recipient ordering and physical width on a common state. If there is no donor of either kind, both do nothing. Neither consumes slack: this isolates changing the composition at a fixed forward width. The existing slack-only arm is a separate cost reference. Across diverged trajectories do not assume equal total Q, number of opportunities, draft, KV or token identities.

**Primary comparison:** paired-backfill vs paired-MASK at Q128; report original30, added60 and all90 paired quality, rescues/losses, macro TPF, physical Q and token efficiency. If protecting MASK does not improve correctness or if TPF goes down, do not describe it as a joint gain. Compare backfill-only to zero for a deployable alternative not gated by the presence of a second MASK. Report Q32 identically, without pooling budgets. Check donor class, recipients, zero and actual widths on all selected steps; compare pre-canvas and pre-draft digests up to the first intervention. Afterward compare whole-policy outcomes, not individual-row causal effects. No 90-task selection/filtering or adjustment to the independent decode seed.

On the archived zero trajectories the paired intervention could have exchanged 5,766 rows over 462 selected forwards in 86/90 requests at Q128; Q32: 413 rows, 98 forwards, 35/90 requests. These counts measure opportunity, not quality. Over the same trajectories backfill-only could use 9,389 and 598 rows respectively. Each new online arm uses all 90 cases regardless of opportunity.

Run-list generator: `python -m experiments.t2t.prefix_window.grid gemma 128 donor` (or `32`). The registered runner separately charges initialization, verification and re-encoding.
