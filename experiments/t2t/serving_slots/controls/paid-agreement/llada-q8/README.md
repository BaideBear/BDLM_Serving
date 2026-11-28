# LLaDA Q8 paid agreement on A1

The three arms use fixed Q8 strict capped IBS, no optional prefix replay
or head/successor fusion, and the same certified EOS stop with real suffix
debt payments. Stock does not change native acceptance. Tracked observes
without applying. Apply uses the sealed Q16 rule: a selected, generated
MASK row must have the same nonspecial argmax at probabilities strictly
above 0.5 on two consecutive physical decoder ticks in the same block,
and must not already be natively accepted. The adapter changes only the
offer and the hard-limit assertion in the hashed Q16 hook.

The Q16 paired audit and independent canvas replay check actual positions,
all 20 layer projection widths, charged input IDs, ordinary and terminal
debt, stock/tracked identity, and the first apply fork. The scorer runs on
n164. The GPU stages use only physical A1 (UUID
`GPU-861c9b5d-1e73-2550-f832-b73988bc64ff`), no holder, and fresh receipts.

## Development gate

Indices 7, 13, 25, 33, 55 and 71 are the development slice and also the
original six of the Q16 study. All prompts and seeds matched the sealed
same-device reference.

| Arm | Correct | N | F | Q | Macro TPF | Pooled TE |
|---|---:|---:|---:|---:|---:|---:|
| Paid stock / observe | 6/6 | 1474 | 574 | 4259 | 2.6394 | 34.61% |
| Agreement | 6/6 | 1450 | 535 | 3885 | 2.7816 | 37.32% |

The six-case gate passed: no quality loss or new failure, higher macro TPF,
lower total F, non-decreasing pooled TE, and a clean audit. All six had a
measured first fork; total added admissions were 114. Case 7 used 27 more F.
The pre-existing strict non-paid endpoint is a separate comparison: same
output IDs and ordinary trace through the paid certificate on 6/6, at
1474 N / 593 F / 4412 Q. Do not use that endpoint in the paired denominator.

The expansion panel is exactly the other 84 indices (27 original and 57
added). The combined 90 includes the development six and is not an
independent confirmation set.

## Independent replication

Each cell reports paid stock (identical to observe) -> agreement.
N is delivered tokens, F is charged generation forwards, and Q is actual
projected generation rows; F and Q include terminal debt payments. TPF is
the mean per-request N/F; TE is pooled N/Q.

| Slice | Correct | N | F | Q | Macro TPF | Pooled TE |
|---|---:|---:|---:|---:|---:|---:|
| Non-development 84 | 64 -> 65 | 41096 -> 43159 | 18480 -> 19743 | 136167 -> 145039 | 2.4362 -> 2.5392 | 30.18% -> 29.76% |
| Non-development, original 27 | 19 -> 19 | 13604 -> 14556 | 6259 -> 6897 | 46080 -> 50464 | 2.3465 -> 2.4250 | 29.52% -> 28.84% |
| Non-development, added 57 | 45 -> 46 | 27492 -> 28603 | 12221 -> 12846 | 90087 -> 94575 | 2.4787 -> 2.5933 | 30.52% -> 30.24% |
| All 90 (includes development) | 70 -> 71 | 42570 -> 44609 | 19054 -> 20278 | 140426 -> 148924 | 2.4498 -> 2.5554 | 30.31% -> 29.95% |

The independent slice had 6 wrong-to-correct flips (3, 11, 17, 43, 60,
69) and 5 correct-to-wrong flips (22, 23, 28, 66, 84). Stock hit its
progress limit on case 11; agreement finished it correctly and introduced
no failures. Agreement made 4,442 extra admissions across 80 exposed
requests. F fell on 44, tied on 6, and rose on 34 requests.

**Decision: no-go for Q8 promotion.** The correctness audit and stock/observe
identity passed all 90, and accuracy and macro TPF improved on the independent
slice. But it used 1,263 more forwards (+6.83%) and pooled TE fell by 0.42
percentage points. Both non-development shards show the same F/TE regression.
The rise in TPF does not imply fewer forwards per request.

## Receipts

`n164:/data1/project_data/gx-erqw-20261123/results/llada-ibs-q8-paid-agreement-a1-v1/attempt-20261128T122257Z/`
contains the frozen code, per-stage stdout receipts, exit statuses, logs,
and score summaries. Both workers exited 0 (development PID 792069,
non-development PID 796687), the completion records contain 6 and 84
pairs respectively, and physical GPU 1 returned to 0 MiB / 0% with no
compute process. `all90-summary.json` combines the two separately sealed
summaries and records both raw receipt hashes; raw receipts remain on n164.
The earlier attempt `attempt-20261128T122154Z`
exited before loading the model due to an import-path conflict; its output
was not used. The second attempt pins the frozen import path.
