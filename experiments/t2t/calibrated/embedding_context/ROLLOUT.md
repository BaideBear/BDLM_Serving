# Bounded LLaDA 2.1 real sparse rollout: frozen before launch

Use the original frozen30 cases, seeds, scorer, native B32 checkpoint and
batch-one online backend. Qualify full-width 30/30 before scoring any sparse
arms. No partial runs or per-request exclusions. Compare `hard_context_fixed`
against `hard_hazard_fixed` at Q4/1, Q8/1, Q16/2. Run the Q4 and Q8 positional
controls alongside the new arms with the original frozen positional weights.
Use the original calibrated fixed Q16/2 positional archive, not the gated
arm. Compare against matched-Q archived zero reserve separately.

**Advance** only if at least two matched Qs have *both* strictly more correct
requests and no lower unrounded mean request TPF (N/F), AND accuracy is no
lower at the remaining Q. Ties in quality are not wins. Include bootstrap,
selection, verification and commit in F; exclude prompt prefill. Zero reserve,
strict hits, physical work, shortlist recall and token efficiency do not
replace this primary rule. No refit on frozen30, prior, gate, tuning or new seed.

Reuse the embedding-context protocol and its calibration24-only logistic fit.
Only the reserved old-row ranking changes. Keep mandatory fresh (including
overflow), first MASK, fixed old-seat cap, remaining MASK fill, neutral old
backfill and lower-position tie-break exactly as fixed positional hazard.
Compute the three context columns from the live pre-step accepted canvas and
the preceding round's actual writes on physically queried editable rows. No
native trajectory, proposal logits, KV, shadow forward or counterfactual
acceptance is available to the selector. Only picked physical rows execute
on a sparse step. Store private coefficients and raw per-request runs outside
the checkout; do not publish IDs, embeddings, projected vectors or weights.

## Parent handoff (not executed by the CPU preparation task)

On n164, use a new TRACE_OUT. The existing supervisor handles holder checks,
staging and native qualification. Q16's matched archive is
`../t2t-native-1125/hazard24-v1-llada21` (select `hard_hazard_fixed`).

```sh
R=/data1/project_data/gx-erqw-20261123/runtime
D="$R/t2t-native-1125"
unset T2T_CALIBRATE T2T_DUAL
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export T2T_ONLINE=1 TRACE_OUT="$D/context-fixed-v1-llada21"
export T2T_HAZARD_WEIGHTS="$R/t2t-experiments/experiments/t2t/calibrated/weights.json"
export T2T_CONTEXT_WEIGHTS="$D/context-fixed-private/calibration24.json"
export T2T_ARMS='4:0:3:hard_hazard_fixed 4:0:3:hard_context_fixed 8:0:7:hard_hazard_fixed 8:0:7:hard_context_fixed 16:0:14:hard_context_fixed'
"$R/venv/bin/python" -B "$D/launch_calibration.py" llada21 4
```

The shared CLI equivalents are `--hazard-weights`, `--context-weights` and
one `--arm` per arm above. Run metadata records coefficient hashes only.
Private prep was already done. To reproduce to a **new** outside-checkout
file (the preparer refuses overwrite), from the repo root:

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$R/venv/bin/python" -B -m experiments.t2t.calibrated.embedding_context.prepare \
  --cal "$D/calibration24-v1-llada21" --checkpoint /data0/ysy/models/LLaDA2.1-mini \
  --out "$D/context-fixed-private/calibration24-repro.json"
```
