# Input-embedding context ablation (frozen before evaluation)

Pass only if total actual reserved strict old-edit hits increase at two or
more of Q4/1, Q8/1, Q16/2 and do not fall at any. Ties are not gains. Top-1,
4 and 8 shortlist hits are descriptive, not an alternative pass rule. One
projection, one seed, one radius, one fit per arm; no quantity or threshold
tuning and no alternate embeddings. Frozen30 is not a new test population.

Use the native untied INPUT table `model.word_embeddings.weight` from the
existing LLaDA 2.1 snapshot, not `lm_head.weight`. Project BF16 rows in FP32
with a fixed `[input_width,16]` FP32 matrix: NumPy Generator(PCG64(20261126)),
`integers(0, 2, size=(width,16), dtype=np.int8)` in C order, map 0 to -1 and
1 to +1 and divide by sqrt(input_width). L2-normalize each projected row;
a zero vector stays zero. Generate once, read individual needed input rows
on CPU and cache their unit vectors only in memory. No token list, projection,
input rows, sketch vectors or fitted weights are to be published.

For each eligible old position i, N is the set of editable positions j in the
same block with 1 <= |i-j| <= 8 that were admitted or strictly rewritten in
the immediately preceding round. Use the token IDs visible *before* this
decision; a revocation, unchanged token or prompt position is not a write.
Append exactly three columns to the 25 positional columns: (1) maximum
projected cosine between i and N (0 if empty; pair score 0 for a zero vector;
retain negative maxima), (2) that same score times the existing nearest-MASK
distance <=4 indicator (`near_mask[2]`, positional column 6, zero-based), and
(3) presence of any such neighbour. Presence is independent of vector norm.
Reset fresh at block boundaries. No current transition, draft, logits, KV,
future write, trainable token-ID features, or lexical/history arm.

Use the shared disjoint calibration24 vs verified native frozen30 comparison:
same unscaled features, L2=2 on slopes, unpenalized intercept, identical seat
allocation, lower-position tie-break and teacher-forced native history. Fit
only on selected old rows in the 24 calibration traces. Report identical
seat counts, strict old-edit hits and differences on the 30 requests. Costs:
separate the CPU input-row cache misses/projection from warm feature computation
on calibration states. No sparse rollout, quality test, downloads, GPU or commit.
