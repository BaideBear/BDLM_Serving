"""Record native decisions. No extra sampling or model calls; full logits never leave the GPU."""
import gzip
import json
from contextlib import contextmanager


@contextmanager
def writer(out, task):
    path = out / f"{task['index']:02d}.trace.jsonl.gz"
    tmp = path.with_suffix(".partial")
    with gzip.open(tmp, "xt", compresslevel=2) as f:
        def emit(**record):
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        emit(schema=1, **{k: task[k] for k in ("index", "task_id", "dataset", "seed")})
        yield emit
    tmp.rename(path)


def values(tensor):
    assert tensor.shape[0] == 1
    return tensor[0].detach().cpu().tolist()


def gemma_trace(out):
    @contextmanager
    def capture(model, task):
        with writer(out, task) as emit:
            original = model._denoising_step

            def step(**kw):
                result = original(**kw)
                after, draft, logits, _ = result
                # Batch one: an already finished row must never be observed as a denoising call.
                # Confidence proxy from already-produced, dtype-cast self-conditioning logits.
                top = logits.topk(2, dim=-1).values.float()
                emit(block=int(kw["input_ids"].shape[1]), clock=int(kw["cur_step"]),
                     before=values(kw["current_canvas"]), after=values(after),
                     accepted=values(kw["sampler"].accepted_token_mask), draft=values(draft),
                     certainty=values(top[..., 0] - top[..., 1]))
                return result

            model._denoising_step = step
            try:
                yield
            finally:
                del model._denoising_step
    return capture


def llada_request(engine, state, facade, emit):
    original = facade._sample_with_temperature_topk_topp
    pending = []
    def sample(logits, **kwargs):
        ids, conf = original(logits, **kwargs)
        pending.append((ids, conf))
        return ids, conf
    facade._sample_with_temperature_topk_topp = sample
    try:
        for _ in range(4096):
            if state.phase == "done":
                return
            start = state.block * state.block_length
            sl = slice(start, start + state.block_length)
            before = state.x[:, sl].clone() if state.phase == "denoise" else None
            phase = engine.step(state)
            if phase == "denoise":
                ids, conf = pending.pop()
                after = state.x[:, sl]
                emit(block=start, clock=state.step_in_block, before=values(before), after=values(after),
                     accepted=values(after != state.mask_id), draft=values(ids), certainty=values(conf))
            if pending:
                raise ValueError("One sample expected per denoise step")
        raise RuntimeError("Reference action limit exceeded")
    finally:
        del facade._sample_with_temperature_topk_topp
