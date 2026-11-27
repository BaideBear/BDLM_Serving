"""Opt-in GPU qualification path for NEW fixed-budget experiments only.

Use run_fixed with already loaded raw weights. Callers own GPU handoff,
checkpoint, sample registration, scoring and independent single-lane receipts.
"""
from contextlib import ExitStack
import time

import torch
from torch.nn import functional as F

from .row_wave import waves


def projection_audit(stack, modules, size, width):
    visits = [0]*len(modules)
    def observe(i, args):
        if args[0].ndim != 3 or tuple(args[0].shape[:2]) != (size, width):
            raise RuntimeError("physical transformer shape disagrees with launch")
        visits[i] += 1
    for i, module in enumerate(modules):
        stack.enter_context(module.register_forward_pre_hook(lambda _, a, i=i: observe(i, a)))
    return visits


def llada_fn(model, item, offer, arm):
    from bdlm_serving.methods.t2t.llada_budget_request import LLaDABudgetRequest

    def evaluate(issue):
        class Request(LLaDABudgetRequest):
            def _project(self, rows, cap, phase):
                if not rows or len(rows) > cap or tuple(sorted(set(rows))) != rows:
                    raise ValueError("invalid row plan")
                idx = self.x.new_tensor(rows)
                self.bank.begin(idx, rows)
                logits, receipt = issue("prefill" if phase == "prefill" else "decoder",
                                        len(rows), (self, rows, idx))
                self.bank.initialized.update(rows)
                return dict(phase=phase, block=self.head//32, absolute_positions=list(rows),
                            layer_physical_q=[len(rows)]*20, batch_receipt=receipt), logits

        with torch.inference_mode():
            request = Request(model, item["input_ids"], mask_id=item["mask_id"],
                              eos_ids=item["eos_ids"], seed=item["seed"], ablation=arm,
                              max_new_tokens=item.get("max_new_tokens", 2048))
            while request.status == "ready":
                request.step(offer)
            return dict(ids=request.tokens, trace=request.records, prefill=request.prefill_record)
    return evaluate


def llada_forward(model, group, width):
    from bdlm_serving.runtime.batch import BatchCache
    jobs, ids, pos, masks = [], [], [], []
    ends = [c.data[0].bank.visible_end for c in group]
    kpad = max(ends)
    for c, end in zip(group, ends):
        req, rows, idx = c.data
        w = len(rows)
        jobs.append(dict(q=w, kv=end, execution_cache=req.bank))
        ids.append(F.pad(req.x.index_select(1, idx), (0, width-w)))
        pos.append(F.pad(idx[None], (0, width-w)))
        k = torch.arange(end, device=idx.device)
        mask = F.pad((idx[:,None]//32 >= k[None,:]//32), (0,kpad-end,0,width-w))
        mask[w:,0] = True
        masks.append(mask[None,None])
    cache = BatchCache(jobs, width, kpad)
    modules = [layer.attention.query_key_value for layer in model.model.layers]
    if len(modules) != 20:
        raise ValueError("expected pinned LLaDA2.1 checkpoint")
    with ExitStack() as stack:
        visits = projection_audit(stack, modules, len(group), width)
        emb = model.get_input_embeddings()(torch.cat(ids))
        logits = model(inputs_embeds=emb, position_ids=torch.cat(pos), attention_mask=torch.cat(masks),
                       use_cache=True, past_key_values=cache, output_attentions=True).logits
    if visits != [1]*20 or any(c.data[0].bank.writes != [(i,c.width,c.width) for i in range(20)]
                               for c in group):
        raise RuntimeError("missing or widened per-lane KV writes")
    return logits


def run_fixed(model, kind, items, *, q, arm, max_active, target_rows=512, telemetry=None):
    """Return results in input order. No automatic replay after a partial write.

    Each item has `seed` and either LLaDA input_ids/mask_id/eos_ids or Gemma
    processor `data`. Never include two concurrent calls from the same request.
    """
    if model.training or model.device.type != "cuda" or arm not in ("00","10","11"):
        raise ValueError("requires a CUDA eval checkpoint and registered arm")
    if kind == "llada21" and q in (4,8,16,32,64):
        fn, forward = llada_fn, llada_forward
    elif kind == "gemma" and q in (32,64,128,256,512):
        from .packed_gemma import gemma_fn as fn, gemma_forward as forward
    else:
        raise ValueError("unregistered model or fixed offer")
    calls = []
    @torch.inference_mode()
    def execute(group):
        n, w = len(group), max(c.width for c in group)
        begin = time.monotonic()
        if group[0].phase == "encoder":
            if n != 1:
                raise RuntimeError("native encoder requires an independent cache")
            out = [group[0].data()]
        else:
            logits = forward(model, group, w)
            if tuple(logits.shape[:2]) != (n,w):
                raise RuntimeError("widened or missing batch rows")
            # A lane must not hold a larger batch's logits alive across turns.
            out = [(logits[i:i+1,:c.width].clone(),len(calls)) for i,c in enumerate(group)]
        torch.cuda.synchronize(model.device)
        receipt = dict(id=len(calls), phase=group[0].phase, lanes=[c.lane for c in group],
                       widths=[c.width for c in group], batch=n, qpad=w, paid_rows=n*w,
                       used_rows=sum(c.width for c in group), wall_s=time.monotonic()-begin)
        calls.append(receipt)
        if telemetry:
            telemetry(receipt)
        return out
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(model.device)
    results = waves([fn(model, item, q, arm) for item in items], execute,
                    max_active=max_active, target=target_rows)
    torch.cuda.synchronize(model.device)
    free, total = torch.cuda.mem_get_info(model.device)
    return dict(results=results, calls=calls, seconds=time.monotonic()-start,
                peak_reserved=torch.cuda.max_memory_reserved(model.device),
                end_free=free, device_bytes=total, target_rows=target_rows, max_active=max_active)
