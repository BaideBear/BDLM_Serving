"""GPU microbenchmark. Compare complete split attention to official FA2.

Prepacked device inputs, same visible keys, contiguous native layouts on both
sides. No model inference, selector, host/device copies, compilation or padding.
Keeps graph outputs alive, interleaves both implementations; latencies in us.
"""
import argparse
import json
import math
import random
import statistics

import torch
import flash_attn
from flash_attn import flash_attn_func
from bdlm_serving.methods.t2t.history_attention import SlotPlan, block_history, state_like


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(6):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        result = fn()
    return g, result


def measure(methods, rounds=15, reps=30):
    graphs = {k: graph(v) for k, v in methods.items()}
    times = {k: [] for k in methods}
    order = list(methods)
    rng = random.Random(123)
    for i in range(rounds + 2):
        rng.shuffle(order)
        for name in order:
            g, _ = graphs[name]
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(reps):
                g.replay()
            end.record()
            end.synchronize()
            if i >= 2:
                times[name].append(start.elapsed_time(end)*1000/reps)
    return {k: {"p50_us": round(statistics.median(v), 2),
                "p95_group_mean_us": round(sorted(v)[math.ceil(.95*len(v))-1], 2)}
            for k, v in times.items()}


@torch.inference_mode()
def case(b, d, g, c, s):
    r, h = max(1, 512//c), 16
    def rand(n, heads):
        return torch.randn((r, heads, n, d), device="cuda", dtype=torch.bfloat16)
    q, bk, bv, hk, hv = rand(b,h), rand(b,g), rand(b,g), rand(s,g), rand(s,g)
    # All keys visible (right boundary unrestricted); valid sliding-layer
    # control only when the model window would expose this same union.
    full_rows = [list(range(b)) for _ in range(r)]
    k = min(c,b)
    rows = [sorted(random.Random(i+77).sample(range(b),k)) for i in range(r)]
    plan = SlotPlan.make(rows,b,q.device)
    state = state_like(q)
    all_plan = SlotPlan.make(full_rows,b,q.device)
    def split(sel):
        return block_history(q,bk,bv,hk,hv,None,None,sel,state,d**-.5)
    full = split(all_plan)
    methods = {"split": lambda: split(plan)}
    err = None
    if d <= 256:
        fq = q.transpose(1,2).contiguous()
        fk, fv = (torch.cat((a,z),2).transpose(1,2).contiguous() for a,z in ((hk,bk),(hv,bv)))
        methods["fa2"] = lambda: flash_attn_func(fq,fk,fv,dropout_p=0.,causal=False)
        dense = methods["fa2"]().transpose(1,2)
        torch.testing.assert_close(full.float(),dense.float(),atol=.01,rtol=.02)
        err = (full.float()-dense.float()).abs().max().item()
    t = measure(methods)
    return {"b":b,"d":d,"h":h,"g":g,"c":c,"batch":r,"history":s,
            "scope":"all-visible, prepacked, warm steady-state",
            "mode":"same-work" if k==b else "sparse-vs-dense",
            "fa2_full_max_abs":err, **t,
            "split_over_fa2":round(t["split"]["p50_us"]/t["fa2"]["p50_us"],3) if "fa2" in t else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history",type=int,nargs="+",default=[128,512])
    args = parser.parse_args()
    torch.manual_seed(20261127)
    print(json.dumps({"device":torch.cuda.get_device_name(),"torch":torch.__version__,
                      "flash_attn":flash_attn.__version__}),flush=True)
    for b, d, g, budgets in ((32,128,4,(4,8,16,32,64)),
                             (256,256,8,(32,64,128,256,512)),
                             (256,512,2,(32,64,128,256,512))):
        for s in args.history:
            for c in budgets:
                print(json.dumps(case(b,d,g,c,s)),flush=True)
