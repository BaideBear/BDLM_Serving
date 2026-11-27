"""One physical block. Partial softmax state is private to a layer and cache epoch.

Input layout [request, head, position, head_dim], contiguous, post-RoPE.
Bounds specify two half-open KV ranges per row in the history and block views,
respectively; the caller supplies model-native visibility and logit scaling.
SlotPlan can be reused across layers with the same active requests.
"""
from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl


_M = 16


@triton.jit
def _partial(Q, K, V, Owners, Rows, Bounds, Max, Norm, Num, BM, BL, BU, Out,
             H: tl.constexpr, G: tl.constexpr, B: tl.constexpr,
             S: tl.constexpr, D: tl.constexpr, scale, PACKED: tl.constexpr,
             M: tl.constexpr, N: tl.constexpr, FULL: tl.constexpr, FINISH: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    if PACKED:
        req = tl.load(Owners+t)
        row = tl.load(Rows+t*M+tl.arange(0, M))
    else:
        req = t // tl.cdiv(B, M)
        row = (t % tl.cdiv(B, M))*M + tl.arange(0, M)
    live = (row >= 0) & (row < B)
    rr = tl.where(live, row, 0)
    d = tl.arange(0, D)
    iq = (req*H+h)*B+rr
    q = tl.load(Q+iq[:, None]*D+d[None, :], mask=live[:, None], other=0.)
    if not FULL:
        offset = (req*B+rr)*2
        lo = tl.load(Bounds+offset, mask=live, other=0)
        hi = tl.load(Bounds+offset+1, mask=live, other=0)
    kbase = (req*G + h//(H//G))*S
    m = tl.full((M,), -float("inf"), tl.float32)
    l = tl.full((M,), 0., tl.float32)
    u = tl.full((M,D), 0., tl.float32)
    if FULL:
        first, last = 0, tl.cdiv(S,N)
    else:
        first = tl.min(tl.where(live & (hi > lo), lo, S), 0)//N
        last = tl.cdiv(tl.max(tl.where(live, hi, 0), 0), N)
    for i in range(first, last):
        n = i*N+tl.arange(0,N)
        kp = (kbase+n[:,None])*D+d[None,:]
        k = tl.load(K+kp, mask=n[:,None]<S, other=0.)
        v = tl.load(V+kp, mask=n[:,None]<S, other=0.)
        x = tl.dot(q, tl.trans(k), out_dtype=tl.float32)*scale
        if FULL:
            if S % N:
                x = tl.where(n[None,:] < S, x, -float("inf"))
        else:
            valid = (n[None,:]>=lo[:,None]) & (n[None,:]<hi[:,None])
            x = tl.where(valid, x, -float("inf"))
        nm = tl.maximum(m, tl.max(x,1))
        if FULL:
            pivot = nm
        else:
            pivot = tl.where(nm == -float("inf"), 0., nm)
        a = tl.exp(m-pivot)
        p = tl.exp(x-pivot[:,None])
        # Match FA2: tensor-core operands use the model dtype; accumulate in fp32.
        u = tl.dot(p.to(v.dtype),v,acc=u*a[:,None],out_dtype=tl.float32)
        l = l*a + tl.sum(p,1)
        m = nm
    tl.store(Max+iq, m, mask=live)
    tl.store(Norm+iq, l, mask=live)
    tl.store(Num+iq[:,None]*D+d[None,:], u, mask=live[:,None])
    if FINISH:
        block_m = tl.load(BM+iq, mask=live, other=-float("inf"))
        peak = tl.maximum(block_m,m)
        peak = tl.where(peak == -float("inf"), 0., peak)
        a, t = tl.exp(block_m-peak), tl.exp(m-peak)
        total_l = a*tl.load(BL+iq,mask=live,other=0.)+t*l
        total_u = a[:,None]*tl.load(BU+iq[:,None]*D+d[None,:],mask=live[:,None],other=0.)+t[:,None]*u
        tl.store(Out+iq[:,None]*D+d[None,:],total_u/tl.where(total_l>0,total_l,1.)[:,None],mask=live[:,None])


@triton.jit
def _merge(BM, BL, BU, HM, HL, HU, Out, D: tl.constexpr):
    i = tl.program_id(0)
    d = tl.arange(0,D)
    bm, hm = tl.load(BM+i), tl.load(HM+i)
    z = tl.maximum(bm,hm)
    z = tl.where(z == -float("inf"),0.,z)
    a, b = tl.exp(bm-z), tl.exp(hm-z)
    l = a*tl.load(BL+i) + b*tl.load(HL+i)
    u = a*tl.load(BU+i*D+d) + b*tl.load(HU+i*D+d)
    tl.store(Out+i*D+d, u/tl.where(l>0,l,1.))


@dataclass
class SlotPlan:
    owners: torch.Tensor
    rows: torch.Tensor
    counts: tuple
    width: int

    @classmethod
    def make(cls, selected, width, device):
        tiles, owners, counts = [], [], []
        for req, seq in enumerate(selected):
            ids = list(seq)
            if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < width for i in ids):
                raise ValueError("expected distinct in-block row IDs per request")
            counts.append(len(ids))
            for start in range(0, len(ids), _M):
                sub = ids[start:start+_M]
                tiles.extend(sub+[-1]*(_M-len(sub)))
                owners.append(req)
        return cls(torch.tensor(owners, device=device, dtype=torch.int32),
                   torch.tensor(tiles, device=device, dtype=torch.int32), tuple(counts), width)


def state_like(q):
    opts = dict(device=q.device, dtype=torch.float32)
    return (torch.full(q.shape[:-1], -math.inf, **opts), torch.zeros(q.shape[:-1], **opts),
            torch.zeros(q.shape, **opts))


@torch.no_grad()
def block_history(q, bk, bv, hk, hv, block_bounds, hist_bounds, plan, history, scale):
    """Updates only selected historical rows in place; never concatenates KV.

    None for either bounds tensor means *every* row sees that entire partition
    (no padding, clipping, or window). Otherwise bounds are int32. Padded rows
    must have empty ranges and an empty cache. Reset history on block/epoch
    changes; do not reuse a SlotPlan after repacking the batch.
    """
    r,h,b,d = q.shape
    bounds = tuple(x for x in (block_bounds,hist_bounds) if x is not None)
    tensors = (q,bk,bv,hk,hv,*bounds,*history,plan.owners,plan.rows)
    if not q.is_cuda or not all(x.device == q.device and x.is_contiguous() for x in tensors):
        raise ValueError("expected contiguous tensors on one CUDA device")
    if q.dtype not in (torch.float16,torch.bfloat16) or not all(x.dtype==q.dtype for x in (bk,bv,hk,hv)):
        raise ValueError("expected homogeneous fp16/bf16")
    if b < 1 or d not in (64,128,256,512) or bk.shape != bv.shape or hk.shape != hv.shape:
        raise ValueError("invalid block/history KV dimensions")
    g,s = hk.shape[1:3]
    if g < 1 or h%g or bk.shape!=(r,g,b,d) or hk.shape!=(r,g,s,d):
        raise ValueError("incompatible GQA shape")
    if any(x.shape!=(r,b,2) or x.dtype!=torch.int32 for x in bounds):
        raise ValueError("bounds must be [batch,block,2] int32")
    if (len(history)!=3 or any(x.dtype!=torch.float32 for x in history) or
        tuple(x.shape for x in history)!=(q.shape[:-1],q.shape[:-1],q.shape)):
        raise ValueError("invalid cached (m,l,u)")
    if plan.width!=b or len(plan.counts)!=r:
        raise ValueError("slot plan does not match batch")
    work = tuple(torch.empty_like(x) for x in history)
    out = torch.empty_like(q)
    with torch.cuda.device(q.device):
        dense = all(i==b for i in plan.counts)
        def launch(k,v,bounds,target,s,packed,finish=False):
            m = 32 if not packed and bounds is None and b>=128 and d==256 else _M
            tasks = plan.owners.numel() if packed else r*triton.cdiv(b,m)
            n = 64 if bounds is None and s>=64 and d<=256 else 32
            _partial[(tasks,h)](q,k,v,plan.owners,plan.rows,q if bounds is None else bounds,
                  *target,*work,out,h,g,b,s,d,scale,packed,m,n,bounds is None,finish,num_warps=4)
        launch(bk,bv,block_bounds,work,b,False)
        if plan.owners.numel():
            launch(hk,hv,hist_bounds,history,s,not dense,finish=dense)
        if not dense:
            _merge[(r*h*b,)](*work,*history,out,d,num_warps=4)
    return out
