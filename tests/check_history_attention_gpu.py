"""GPU-only regression: real head widths, ragged bounds, partial/stale/cold rows."""
import torch

from bdlm_serving.methods.t2t.history_attention import SlotPlan, block_history, state_like


def ref(q,k,v,bounds,scale):
    kvh = q.shape[1]//k.shape[1]
    score = (q.float()@k.float().repeat_interleave(kvh,1).transpose(-1,-2))*scale
    pos = torch.arange(k.shape[2],device=q.device)
    legal = (pos>=bounds[:,None,:,0,None]) & (pos<bounds[:,None,:,1,None])
    score = score.masked_fill(~legal,-torch.inf)
    m = score.amax(-1)
    pivot = torch.where(m.isfinite(),m,0.)
    e = (score-pivot[...,None]).exp()
    return m,e.sum(-1),e@v.float().repeat_interleave(kvh,1)


def merge(local,hist):
    m = torch.maximum(local[0],hist[0])
    m = torch.where(m.isfinite(),m,0.)
    a,b = (local[0]-m).exp(),(hist[0]-m).exp()
    u = a[...,None]*local[2]+b[...,None]*hist[2]
    l = a*local[1]+b*hist[1]
    return u/torch.where(l>0,l,1.)[...,None]


def verify(b,d,dtype):
    device = "cuda"
    r,h,g,s = 3,4,2,103
    def rand(shape):
        return torch.randn(shape,device=device,dtype=dtype)
    k,v = (rand((r,g,b,d)) for _ in range(2))
    hk,hv = (rand((r,g,s,d)) for _ in range(2))
    q = rand((r,h,b,d))
    state = state_like(q)
    truth = tuple(x.clone() for x in state)
    bounds = torch.zeros((r,b,2),device=device,dtype=torch.int32)
    bounds[:,:,1] = b
    hb = bounds.clone()
    hb[0,:,1] = s
    hb[1,:,0] = 7
    hb[1,:,1] = 57
    hb[2,:,:] = 0
    bounds[2,-3:,:] = 0
    scale = d**-.5
    pat = [list(range(b)),list(range(2,b,3)),[]]
    maximum = 0.
    for selected in (pat,[[],[1,b-1],[0]], [list(range(b)) for _ in range(r)]):
        q = rand(q.shape)
        before = tuple(x.clone() for x in state)
        out = block_history(q,k,v,hk,hv,bounds,hb,SlotPlan.make(selected,b,device),state,scale)
        hpart = ref(q,hk,hv,hb,scale)
        keep = torch.ones((r,b),device=device,dtype=torch.bool)
        for req, ids in enumerate(selected):
            if ids:
                keep[req,ids] = False
        for j in range(3):
            mask = keep[:,None,:] if j<2 else keep[:,None,:,None]
            assert torch.equal(state[j].masked_select(mask),before[j].masked_select(mask))
            truth[j].copy_(torch.where(mask,truth[j],hpart[j]))
            assert state[j].dtype == torch.float32
            if j < 2:
                torch.testing.assert_close(state[j],truth[j],atol=5e-5,rtol=3e-5)
            else:
                # Unnormalized numerator uses rounded tensor-core weights. Bound
                # the error in output units, including rows near cancellation.
                norm = truth[1].clamp_min(1)[...,None]
                torch.testing.assert_close(state[j]/norm, truth[j]/norm,
                                           atol=.004 if dtype==torch.bfloat16 else .0006, rtol=.002)
        target = merge(ref(q,k,v,bounds,scale),truth)
        torch.testing.assert_close(out.float(),target,atol=.012,rtol=.025)
        maximum = max(maximum,(out.float()-target).abs().max().item())
    print(f"pass {dtype} B={b} D={d} max_abs={maximum:.6g}",flush=True)
    bounds[...,0], bounds[...,1] = 0,b
    hb[...,0], hb[...,1] = 0,s
    for sel in (pat,[list(reversed(range(b))) for _ in range(r)]):
        plan = SlotPlan.make(sel,b,device)
        prior, fast = state_like(q),state_like(q)
        expected = block_history(q,k,v,hk,hv,bounds,hb,plan,prior,scale)
        got = block_history(q,k,v,hk,hv,None,None,plan,fast,scale)
        torch.testing.assert_close(got.float(),expected.float(),atol=.009,rtol=.01)
        torch.testing.assert_close(fast[0],prior[0],atol=.0001,rtol=.0001)
        torch.testing.assert_close(fast[1],prior[1],atol=.0001,rtol=.0001)
        norm = prior[1].clamp_min(1)[...,None]
        torch.testing.assert_close(fast[2]/norm,prior[2]/norm,atol=.004,rtol=.005)


if __name__=="__main__":
    torch.manual_seed(17000)
    torch.backends.cuda.matmul.allow_tf32 = False
    for dtype in (torch.bfloat16,torch.float16):
        for case in ((32,128),(256,256),(256,512)):
            verify(*case,dtype)
