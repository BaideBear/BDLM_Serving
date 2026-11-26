import pytest
import torch

from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel
from bdlm_serving.methods.t2t.verified_commit import VerifiedCommitKernel
from experiments.t2t.serving_slots.promotion90 import check_pair
from test_t2t_llada_sparse import _Toy, _state


def setup(cls, q=8, **state_args):
    model = _Toy(edit=True).eval()
    kernel = cls(model,q,0,q,policy="hard_prefix_zero")
    return model, kernel, _state(kernel,prompt_length=58,gen_length=38,**state_args)


def equal_kv(a,b):
    assert a.get_seq_length() == b.get_seq_length()
    for kl,kr in zip(a.to_legacy_cache(),b.to_legacy_cache(),strict=True):
        for left,right in zip(kl,kr,strict=True):
            assert torch.equal(left,right)


@pytest.mark.parametrize("q",[4,8,16,32])
def test_promoting_verify_skips_a_real_projection_only_after_unchanged_full_input(q):
    a,ka,sa = setup(LLaDASparseKernel,q)
    b,kb,sb = setup(VerifiedCommitKernel,q)
    for _ in range(240):
        if sa.phase=="done":
            break
        n=len(b.embedding_rows)
        action=ka.step(sa)
        assert kb.step(sb)==action
        assert (sa.phase,sa.block,sa.committed_end,sa.context_version)==(
                sb.phase,sb.block,sb.committed_end,sb.context_version)
        assert torch.equal(sa.x,sb.x) and torch.equal(sa.rng_cpu,sb.rng_cpu)
        if action=="commit":
            assert sb.commits[-1]["promoted"] and n==len(b.embedding_rows)
            equal_kv(sa.cache,sb.cache)
    assert sa.phase==sb.phase=="done" and len(sb.commits)==2
    assert len(a.embedding_rows)-len(b.embedding_rows)==2
    assert sum(a.embedding_rows)-sum(b.embedding_rows)==64


@pytest.mark.parametrize("changed",["tokens","resident","qualification"])
def test_invalid_verification_must_pay_for_commit(changed):
    model,k,s=setup(VerifiedCommitKernel)
    for _ in range(120):
        if s.phase=="commit":
            break
        k.step(s)
    assert s.phase=="commit" and s.verifier_input
    if changed=="tokens":
        s.x[0,s.block*32+31]=3 if int(s.x[0,s.block*32+31])!=3 else 1
    elif changed=="resident":
        s.resident=None
    else:
        s.events[-1]["edited"]=1
    n=len(model.embedding_rows)
    assert k.step(s)=="commit" and not s.commits[-1]["promoted"]
    assert len(model.embedding_rows)==n+1 and model.embedding_rows[-1]==32


def test_gpu_study_counts_real_work_and_retains_divergent_outputs():
    records=[]
    for cls in (LLaDASparseKernel,VerifiedCommitKernel):
        _,kernel,state=setup(cls)
        kernel.run(state)
        rows=[e["physical_q"] for e in state.events if e["phase"]!="prefill"]
        records.append(dict(events=state.events,logical_commits=getattr(state,"commits",[]),
                            generation_forwards=len(rows),generation_physical_padded_rows=sum(rows),
                            output_ids=kernel.output(state)[0].tolist(),output_tokens=0,output_text="",
                            eos=False,truncated=True,t2t_edits=sum(e.get("edited",0) for e in state.events)))
    ref,arm=records
    assert check_pair(ref,arm)==dict(exact_path=True,exact_output=True,promoted=2,recomputed=0)
    assert not check_pair(ref,dict(arm,output_text="changed"))["exact_output"]
    with pytest.raises(ValueError,match="physical work"):
        check_pair(ref,dict(arm,generation_forwards=ref["generation_forwards"]))
