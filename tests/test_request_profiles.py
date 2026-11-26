"""Both user-facing profiles must preserve outer control and the shared accounting."""
import pytest
import torch

from bdlm_serving.methods.t2t.serving_profiles import gemma_plan, open_gemma, open_llada
from experiments.t2t.serving_slots import request_handle as raw
from experiments.t2t.serving_slots.llada_spawn import SpawnGatedKernel
from test_llada_all_slots import TapedCPUModel, CFG
from test_gemma_request import B, inputs, model_factory, plan


def drive(handle, offers):
    tape=[]
    try:
        while handle.status=="ready":
            menu=handle.menu()
            q=offers[len(tape)%len(offers)]
            assert menu[q].offer==q and len(menu)==5
            tape.append(handle.advance(menu[q]))
            assert len(tape)<300
        return handle.result(),tape
    finally:
        handle.close()


@pytest.mark.parametrize("name,threshold",[("economy",None),("progress80_candidate",.8)])
def test_llada_profiles_match_reference_with_switches(name,threshold):
    x=torch.full((1,34),5)
    opts=dict(gen_length=64,seed=29,mask_id=0,eos_ids=())
    offers=(64,8,64,16,32,4)
    a=open_llada(TapedCPUModel().eval(),x,CFG,profile=name,**opts)
    k=SpawnGatedKernel(TapedCPUModel().eval(),spawn_threshold=threshold)
    b=raw.open_llada(k,x,CFG,**opts)
    assert drive(a,offers)==drive(b,offers)


@pytest.mark.parametrize("name", ["economy","q128_prefix_candidate"])
def test_gemma_profiles_match_reference_with_switches(model_factory,name):
    base=plan(2*B)
    offers=(512,512,128,32,64,256,512,32)
    a=open_gemma(model_factory(),inputs(5),base,seed=23,profile=name)
    b=raw.open_gemma(model_factory(),inputs(5),gemma_plan(base,name),seed=23,causal=True)
    assert drive(a,offers)==drive(b,offers)
