import json
from types import SimpleNamespace

from bdlm_serving.methods.t2t.protocol import Arm
from tools.run_t2t import _run_matrix


def test_warm_timing_is_separate_and_does_not_replace_task_zero(tmp_path,monkeypatch):
    import torch
    from bdlm_serving.methods.t2t import online
    calls, fences = [], []
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda: "cpu-stub")
    monkeypatch.setattr(torch.cuda,"synchronize",lambda: fences.append(1))
    tasks = [dict(index=i, task_id=str(i), dataset="test", seed=i+13) for i in range(2)]
    def backend(task,arm):
        calls.append((task["index"],arm))
        return dict(output_ids=[7],output_tokens=1,generation_forwards=1,generation_query_rows=4,
                    generation_physical_padded_rows=4,max_q=4,tpf=1.,token_efficiency=.25,status="success")
    for task in tasks:
        ref = {**backend(task,None),**task}
        (tmp_path/f"{task['index']:02d}-native32.json").write_text(json.dumps(ref))
    calls.clear()
    monkeypatch.setattr(online,"load_backend",lambda *a,**k: backend)
    args = SimpleNamespace(output=tmp_path/"run",reference=tmp_path,model="llada21",time_requests=True)
    arm = Arm(4,0,4,"hard_prefix_zero")
    _run_matrix(args,None,tasks,[arm],dict(suite="frozen30",samples_sha256="test"),32,0,None)
    assert calls == [(0,None),(1,None),(0,arm),(0,arm),(1,arm)] and len(fences)==4
    timed = json.loads((args.output/"timing"/f"{arm.policy}.json").read_text())["requests"]
    assert [r["index"] for r in timed] == [0,1]
    assert all(r["wall_ns"]>0 and (r["N"],r["F"],r["physical_Q"])==(1,1,4) for r in timed)
    for i in range(2):
        row = json.loads((args.output/f"{i:02d}-{arm.policy}.json").read_text())
        assert "wall_ns" not in row and row["status"]=="success"
