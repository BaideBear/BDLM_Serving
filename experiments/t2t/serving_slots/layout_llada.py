"""Pooled frozen90 LLaDA ablations. One model per worker; only compact metrics are public."""
import argparse
import fcntl
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

from .layout_sweep import write


def main():
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "samples", "reference", "output", "control"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--deadline", required=True, type=int)
    a = p.parse_args()
    import torch
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score
    from bdlm_serving.methods.t2t.budget_family import Law
    from bdlm_serving.methods.t2t.history_llada import LLadaHistory
    from tools.run_history_llada import load_model, load_reference, BatchedHistoryRunner, make_request, package_request

    def ready():
        (a.control/"ready").touch(exist_ok=False)
        while not (a.control/"go").exists():
            if time.time() >= a.deadline:
                raise TimeoutError("GPU handoff expired")
            time.sleep(.1)
    opts = SimpleNamespace(checkpoint=a.checkpoint, prepare_script=None, reference=a.reference)
    tok, model = load_model(opts, before_cuda=ready)
    assert model.device.type == "cuda" and torch.cuda.device_count() == 1
    tasks = load_tasks(a.samples)
    pins = {t["index"]: load_reference(opts, tok, t) for t in tasks}
    adapter = LLadaHistory(model)
    mask, eos = tok.convert_tokens_to_ids("<|mask|>"), [tok.eos_token_id]
    special = set(tok.all_special_ids) | set(eos) | {mask}
    parts = [[x+j for x in (0,10,20) for j in range(5)],
             [x+j for x in (5,15,25) for j in range(5)]] + [list(range(i,i+15)) for i in (30,45,60,75)]
    # The unchanged policy is measured on this host as a paired control.
    arms = [(50,8),(0,8),(50,4),(100,8),(0,4),(50,16)]
    for part in parts:
        for c in (16,8,4):
            for cold, den in arms:
                if time.time() >= a.deadline:
                    return
                key = f"cold{cold}-pref1of{den}-C{c}"
                def name(i):
                    return a.output / f"{key}-{i:02d}.json"
                if all(name(i).exists() for i in part):
                    continue
                with (a.output/f"{key}-group{part[0]}.lock").open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if all(name(i).exists() for i in part):
                        continue
                    print("START", key, part, flush=True)
                    req = [make_request(model, i, dict(tasks[i], _mask_id=mask, _eos_ids=eos),
                                        pins[i][1], generation_cap=pins[i][0]["generation_cap"]) for i in part]
                    with torch.inference_mode():
                        runner = BatchedHistoryRunner(model, adapter, req, budget=c, batch_size=len(req),
                              bucket="pooled", cold_share=cold, law=Law(1,den))
                        runner.run()
                    for request in req:
                        i = request.index
                        r = package_request(request,tok,special)
                        row = {k:r[k] for k in ("index","task_id","seed","dataset","output_tokens",
                             "output_text","output_ids","generation_forwards","generation_physical_padded_rows",
                             "denoise_history_rows","commit_history_rows","truncated","history_q_per_forward")}
                        assert row["denoise_history_rows"]+row["commit_history_rows"] == sum(row["history_q_per_forward"])
                        row.update(budget=c,cold_share=cold,prefix_den=den,history_query_rows=r["history_rows"],
                            layout="split",local_attention_Q=r["generation_query_rows"],host=socket.gethostname(),
                            gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),samples_sha256=sha(a.samples))
                        try:
                            row["correct"]=bool(score(r["output_text"],tasks[i])["correct"])
                        except (RuntimeError,FileNotFoundError) as e:
                            row["score_pending"]=str(e)
                        write(name(i),row)
                    print("DONE",key,part,flush=True)
                    del req, runner, request, r


if __name__ == "__main__":
    main()
