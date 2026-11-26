"""Batch-one, open-loop offered traffic. A producer really enqueues during generate()."""
import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from threading import Condition, Event, Thread
from time import monotonic_ns

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.protocol import Arm, verify_work
from tools.run_t2t import durable

MODES = ("fixed_high","busy_low","busy_high","fixed_low")
SLOTS = {"gemma": (32,64),"llada21": (8,16)}


class Inbox:
    def __init__(self):
        self.cv, self.waiting, self.after, self.cancel = Condition(), deque(), [], Event()

    def put(self,index,epoch):
        with self.cv:
            actual = monotonic_ns()-epoch
            self.after.append(actual)
            self.waiting.append((index,actual))
            self.cv.notify_all()

    def pop(self):
        with self.cv:
            self.cv.wait_for(lambda: self.waiting)
            return self.waiting.popleft()

    def snapshot(self,epoch):
        with self.cv:
            return monotonic_ns()-epoch,len(self.waiting)


class Pressure:
    def __init__(self,mode,slots):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode,self.slots,self.busy = mode,slots,False

    def offer(self,depth):
        if depth < 0:
            raise ValueError("Negative FIFO waiting depth")
        if depth >= 2:
            self.busy = True
        elif depth == 0:
            self.busy = False
        low,high = self.slots
        return {"fixed_high":high,"fixed_low":low,"busy_high":high if self.busy else low,
                "busy_low":low if self.busy else high}[self.mode]


def arrival_trace(tasks,unit):
    if unit <= 0:
        raise ValueError("Positive reference time required")
    order = sorted((t["index"] for t in tasks),
                   key=lambda i:sha256(f"fifo90-v1:{i}".encode()).digest())
    # Repeated burst of six, with time for the queue to drain between bursts.
    return [(idx,(8*(rank//6)*8 + rank%6)*unit//8) for rank,idx in enumerate(order)]


def replay(trace,mode,slots,execute,sync):
    box, state = Inbox(),Pressure(mode,slots)
    epoch = monotonic_ns()
    def produce():
        for index,target in trace:
            while (remaining := epoch+target-monotonic_ns()) > 0:
                if box.cancel.wait(remaining/1e9):
                    return
            if box.cancel.is_set():
                return
            box.put(index,epoch)
    producer = Thread(target=produce,daemon=True)
    producer.start()
    rows, timing = [], []
    try:
        for index,target in trace:
            idx,arrival = box.pop()
            if idx != index or arrival < target:
                raise RuntimeError("Producer did not follow the registered trace")
            decisions = []
            def source(info):
                t,depth = box.snapshot(epoch)
                q = state.offer(depth)
                decisions.append((info.tick,t,depth,q))
                return q
            start = monotonic_ns()-epoch
            row = execute(idx,source)
            sync()
            finish = monotonic_ns()-epoch
            rows.append(row)
            timing.append(dict(index=idx,planned=target,enqueued=arrival,start=start,finish=finish,
                               offers=decisions))
    finally:
        box.cancel.set()
        producer.join()
    return rows,timing


def run(args,prepare=None):
    import torch
    from bdlm_serving.methods.t2t.online import load_backend
    from experiments.t2t.prefix_window.score_run import snapshot
    from experiments.t2t.serving_slots.switch_report import audit
    rows = load_tasks(args.samples)
    lo,hi = SLOTS[args.model]
    dig = sha(args.samples)
    args.output.mkdir(parents=True,exist_ok=False)
    control = json.loads((args.controls/"run.json").read_text())
    if control["model"] != args.model or control["batch_size"] != 1:
        raise ValueError("Fixed endpoints belong to a different model or batch")
    fixtures = {q:snapshot(args.controls,f"slottrace_{q}",dig,args.output/".fixed") for q in (lo,hi)}
    prompt_refs = ({i:json.loads((args.reference/f"{i:02d}-native32.json").read_text())["prompt_ids"]
                    for i in range(90)} if args.model == "llada21" else {})
    backend = load_backend(args,prepare)
    def exe(i,source):
        return backend(rows[i],Arm(lo,0,lo,"hard_prefix_zero"),next_budget=source,
                       **({"prompt_reference":prompt_refs[i]} if prompt_refs else {}))
    def constant(_):
        return hi
    sync = torch.cuda.synchronize
    exe(0,constant)
    sync()
    ticks = []
    for i in (0,10,20,30,50,70):
        begin = monotonic_ns()
        exe(i,constant)
        sync()
        ticks.append(monotonic_ns()-begin)
    unit = int(median(ticks))
    trace = arrival_trace(rows,unit)
    durable(args.output/"run.json",dict(model=args.model,suite="frozen90",samples_sha256=dig,
            block=256 if args.model=="gemma" else 32,policies=list(MODES),slots=[lo,hi],
            unit_ns=unit,calibration_ns=ticks,trace=trace,clock="monotonic_ns",batch_size=1,
            device=torch.cuda.get_device_name(),torch=str(torch.__version__)))
    for mode in MODES:
        # Same warmup before every arm; its results do not enter the 90 scored requests.
        exe(0,constant)
        sync()
        out,times = replay(trace,mode,(lo,hi),exe,sync)
        for row,t in zip(out,times,strict=True):
            i = t["index"]
            row.update({k:rows[i][k] for k in ("index","task_id","dataset","seed")},model=args.model,policy=mode)
            verify_work(row)
            observed = [e for e in row["events"] if "slot_offer" in e]
            offers = t["offers"]
            if [e["slot_offer"] for e in observed] != [q for _,_,_,q in offers]:
                raise ValueError("A live offer was lost or substituted")
            audit(row,[q for _,_,_,q in offers] or [lo])
            if mode.startswith("fixed"):
                q = hi if mode=="fixed_high" else lo
                old = json.loads((fixtures[q]/f"{i:02d}-slottrace_{q}.json").read_text())
                if json.loads(json.dumps(dict(row,policy=old["policy"]))) != old:
                    raise ValueError(f"Fixed endpoint is not reproducible: {mode}/{i}")
            durable(args.output/f"{i:02d}-{mode}.json",row)
        durable(args.output/f"clock-{mode}.json",dict(mode=mode,requests=times))
        durable(args.output/f"complete-{mode}.json",dict(policy=mode,requests=len(times)))
        print(f"complete {mode} {len(times)} FIFO requests",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=SLOTS,required=True)
    for field in ("samples","output","checkpoint","reference","controls"):
        parser.add_argument("--"+field,type=Path,required=True)
    run(parser.parse_args())
