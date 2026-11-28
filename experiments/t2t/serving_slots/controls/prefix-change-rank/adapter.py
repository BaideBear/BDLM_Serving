"""Request-local inserts into the sealed IBS executor. Default entry point is untouched."""
from inspect import getsource
from pathlib import Path
from textwrap import dedent

from rank import Versions
from replay import filehash, need


def install(fixed,arm,tape):
    need(arm in ("none","right","change") and filehash(Path(fixed.__file__))==
         "b95003b5f8867bd6ef7d5126c9814e08fb6d03bab57f67642618028d8a1960d4","unknown IBS")
    src=dedent(getsource(fixed.FixedOptimus.run))
    inserts={
        "    p = prompt.shape[1]\n": """    _need(self.kind=='llada21' and self.mode=='ibs' and self.caps==(8,) and self.chunk==8
          and self.prefix_refresh==(0 if _arm=='none' else 2) and self.prefix_order=='right','wrong endpoint')
    certified_end = None
    versions = _Versions()
""",
        "            projected = positions\n": """            if not self.prefix_refresh:
                _, eligible_abs, slack = plan_prefix_refresh(prompt_len=p,block_start=start,
                    cached_prefix_before=prefix_before,old_mask=old_mask,dirty=dirty,
                    initialized=cache.initialized,original_positions=positions.tolist(),
                    offer=offer,quota=2)
            lo,hi = max(start,p),min(start+width,p+gen_length)
            pool = [int(i) for i in eligible_abs if lo<=i<hi]
            plan = versions.plan(pool,min(2,slack,len(pool)),lo,hi)
            if self.prefix_refresh:
                _need(plan['right']==extra_abs.tolist(),'native right picks mismatch')
                extra_abs = np.asarray(plan[_arm],dtype=np.int64)
            record = dict(tick=tick,plan=plan)
""",
        "            if ready >= 32:\n": """            # Log actual projected input versions before applying this tick's token writes.
            actual = projected.tolist()
            record.update(paid=actual,mask=masked.tolist())
            versions.project(tick-1,actual)
            before_ids = before[0,selected].tolist()
            after_ids = x[0,positions].tolist()
            changes = [[a,u,v] for a,u,v in zip(positions.tolist(),before_ids,after_ids,strict=True) if u!=v]
            record['changes'] = changes
            if _arm=='change':
                events[-1]['prefix_refresh_order'] = 'change'
            versions.write(tick-1,[w[0] for w in changes])
            _tape.append(record)
            lo,hi = max(start,p),min(start+ready,p+gen_length)
            frozen = x[0,lo:hi].tolist() if hi>lo else []
            eos_pos = next((lo+j for j,v in enumerate(frozen) if v in eos_ids),None)
            if eos_pos is not None:
                due = sorted(dirty)
                _need(len(due)<=offer and all(start+ready<=a<start+width and a in cache.initialized for a in due),
                      'uncertified debt')
                if due:
                    pos = torch.as_tensor(due,device=x.device,dtype=torch.long)
                    forward(pos,'certificate_debt')
                    events[-1].update(block=start//32,step=step,physical_cap=offer,
                        visible_kv=cache.visible_end,dirty_before=due,written_absolute=[],
                        sampled_absolute=[],edited_positions=[])
                    dirty.difference_update(due)
                _need(not dirty,'unpaid close')
                certified_end = eos_pos+1
                break
""",
        "        if any(bool((x[:, p:start+32] == eos).any()) for eos in eos_ids):\n": """        if certified_end is not None:
            x = x[:,:certified_end]
            break
""",
    }
    for anchor,body in inserts.items():
        need(src.count(anchor)==1,"unregistered anchor")
        src=src.replace(anchor,body+anchor)
    ns=dict(vars(fixed),_arm=arm,_need=need,_Versions=Versions,_tape=tape)
    exec(compile(src,"<ibs8-paid-prefix-rank>","exec"),ns)
    return ns["run"]
