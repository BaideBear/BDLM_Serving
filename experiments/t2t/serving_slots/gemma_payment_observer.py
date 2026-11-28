"""Observe a paid row after its first write. Read-only: never change IDs or sample."""
from . import gemma_paid_agreement_eval as ev
from .gemma_ibs_certified_eos import require
from .gemma_paid_agreement import Agreement


def probe(arms,rt,torch,model,data,proc,controls,unmask,task,scoring,special):
    origin = data["input_ids"].shape[1]
    pending, events = {}, []
    physical_tick = -1

    class Track(Agreement):
        def observe(self,*args,**kwargs):
            updated = super().observe(*args,**kwargs)
            step = self.observations[-1]
            require(step["tick"] == physical_tick, "Sampler/decoder clock mismatch")
            extra = set(step["extra"])
            for pos, token, native in zip(step["positions"], step["proposals"],
                                         step["native"],strict=True):
                if native or pos in extra:
                    require(pos not in pending, "Second write ahead of first payment")
                    pending[pos] = token, "extra" if pos in extra else "native", physical_tick
            return updated

    def forward(_module,_args,kw,output):
        nonlocal physical_tick
        physical_tick += 1
        pos = kw["decoder_position_ids"][0].tolist()
        ids = kw["decoder_input_ids"][0].tolist()
        paying = [(j,p-origin) for j,p in enumerate(pos) if p-origin in pending]
        for begin in range(0,len(paying),8):
            chunk = paying[begin:begin+8]
            projected = output.logits[0,[j for j,_ in chunk]].float()
            prob = projected.softmax(-1)
            best_p,best = prob.max(-1)
            written_p = prob.gather(-1,torch.tensor([[ids[j]] for j,_ in chunk],
                                                     device=prob.device)).flatten().tolist()
            for (j,p),current,conf,oldconf in zip(chunk,best.tolist(),best_p.tolist(),written_p,strict=True):
                old,role,tick = pending.pop(p)
                require(tick < physical_tick and old == ids[j], "Unpaid ID supplied to decoder")
                events.append(dict(role=role,pos=p,write=tick,payment=physical_tick,old=old,
                                   top=current,top_p=conf,written_p=oldconf,ordinary=current not in special))

    with model.register_forward_hook(forward,with_kwargs=True):
        checked = ev.run_arm(rt,torch,model,data,proc,controls,unmask,task,"apply",
                             scoring,special,agreement_type=Track)
    require(checked == arms["apply"] and not pending and
            len(events)==checked["bill"]["paid_rows"] and physical_tick+1==checked["bill"]["D"],
            "Observer alters application or does not witness all payments")
    require(len([r for r in events if r["role"]=="extra"]) ==
            sum(len(o["extra"]) for o in checked["agreement"]["observations"]), "Extra payment mismatch")
    return dict(identical=True,events=events)


if __name__ == "__main__":
    ev.main(probe,control_name="payment_observer")
