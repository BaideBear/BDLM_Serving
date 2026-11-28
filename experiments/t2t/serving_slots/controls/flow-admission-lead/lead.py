"""One-step lookahead from measured single-head frontiers only."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LeadState:
    head: int | None = None
    prev_f: float | None = None
    f: float | None = None

    def observe(self, head, opened, finished, frontier):
        if opened != 1 or finished:
            self.head = self.prev_f = self.f = None
            return
        self.prev_f = self.f if self.head == head else None
        self.head, self.f = head, frontier

    def opportunity(self, head, right, blocks, finished, f):
        return (right == head+1 and right < blocks and not finished and self.head == head
                and self.prev_f is not None and self.f == f and f < .8
                and f+max(0., f-self.prev_f) >= .8)


def factory(base_class, control_module, enabled):
    """Build on the sealed compact executor; keep native gate first in the OR."""
    import ast
    from inspect import unwrap
    from types import MethodType

    class Lead(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state=LeadState()
            joint=self.owner._joint_decode_slot_packed

            def sampled(*a):
                result=joint(*a)
                t=self.ticks[-1]
                slot=t["slots"][-1]
                state.observe(t["commit"],t["open"],slot["fin"],slot["frontier"])
                return result

            def admit(c,r,count,finished,f):
                if not state.opportunity(c,r,count,finished,f):
                    return False
                self.ticks[-1]["lead_vote"]=dict(tick=len(self.ticks)-1,head=c,
                        prev_f=state.prev_f,f=f,opened=enabled)
                return enabled

            self.owner._joint_decode_slot_packed=sampled
            self.owner._lead_allows=admit
            path=control_module.control_directory("FlowBlock")/"flowblock_packed.py"
            orig=unwrap(self.generate.__func__)
            fn,=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)
                 and n.name==orig.__name__]
            spawn,=[n for n in fn.body if isinstance(n,ast.FunctionDef) and n.name=="spawn_all"]
            gate,=[n for n in ast.walk(spawn) if isinstance(n,ast.If)
                   and ast.unparse(n.test)=="float(blk_frontier[rightmost, b].item()) < spawn_threshold"]
            gate.test=ast.BoolOp(ast.And(),[gate.test,ast.parse(
                "not self._lead_allows(cp[b], rp[b], num_blocks, bool(blk_finished[rightmost,b].item()), "
                "float(blk_frontier[rightmost,b].item()))",mode="eval").body])
            loop,=[n for n in fn.body if isinstance(n,ast.While)]
            j,=[i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign)
                and ast.unparse(n.targets[0])=="output" and isinstance(n.value,ast.Call)
                and ast.unparse(n.value.func)=="self.model"]
            # Same observation point as the sealed ActiveW2 instrumentation.
            loop.body.insert(j,ast.parse("self._observe_geometry(o_start,commit_ptr,right_ptr,"
                                         "tail_start,kvpos,attn_mask,seq_idx,total_length)").body[0])
            ns=dict(orig.__globals__)
            exec(compile(ast.fix_missing_locations(ast.Module([fn],[])),str(path),"exec"),ns)
            self.generate=MethodType(ns[fn.name],self.owner)
    return Lead
