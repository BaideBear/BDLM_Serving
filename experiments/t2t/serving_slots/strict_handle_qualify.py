"""CUDA qualification of a public single-request API, not a policy ranking."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from bdlm_serving.methods.t2t.serving_profiles import open_strict_gemma
from . import strict_single512_qualification as single
from .llada_frontier_accept import PROFILE as FRONTIER32

TIERS = (32,64,128,256,512)
PROMOTION = "dense_stop_promotion_fused"
JOINT = "dense_stop_joint_verify"
JOINT_PROFILES = (JOINT, single.PROFILE)
PROMOTION_PROFILES = (PROMOTION,*JOINT_PROFILES)
STOP_PROFILES = ("dense_b_stop_fused", *PROMOTION_PROFILES)


def offer_cycle(tiers=TIERS):
    # Euler cycle of directed offer pairs, including the five self transitions.
    arcs = {q:list(reversed(tiers)) for q in tiers}
    path, finished = [tiers[0]], []
    while path:
        q = path[-1]
        if arcs[q]:
            path.append(arcs[q].pop())
        else:
            finished.append(path.pop())
    return tuple(reversed(finished))[:-1]


def reference_path(root, q, index, profile="fixed90_candidate"):
    if profile in ("fixed90_fused","fixed90_dense_fused","phase_high_fused",*STOP_PROFILES):
        if q==512 and profile==single.PROFILE:
            return root/"single512-stop-1128"/f"C512-single1-{index:02}.json"
        if q==512 and profile==JOINT:
            return root/"joint-verify-1128"/f"C512-joint1-{index:02}.json"
        if q==512 and profile==PROMOTION:
            return root/"promotion-1128"/f"C512-promotion1-{index:02}.json"
        if q in (128,512):
            return root/"profile-close-1128"/f"C{q}-profile-fuse1-{index:02}.json"
        if q==256 and profile=="fixed90_dense_fused":
            return root/"capacity-head-1128"/f"C256-capacity1-{index:02}.json"
        if q==256 and profile=="phase_high_fused":
            return root/"phase-density-1128"/f"C256-density-high-{index:02}.json"
        if q==256 and profile in STOP_PROFILES:
            return root/"dense-stop-1128"/f"C256-dense-stop1-{index:02}.json"
        return root/"paid-close-1128"/f"C{q}-fuse1-{index:02}.json"
    if q == 128:
        return root/"cross-race-1128"/f"C128-gateoff-slack0-4-pref1of4-{index:02}.json"
    if q == 512:
        return root/"cross-race-1128"/f"C512-gatehalf-slack0-4-pref1of8-{index:02}.json"
    return root/"prefix-rank-1128"/f"C{q}-rankright-reuse0-{index:02}.json"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def drive(model, data, task, schedule, *, deadline, profile="fixed90_candidate", drop_after_stop=False,
          drop_after_promotion=False, drop_at_joint=False, dense_shadow=False, boundary=None, eos_audit=None):
    from bdlm_serving.runtime.diffusiongemma import observe
    from .gemma_strict_eval import encoder_tape
    from .gemma_unified_audit import projectors
    from .mask_pair_gemma import input_digest
    origin = data["input_ids"].shape[1]
    debt, ages, ledger = set(), {}, [("encoder", origin)]
    transitions, routes = Counter(), Counter()
    old, head, verify, encoded, d_q = None, 0, 0, 0, 0
    parked_pay = physical_downshift = 0
    last_full, stopped_head = None, None
    prev_promoted, prev_rebuild_sample = None, False
    held_head = None
    triggers = Counter()
    cfg = dict(seed=task["seed"], initial_offer=schedule[0], profile=profile)
    if profile=="dense_stop_joint_verified_eos":
        check(eos_audit is not None, "Need an independent terminal measurement for this profile")
        profile=single.PROFILE  # Identical routing, stops, writes and rebuilds until terminal EOS.
    with ExitStack() as stack:
        events = stack.enter_context(observe(model))
        proj, positions = stack.enter_context(projectors(model))
        enc = stack.enter_context(encoder_tape(model))
        h = open_strict_gemma(model, data, **cfg)
        stack.callback(h.close)
        if eos_audit is not None:
            eos_audit.install(stack,model,h._request.eos,origin)
        if dense_shadow:
            from .gemma_draft_shadow import DraftShadow
            check(profile=='dense_b_stop_fused' and schedule==(256,) and not drop_after_stop, 'wrong shadow policy')
            shadow=DraftShadow(h._request.stop_observer,h._request.eos)
            stack.enter_context(shadow.install(model))
        check(h.opened_prefill_q == origin and events == ledger, "open prefill")
        tick = 0
        while h.status == "ready":
            check(time.time() < deadline and tick < 12000, "deadline or no bounded progress")
            q = schedule[tick % len(schedule)]
            if drop_after_stop:
                q = 32 if stopped_head==head else 256
            if drop_after_promotion:
                q = 32 if prev_rebuild_sample else 512
            if boundary is not None:
                q = boundary.offer(tick,head)
            joint_opportunity = False
            if profile in JOINT_PROFILES:
                raw=h._request
                had_tail = raw.tail is not None
                joint_opportunity=(raw.tail is not None and not raw.rebuild and not raw.cursor
                      and len(raw.rows.initialized)==256 and raw.sampler.phase=="sample" and bool(debt)
                      and (raw.rows._native_stop or raw.rows.age>=raw.rows.max_steps or len(raw.rows.accepted)==256))
                if drop_at_joint:
                    if joint_opportunity:
                        held_head=head
                    q=32 if held_head==head else 512
            e = h.step(q)
            if dense_shadow:
                shadow.after(e,origin)
            rows = positions[-1]
            if eos_audit is not None:
                eos_audit.observe(e,rows)
            sampled, changed = set(e["sampled_absolute"]), set(e["written_absolute"])
            check(e["tick"] == tick and e["ordinal"] == head and e["positions"] == rows
                  and len(rows) == e["q"] == e["decoder_Q"] == e["history_Q"] <= q
                  and len(set(rows)) == len(rows)
                  and all(t[-1] == len(rows) for t in proj.values()), "unmeasured physical projection")
            check(e["dirty_before_absolute"] == sorted(debt) and set(sorted(debt)[:q]) <= set(rows)
                  and changed <= sampled <= set(rows), "unpaid old or unmeasured new writes")
            if profile in JOINT_PROFILES:
                joint=e["route"]=="verify_joint_paid"
                check(joint==(q==512 and joint_opportunity),"joint verification not gated by current offer/state")
                if joint:
                    full=set(range(origin+256*head,origin+256*(head+1)))
                    tail=debt-full
                    check(set(rows)==full|tail and not verify and not changed and not sampled
                          and not e['dirty_after_absolute'] and e['head_initialized_before']==256
                          and all(p['phase']=='drain' and p['rows']==p['dirty_before']
                                  and not p['sampled'] and p['age_before']==p['age_after']
                                  for p in e['parts'][1:]),"wrong terminal projection, write or tail sampling")
                    triggers['joint_calls']+=1
                    triggers['tail_dirty_Q_in_joint']+=len(tail)
                if joint_opportunity and q<512:
                    check(e['route']=='drain' and not sampled and set(rows)==set(sorted(debt)[:q]),
                          "small offer must drain in-cap on parked joint close")
                    triggers['joint_deferred']+=1
                    if old==512:
                        triggers['top_to_small_at_joint']+=1
                if drop_at_joint and held_head==head and e['route']=='verify':
                    check(not debt and q==32 and len(rows)<=q,"unpaid verification after a downshift")
                    triggers['small_verify_chunks']+=1
            if profile in PROMOTION_PROFILES:
                if e["route"]=="rebuild_sample":
                    check(q==512 and prev_promoted==head and head>0 and not debt
                          and e["opened"] is None and e["head_initialized_before"]==256
                          and rows==list(range(origin+256*head,origin+256*(head+1)))
                          and e["sampled_absolute"]==rows and len(e["parts"])==1
                          and e["parts"][0]["phase"]=="sample","sampled incomplete or stale rebuild")
                    triggers["sampled_promotions"]+=1
                if prev_rebuild_sample and q<512:
                    triggers["post_promotion_downshifts"]+=1
                    triggers["post_promotion_downshifts_with_debt"]+=int(bool(debt))
                prev_promoted = e["promoted"] if e["encoded"] else None
                prev_rebuild_sample = e["route"]=="rebuild_sample"
            if profile in STOP_PROFILES:
                mark=e['fullstop']
                if profile in JOINT_PROFILES:
                    check(mark['resident']==1+int(had_tail or e["opened"] is not None), "wrong physical residency")
                full=((q==256 or (q==512 and profile==single.PROFILE)) and mark['resident']==1 and e['route']=='dense_single'
                      and sampled==set(range(origin+head*256,origin+(head+1)*256)))
                pair=full and last_full==(tick-1,head,q)
                check(mark['in_scope']==full and mark['epoch']==origin+head*256 and
                      (not mark['hit'] or pair) and mark['latched']==mark['hit'], 'stale or partial stop')
                if stopped_head==head:
                    check(not sampled and e['route'] in ('drain','verify_paid','verify'), 'sampling after stop')
                    triggers['post_stop_sub_B_ticks']+=int(q<256)
                if mark['latched']:
                    stopped_head=head
                    triggers['latches']+=1
                triggers['eligible_pairs']+=int(pair)
                if q==512 and profile==single.PROFILE:
                    triggers['top_full_ticks']+=int(full)
                    triggers['top_eligible_pairs']+=int(pair)
                    triggers['top_latches']+=int(mark['latched'])
                    if mark['latched']:
                        hp=e['parts'][0]
                        masks=e['head_mask_before']-len(hp['admitted'])+len(hp['revoked'])
                        triggers['top_actionable']+=int(masks>0 and hp['age_after']<48)
                last_full=(tick,head,q) if full else None
            if boundary is not None:
                boundary.observe(e)
            if old is not None:
                transitions[old,q] += 1
                if old > q:
                    physical_downshift += int(bool(debt))
                    parked_pay += sum(p >= origin+256*(head+1) for p in debt.intersection(rows))
            for p in e["parts"]:
                ordinal = p["ordinal"]
                check(p["age_before"] == ages.get(ordinal, 0) and
                      p["age_after"] == p["age_before"] + bool(p["sampled"]), "sampler clock reset")
                ages[ordinal] = p["age_after"]
                if p["phase"] == "verify":
                    if e["route"]=="verify_paid":
                        check(profile!="fixed90_candidate" and verify==0 and len(e["parts"])==1
                              and debt and debt<=set(rows) and e["head_initialized_before"]==256
                              and e["promoted"] is None, "unpaid or reordered fused verification")
                    elif e["route"]=="verify_joint_paid":
                        check(profile in JOINT_PROFILES and q==512 and verify==0 and bool(debt)
                              and debt<=set(rows),"missing write on joint verify")
                    else:
                        check(not debt,"pending write on plain verification")
                    check(ordinal == head and not sampled
                          and p["rows"] == list(range(verify, verify+len(p["rows"]))), "terminal sweep")
                    verify += len(p["rows"])
            debt = (debt - set(rows)) | changed
            check(sorted(debt) == e["dirty_after_absolute"], "lost dirty across offers")
            ledger.append(("decoder",len(rows)))
            if e["encoded"]:
                check(verify == 256 and e["encoder_Q"] == 256, "encoder before verification")
                ledger.append(("encoder",256))
                head += 1
                encoded += 1
                verify = 0
            routes[e["route"]] += 1
            tick, old, d_q = tick+1, q, d_q+len(rows)
        ids = h.result()
        short_end=eos_audit.finish(ids) if eos_audit is not None else False
        check((verify == 256 or short_end) and len(proj) == 85 and events == ledger and len(enc) == encoded+1
              and enc[0][0] == data["input_ids"][0].tolist() and not debt
              and len(positions) == tick and all(len(t) == tick for t in proj.values()), "close/accounting")
        for j,(block, pos) in enumerate(enc[1:]):
            check(list(ids[j*256:(j+1)*256]) == block
                  and pos == list(range(origin+256*j, origin+256*(j+1))), "bad committed encoder input")
    return dict(input_sha256=input_digest(data), index=task["index"], seed=task["seed"],
                task_id=task["task_id"], dataset=task["dataset"], F=tick+encoded, decoder_F=tick,
                decoder_Q=d_q, encoder_F=encoded, encoder_Q=256*encoded, routes=dict(routes),
                directed_pairs={f"{a}>{b}":v for (a,b),v in sorted(transitions.items())},
                downshifts_with_debt=physical_downshift, parked_rows_paid_on_downshift=parked_pay,
                decoder_projection_modules=len(proj), **({'stop_checks':dict(triggers)} if profile in STOP_PROFILES else {}),
                **({'top_boundary':boundary.finish()} if boundary is not None else {}),
                **({'draft_shadow':shadow.blocks} if dense_shadow else {}), output_digest=sha256(
                    json.dumps(ids,separators=(",",":")).encode()).hexdigest()), ids


def main(argv=None):
    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from .mask_pair_gemma import SAMPLES_SHA
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","samples","receipts","out"):
        ap.add_argument("--"+key, type=Path, required=True)
    ap.add_argument("--profile",choices=("fixed90_candidate","fixed90_fused","fixed90_dense_fused","phase_high_fused",*STOP_PROFILES),
                    default="fixed90_candidate")
    ap.add_argument("--deadline",type=int,required=True)
    ap.add_argument("--dense-shadow",action="store_true",help="Frozen90 observation of fixed Q256 draft vs fresh verify")
    a = ap.parse_args(argv)
    if a.profile==single.PROFILE:
        from .gemma_joint_verify_eval import available
        from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
        import transformers
        available(False)
        check(transformers.__version__=="5.12.0" and
              sha256(Path(native.__file__).read_bytes()).hexdigest()=="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc",
              "Wrong native stopper")
    check(torch.cuda.is_available() and torch.cuda.device_count()==1 and
          sha256(a.samples.read_bytes()).hexdigest()==SAMPLES_SHA, "GPU-only frozen qualification")
    tasks = load_tasks(a.samples)
    a.out.mkdir(parents=True,exist_ok=True)
    proc = AutoProcessor.from_pretrained(a.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(a.checkpoint,local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa", experts_implementation="grouped_mm",
        device_map={"":"cuda:0"}).eval()
    jobs = [(f"fixed{q}",(q,)) for q in TIERS] + [("switch25",offer_cycle()),
                                                    ("park", (512,)*5+(32,)*3+(512,))]
    if a.profile in STOP_PROFILES:
        jobs.append(('drop_after_stop',(256,32)))
    if a.profile in PROMOTION_PROFILES:
        jobs.append(('drop_after_promotion',(512,32)))
    if a.profile in JOINT_PROFILES:
        jobs.append(('drop_at_joint',(512,32)))
    indices=(0,10,20)
    if a.dense_shadow:
        check(a.profile=='dense_b_stop_fused','dense shadow profile')
        jobs=[('fixed256',(256,))]
        indices=[j+k for j in range(10) for k in (0,10,20)]+list(range(30,90))
    with torch.inference_mode():
        for i in indices:
            t = tasks[i]
            data = proc.apply_chat_template([{"role":"user","content":t["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            ref_top = None
            case_jobs = list(jobs)
            if a.profile==single.PROFILE:
                src_top = reference_path(a.receipts,512,i,a.profile)
                ref_top = single.reference(src_top)
                check(ref_top[0]['index']==i and ref_top[0]['samples_sha256']==SAMPLES_SHA, "Wrong stop case")
                if any(h['actionable'] for h in ref_top[0]['stop']['hits']):
                    case_jobs.extend((s,(512,32 if s!="interrupt_top_256" else 256)) for s in single.BOUNDARIES)
            for name, schedule in case_jobs:
                dst = a.out/f"{name}-{i:02}.json"
                if dst.exists():
                    previous = json.loads(dst.read_bytes())
                    check(previous.get("profile","fixed90_candidate")==a.profile and
                          ('draft_shadow' in previous)==a.dense_shadow, "wrong profile in destination")
                    continue
                boundary = single.Boundary(name,*ref_top) if ref_top and name in ("fixed512",*single.BOUNDARIES) else None
                if ref_top:
                    available(True)
                row, ids = drive(model,data,t,schedule,deadline=a.deadline,profile=a.profile,
                                  drop_after_stop=name=='drop_after_stop',drop_after_promotion=name=='drop_after_promotion',
                                  drop_at_joint=name=='drop_at_joint',dense_shadow=a.dense_shadow,boundary=boundary)
                if len(schedule)==1:
                    src = reference_path(a.receipts,schedule[0],i,a.profile)
                    blob = src.read_bytes()
                    ref = json.loads(blob)
                    joint_ref = a.profile==JOINT and schedule==(512,)
                    top_ref = ref_top is not None and schedule==(512,)
                    if joint_ref:
                        check(ref["schema"]=="joint-verify-v1" and ref["on"] is True and
                              ref["samples_sha256"]==SAMPLES_SHA,"wrong joint reference")
                    ids_match = row["output_digest"]==ref["ids"] if joint_ref or top_ref else list(ids)==ref["output_ids"]
                    same = ids_match and all(row[k]==ref[k] for k in
                            ("index","seed","task_id","dataset","input_sha256","F","decoder_F","encoder_F","decoder_Q","encoder_Q"))
                    row.update(reference_sha256=sha256(blob).hexdigest(), identical=same)
                row.update(schema="strict-handle-gemma-qualification-v1", suite=name, profile=a.profile, samples_sha256=SAMPLES_SHA,
                           gpu=torch.cuda.get_device_name(), run_pid=os.getpid(), timestamp=time.time())
                tmp = dst.with_suffix(".tmp")
                tmp.write_text(json.dumps(row,sort_keys=True)+"\n")
                tmp.replace(dst)
                print(name,i,row["decoder_F"],row.get("identical"),flush=True)
                check(row.get("identical",True), "fixed decoder differs from frozen control")


def llada_main(argv=None):
    """Separate pinned environment. Admission/edits and changed inputs stay on GPU."""
    import torch
    from .cross_race_llada import load_gpu
    from .llada_fixed_family_eval import evaluate_fixed, fixed_cross, TIERS as slots
    from .mask_pair_gemma import SAMPLES_SHA
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.serving_profiles import open_strict_llada
    from bdlm_serving.methods.t2t.budget_family import Law
    ap = argparse.ArgumentParser(description=llada_main.__doc__)
    for k in ("checkpoint","samples","reference","out"):
        ap.add_argument("--"+k, type=Path, required=True)
    ap.add_argument("--deadline",type=int,required=True)
    ap.add_argument("--profile", choices=("legacy", "quarter_fullwrite", "quarter_edit_warm", "quarter_edit_window","quarter_window_repeat64","quarter_window_repeat64_paid16",FRONTIER32), default="legacy")
    a = ap.parse_args(argv)
    candidate = a.profile != "legacy"
    frontier32 = a.profile == FRONTIER32
    paid16 = frontier32 or a.profile == "quarter_window_repeat64_paid16"
    repeat = a.profile == "quarter_window_repeat64" or paid16
    window = repeat or a.profile == "quarter_edit_window"
    editable = window or a.profile == "quarter_edit_warm"
    window_fixed = None
    paid_fixed = None
    if window:
        from .llada_frontier_seat import classes
        window_fixed = classes("window")[1]
        if paid16:
            paid_fixed = classes("paid")[1]
    if repeat:
        from .llada_agreement import request_class
        from .llada_hfull_single_factor import request_classes
        from .llada_switch_votes import audit as audit_votes
        repeat_fixed = request_class(request_classes()[1])
        if frontier32:
            from .llada_frontier_accept import frontier_class
            frontier_fixed = frontier_class(request_classes()[1])
    check(torch.cuda.is_available() and torch.cuda.device_count()==1 and
          sha256(a.samples.read_bytes()).hexdigest()==SAMPLES_SHA, "GPU-only frozen qualification")
    tok, model = load_gpu(a, lambda:None)
    a.out.mkdir(parents=True,exist_ok=True)
    tasks = load_tasks(a.samples)
    with torch.inference_mode():
        for i in (0,10,20):
            t = tasks[i]
            ref = json.loads((a.reference/f"{i:02d}-native32.json").read_text())
            check(ref["index"]==i and ref["generation_cap"]==2048, "unregistered LLaDA reference")
            x = torch.tensor([ref["prompt_ids"]],device=model.device)
            jobs = [(f"fixed{q}", (q,)) for q in slots] + [
                ("switch25", offer_cycle(slots)), ("park",(64,)*5+(4,)*3+(64,))]
            if paid16:
                jobs.append(("park16",(64,)*5+(16,)*3+(64,)))
            for name, schedule in jobs:
                dst = a.out/f"{name}-{i:02}.json"
                if dst.exists():
                    prior = json.loads(dst.read_bytes())
                    check(prior.get("profile","legacy")==a.profile and prior.get("identical",True),
                          "wrong existing LLaDA receipt; use an isolated output directory")
                    continue
                kwargs = dict(seed=t["seed"], mask_id=tok.convert_tokens_to_ids("<|mask|>"),
                              eos_ids=[tok.eos_token_id])
                kwargs.update(dict(profile=a.profile) if candidate else dict(coverage="head",top_gate="half"))
                debt, widths, pos, directed = set(), [], [], Counter()
                checks = Counter()
                parked, switches, clock, d_q, tick, old = 0, 0, 0, 0, 0, None
                park_target, park_tested = set(), False
                def hook(_m, _args, kw):
                    p=kw["position_ids"][0]
                    mask=kw["attention_mask"][0,0]
                    expected=p[:,None]//32 >= torch.arange(mask.shape[1],device=p.device)[None,:]//32
                    check(torch.equal(mask,torch.zeros_like(mask).masked_fill(
                        ~expected,torch.finfo(mask.dtype).min)), "wrong physical block mask")
                    pos.append(p.tolist())
                with ExitStack() as hooks:
                    hooks.enter_context(model.model.layers[0].attention.register_forward_pre_hook(hook,with_kwargs=True))
                    hooks.enter_context(model.lm_head.register_forward_pre_hook(
                        lambda _,args:widths.append(args[0].shape[1])))
                    h = open_strict_llada(model,x,**kwargs)
                    hooks.callback(h.close)
                    if repeat:
                        h._request.audit_admission=True
                        previous_votes={}
                    check(pos==[list(range(x.shape[1]))], "wrong prompt prefill")
                    while h.status=="ready":
                        check(time.time()<a.deadline, "deadline")
                        q=schedule[tick % len(schedule)]
                        if candidate:
                            req = h._request
                            lo, split = max(req.origin,req.head), min(req.end,req.head+32)
                            head_rows = set(range(lo,split))
                            if window and name in ("park","park16"):
                                # Drop on a measured write to a live successor, not
                                # at a fixed step that may have nothing to park.
                                if not park_tested and old==64:
                                    park_target = {p for p in debt if p>=split}
                                    park_tested = bool(park_target)
                                q = (4 if name=="park" else 16) if park_target & debt else 64
                            close_due = (q==32 and not req.stopping and debt and debt<=head_rows
                                 and head_rows<=req.bank.initialized and req.close_attempted!=req.head
                                 and not (req.x[0,lo:split]==kwargs["mask_id"]).any().item())
                        if repeat:
                            vote_head=req.head
                            vote_warm=req.bank.last_end>=min(req.head+32,req.end)
                            left=None
                            if frontier32 and q==32 and vote_warm and not req.stopping:
                                left=next((p for p,token in zip(range(lo,split),req.x[0,lo:split].tolist())
                                           if token==kwargs["mask_id"]),None)
                        e=h.step(q)
                        if repeat:
                            prior=previous_votes
                            extra32=set()
                            previous_votes,extras=audit_votes(q,vote_head,vote_warm,e,kwargs["mask_id"],prior,
                                                              frontier_left=left,frontier_added=extra32)
                            if extra32:
                                checks["frontier32_rows"]+=len(extra32)
                            if extras:
                                checks["repeat_Q"]+=len(extras)
                            if prior and old==64 and q<64:
                                checks["discard_on_downshift"]+=len(prior)
                        rows=set(e["rows"])
                        check(e["rows"]==pos[-1] and e["q"]==len(rows)==widths[-1]<=q
                              and e["layer_physical_q"]==[len(rows)]*20 and e["encoder_F"]==0
                              and e["dirty_before"]==sorted(debt) and set(sorted(debt)[:q])<=rows,
                              "wrong projection or debt")
                        if candidate:
                            if window and q<32:
                                check(not e["borrowed_rows"] and rows-head_rows <= debt and
                                      set(e["sampled"])<=head_rows, "window projected clean or sampled parked rows")
                                if name in ("park","park16"):
                                    checks["parked_dirty_Q"]+=len((rows-head_rows)&park_target)
                                    park_target-=rows
                            if "window_geometry" in e:
                                g=e["window_geometry"]
                                pinned=set(e["dirty"])|set(e["prefix"])
                                expected_sampled=(set(e["cold"] or rows)&head_rows)|set(e.get("paid_waiting_mask",()))
                                check(window and q<32 and len(set(g["before"]))==len(rows)
                                      and rows!=set(g["before"]) and pinned<=rows&set(g["before"])
                                      and set(g["interior"])<=rows&head_rows
                                      and set(e["sampled"])==expected_sampled,
                                      "wrong width, pins or sampler on switched window")
                                checks["window_F"]+=1
                            paid=set(e.get("paid_waiting_mask",()))
                            if paid:
                                check(paid16 and q==16 and e["cold"] and debt<=rows
                                      and paid<=head_rows&set(e["mask"]) and paid.isdisjoint(debt)
                                      and set(e["sampled"])==set(e["cold"])|paid,
                                      "paid warm MASK did not come from this tick's paid head MASKs")
                                checks["paid_mask_rows"]+=len(paid)
                            closed = e["phase"] in ("close_full_write","close_edit_once")
                            check(closed==bool(close_due), "missed or premature full-head close")
                            if closed:
                                check(rows==head_rows and (e["phase"]=="close_edit_once")==editable and
                                      (set(e["sampled"])==rows if editable else not e["sampled"] and not e["written_absolute"]),
                                      "wrong terminal row selection or sampling")
                                checks["edit_once_F" if editable else "full_write_F"]+=1
                                if editable:
                                    checks["terminal_writes"]+=len(e["written_absolute"])
                            warm=set(e.get("warm_head_rows",()))
                            if warm:
                                check(editable and q==64 and e["cold"] and set(e["cold"]).isdisjoint(head_rows)
                                      and debt<=rows and warm<=head_rows&set(e["mask"])
                                      and warm.isdisjoint(debt) and set(e["sampled"])==set(e["cold"])|warm,
                                      "warm sample did not come from this tick's paid head MASKs")
                                checks["warm_sample_rows"]+=len(warm)
                            if e["borrowed_rows"]:
                                check(q==16 and e["borrowed_rows"]==len(e["tail_read_only"])
                                      and not debt.intersection(e["tail_read_only"]),
                                      "borrowing displaced a pending write or sampled a successor")
                                checks["borrow_Q"]+=e["borrowed_rows"]
                            if e["tail_opened"] and not e["tail_resident"]:
                                check(q==64 and 4*e["head_mask_at_plan"]<=len(head_rows),
                                      "new successor admitted before the quarter gate")
                                checks["quarter_open_F"]+=1
                        if old is not None:
                            directed[old,q]+=1
                            if old>q:
                                switches+=bool(debt)
                                parked+=sum(p >= (e["block"]+1)*32 for p in rows.intersection(debt))
                        clock+=bool(e["sampled"])
                        debt=(debt-rows)|set(e["written_absolute"])
                        check(e["dirty_after"]==sorted(debt) and e["sample_clock"]==clock
                              and set(e["written_absolute"])<=set(e["sampled"])<=rows,
                              "dirty or sampler clock mismatch")
                        tick, d_q, old = tick+1, d_q+len(rows), q
                    ids=h.result()
                    check(not debt and len(pos)==len(widths)==tick+1, "unaccounted close")
                row = dict(schema="strict-handle-llada-qualification-v1", suite=name, index=i, seed=t["seed"],
                           dataset=t["dataset"], task_id=t["task_id"], samples_sha256=SAMPLES_SHA,
                           F=tick, decoder_F=tick, decoder_Q=d_q, encoder_F=0, encoder_Q=0,
                           directed_pairs={f"{a}>{b}":v for (a,b),v in sorted(directed.items())},
                           downshifts_with_debt=switches, parked_rows_paid_on_downshift=parked,
                           decoder_projection_modules=20, gpu=torch.cuda.get_device_name(),
                           timestamp=time.time(), run_pid=os.getpid(),
                           output_digest=sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest())
                if len(schedule)==1:
                    q=schedule[0]
                    if candidate:
                        fixed=evaluate_fixed(model,tok,t,ref["prompt_ids"],q,Law(),
                              cross=("quarter" if q==64 else "head", q==16 and not window),coverage="head" if q==64 else "sparse",
                              close_policy=("edit_once" if editable else "full_write") if q==32 else "unchanged",
                              warm_head_sample=editable and q==64,
                              request_cls=(paid_fixed if paid16 and q==16 else window_fixed if q<32 else
                                           repeat_fixed if repeat and q==64 else
                                           frontier_fixed if frontier32 and q==32 else None))
                    else:
                        fixed=fixed_cross(model,tok,t,ref["prompt_ids"],q,Law(),mode="half" if q==64 else "head",
                                          borrow=False,coverage="head" if q==64 else "sparse")
                    row["identical"]=(list(ids)==fixed["output_ids"] and fixed["F"]==tick
                                      and fixed["Q"]==d_q and pos[1:]==[e["rows"] for e in fixed["trace"]])
                if candidate:
                    row.update(profile=a.profile, policy_checks=dict(checks))
                tmp=dst.with_suffix(".tmp")
                tmp.write_text(json.dumps(row,sort_keys=True)+"\n")
                tmp.replace(dst)
                print(name,i,tick,row.get("identical"),flush=True)
                check(row.get("identical",True),"fixed LLaDA positions/output differ from control")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model", choices=("gemma", "llada"), default="gemma")
    options, rest = parser.parse_known_args()
    (main if options.model == "gemma" else llada_main)(rest)
