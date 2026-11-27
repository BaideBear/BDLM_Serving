"""Frozen Gemma five-slot request evaluation. GPU0 is acquired by the protected launcher."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.evaluation.original30 import save_json
from experiments.t2t.serving_slots import gemma_fixed_caps as caps

HERE = Path(__file__).resolve()
SLOTS = (32, 64, 128, 256, 512)
# An external workload replay, not a budget controller inside the request.
OFFERS = (512, 512, 128, 32, 64, 256, 512, 32)
ARMS = {f"q{b}": ((b,), True, {}) for b in SLOTS}
ARMS.update(q128_slack_random=((128,), True, {128: 32}),
            replay=(OFFERS, True, {}), replay_open=(OFFERS, False, {}),
            q512_open=((512,), False, {}))
GROUPS = (("replay", "replay_open"), ("q512", "q512_open"), ("q32",),
          ("q64",), ("q128", "q128_slack_random"), ("q256",))
BASE_GROUPS = len(GROUPS)
RANKS = {}
for budget, quota in ((64,16), (32,8), (128,32), (128,64), (128,96)):
    group = tuple(f"q{budget}_k{quota}_{mode}" for mode in ("random","oldest"))
    GROUPS += (group,)
    for policy, mode in zip(group, ("random","oldest")):
        ARMS[policy] = ((budget,), True, {budget:quota})
        RANKS[policy] = mode
KEYS = ("index", "task_id", "dataset", "seed")
EXCHANGE = ("q128_k32_oldest", "q128_k32_oldest_swap4")
ARMS[EXCHANGE[1]] = ARMS[EXCHANGE[0]]
RANK_END = len(GROUPS)
GROUPS += (EXCHANGE,)
PILOT_EX = "gemma-q128-k32-exchange-cpu3-v1"
EX_SPEC = {"offer":128, "quota":32, "cap":4, "tick":"even", "mode":"oldest"}
SPAWN = ("q512","q512_spawn50")
ARMS[SPAWN[1]] = ARMS[SPAWN[0]]
GROUPS += (SPAWN,)
PILOT_SPAWN = "gemma-q512-spawn50-cpu3-v1"
TRANSFER = ("replay","replay_q128_prefix32")
ARMS[TRANSFER[1]] = (OFFERS,True,{128:32})
GROUPS += (TRANSFER,)
PAIRED_OFFERS = (512,512,128,128,64,256,512,32,512,512,32,32,64,256,512,32)
DWELL = ("dwell","dwell_q128_prefix32")
for arm,quota in zip(DWELL, ({}, {128:32}), strict=True):
    ARMS[arm] = (PAIRED_OFFERS, True, quota)
GROUPS += (DWELL,)
MASK_OFFERS = SLOTS[:-1]
DYNAMIC_MASK = ("replay",) + tuple(f"replay_unmask{q}" for q in MASK_OFFERS)
FIXED_MASK = tuple((f"q{q}",f"q{q}_unmask{q}") for q in MASK_OFFERS)
SWITCH_MASK = tuple((f"alternate{q}",f"alternate{q}_unmask{q}") for q in MASK_OFFERS)
for (base, _), q in zip(SWITCH_MASK,MASK_OFFERS,strict=True):
    ARMS[base] = ((512,q),True,{})
MASK_GROUPS = (DYNAMIC_MASK,*FIXED_MASK,*SWITCH_MASK)
UNMASK = {f"{base}_unmask{q}":q for q in MASK_OFFERS for base in ("replay",f"q{q}")}
UNMASK.update({g[1]:q for g,q in zip(SWITCH_MASK,MASK_OFFERS,strict=True)})
for group in MASK_GROUPS:
    GROUPS += (group,)
    for arm in group[1:]:
        ARMS[arm] = ARMS[group[0]]


def offer_witness(trace, target, unmask, origin):
    """Use physical decoder edges, not merely two outstanding canvas objects."""
    from hashlib import sha256
    def digest(x):
        return sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    first, hits, edges = None, 0, 0
    for i,e in enumerate(trace):
        boundary = origin + 256 * (e["head"]+1)
        count = sum(p<boundary for p in e["selected"])*sum(k>=boundary for k in e["visible"])
        if e["future_edges"]!=count or e["open_edges"]!=(count if e["offer"]==unmask else 0):
            raise ValueError("Observed mask does not match intervention or geometry")
        if e["offer"]==target and count:
            if first is None:
                first=i
            hits+=1
            edges+=count
    geom = None
    if first is not None:
        e = trace[first]
        if not e["first_offer_state"]:
            raise ValueError("Missing first offer state digest")
        geom = {k:e[k] for k in ("tick","head","offer","visible","selected","sampled",
                                  "dirty_before","ages_before","first_offer_state")}
    return dict(first_tick=first, pretrace_sha256=digest(trace[:first]), geometry_sha256=digest(geom),
                rounds=hits, edge_pairs=edges)


def compare_offer(c,t,target):
    wc,wt = [r["offer_mask_witness"][str(target)] for r in (c,t)]
    if any(c[k]!=t[k] for k in (*KEYS,"input_sha256","prefill_q")) or any(
            wc[k]!=wt[k] for k in ("first_tick","pretrace_sha256","geometry_sha256")):
        raise ValueError("Request diverged before its first treated offer")
    if wc["first_tick"] is None and any(c[k]!=t[k] for k in
            ("output_ids","trace","generation_forwards","generation_physical_padded_rows")):
        raise ValueError("Unexposed pair changed")


def common_offers(a,b):
    """On a permuted load path, keep the observed prefix of the same-profile replay."""
    if any(a[k]!=b[k] for k in (*KEYS,"input_sha256","prefill_q")):
        raise ValueError("Changed same-profile task or input")
    for x,y in zip(a["trace"],b["trace"]):
        if x["offer"]!=y["offer"]:
            return x["tick"]
        if x!=y:
            raise ValueError("Same-profile trace changed before the first different offer")
    if a["trace"]!=b["trace"] or a["output_ids"]!=b["output_ids"]:
        raise ValueError("No different offers but changed result")
    return None


def check_early_spawn(control, trial, ids_a, ids_b):
    """The first different admission must open an extra canvas from a shared head."""
    for tick,(a,b) in enumerate(zip(control,trial)):
        if a==b:
            continue
        if (not tick or len(a["ages_before"])!=1 or len(b["ages_before"])!=2 or b["ages_before"][1]!=0
                or b["ages_before"][0]!=a["ages_before"][0] or not a["ages_before"][0]
                or b["dirty_before"]!=a["dirty_before"]+[[]] or a["offer"]!=512 or b["offer"]!=512
                or a["head"]!=b["head"] or (a["q"],b["q"])!=(256,512)
                or b["selected"][:256]!=a["selected"]
                or a["sampled"]!=a["selected"] or b["sampled"]!=b["selected"]):
            raise ValueError("First contrast must be matched head plus an earlier fresh tail")
        # Native state starts from zero accepted rows, and its accepted set is
        # updated only through admissions and revocations. Reconstruct the last
        # progress fraction on this head, including its time as a tentative tail.
        accepted=set()
        num=den=None
        for e in control[:tick]:
            k=a["head"]-e["head"]
            if not 0<=k<len(e["admitted"]):
                continue
            num=len(e["admitted"][k]); den=max(1,256-len(accepted))
            accepted.update(e["admitted"][k])
            accepted.difference_update(e["revoked"][k])
        if num is None or not 0.5<=num/den<0.8:
            raise ValueError("Earlier spawn is not explained by the registered thresholds")
        return dict(tick=tick,head=a["head"],newly_accepted=num,previously_unaccepted=den)
    if len(control)!=len(trial) or ids_a!=ids_b:
        raise ValueError("Unchanged admission changed the request")
    return None


def check_exchange(control, trial, ids_a, ids_b):
    """Only compare geometry across arms until the first physical intervention."""
    first = None
    exposures = 0
    for i,e in enumerate(trial):
        dropped = set(e["displaced_mask"])
        if not dropped:
            continue
        added = set(e["prefix_selected"])
        selected = set(e["selected"])
        if (e["phase"] != "sparse" or e["offer"] != 128 or i % 2 or
                not 1 <= len(dropped) <= min(4,len(added)) or
                dropped & selected or not added <= selected or len(added)>32 or
                len(e["dirty_before"]) != 1 or e["prefix_end"] is None):
            raise ValueError("Invalid exchange intervention")
        base = min(e["visible"])
        if (not all(base+e["prefix_end"] <= r < base+256 for r in dropped)
                or not all(base <= r < base+e["prefix_end"] for r in added)
                or len(selected) - e["mandatory_q"] - len(added) < 3*len(dropped)):
            raise ValueError("Prefix or preserved unresolved count outside contract")
        first = i if first is None else first
        exposures += len(dropped)
    for i,(a,b) in enumerate(zip(control,trial)):
        if i == first:
            if (any(a[k] != b[k] for k in ("q","tick","offer","head","visible","ages_before",
                                            "dirty_before","mandatory_q","phase","prefix_end"))
                    or set(a["selected"])-set(b["selected"]) != set(b["displaced_mask"])
                    or set(b["selected"])-set(a["selected"]) != set(b["prefix_selected"])-set(a["prefix_selected"])
                    or not set(a["prefix_selected"]) <= set(b["prefix_selected"])):
                raise ValueError("First exchange was not equal-width and one-for-one")
            break
        if a != {k:v for k,v in b.items() if k!="displaced_mask"}:
            raise ValueError("Trajectory differs before exchange")
    if first is None and (len(control)!=len(trial) or ids_a!=ids_b):
        raise ValueError("Changed output without exchanging any rows")
    if first is not None and first >= len(control):
        raise ValueError("Exchange state missing from control")
    return first, exposures


def read(path):
    return json.loads(path.read_text())


def transfer_witness(control, trial, ids_a, ids_b, origin):
    """Check the complete tape through the first physically added prefix Q."""
    first=None
    n_rows=0
    for i,e in enumerate(trial):
        extra=e["prefix_selected"]
        if not extra:
            continue
        first=i if first is None else first
        start=origin+256*e["head"]
        if (e["offer"]!=128 or e["phase"]!="sparse" or e["q"]>128
                or not 1<=len(set(extra))==len(extra)<=32 or e["prefix_end"] is None
                or any(not start<=p<start+e["prefix_end"] for p in extra)
                or not set(extra)<=set(e["sampled"]) or set(extra)&set(e["refresh_only"])):
            raise ValueError("Invalid dynamic prefix selection")
        n_rows+=len(extra)
    stop=len(control) if first is None else first
    if first is not None and first>=len(control):
        raise ValueError("Missing matching control tick")
    def normalized(e):
        return {k:v for k,v in e.items() if k not in ("prefix_mode","prefix_quota")}
    if any(normalized(x)!=normalized(y) for x,y in zip(control[:stop],trial[:stop],strict=True)):
        raise ValueError("Dynamic path diverged before prefix refresh")
    if first is None:
        if len(control)!=len(trial) or ids_a!=ids_b:
            raise ValueError("Unexposed request changed")
    else:
        x,y=control[first],trial[first]
        extra=set(y["prefix_selected"])
        if (any(x[k]!=y[k] for k in ("offer","tick","head","phase","dirty_before","ages_before","visible",
                                     "mandatory_q","head_capacity","prefix_end","refresh_only"))
                or any(set(x[k])&extra or set(y[k])!=set(x[k])|extra for k in ("selected","sampled"))
                or x["q"]+len(extra)!=y["q"]):
            raise ValueError("First dynamic refresh displaced an existing row")
    return first,n_rows


def spawn_witness(a,b):
    first = a["first_earlier_spawn"]
    if first != b["first_earlier_spawn"] or a["input_sha256"] != b["input_sha256"]:
        raise ValueError("Unpaired spawn observation")
    for r in (a,b):
        t=r["trace"]
        enc=sum(len(e["encoded"]) for e in t)
        if (r["quote_verified_decoder_calls"]!=len(t) or not t or r["dual_ticks"]!=sum(e["opportunity"] for e in t)
                or r["projection_modules"]!=85 or r["decoder_layers"]!=30 or r["status"]!="success"
                or r["generation_forwards"]!=len(t)+enc or r["encoder_q"]!=256*enc
                or r["decoder_q"]!=sum(e["q"] for e in t)
                or r["generation_physical_padded_rows"]!=r["encoder_q"]+r["decoder_q"]
                or any(e["tick"]!=j or e["offer"]!=512 for j,e in enumerate(t))):
            raise ValueError("Spawn cost/quote mismatch")
    if first is None:
        if (a["output_ids"],a["trace"])!=(b["output_ids"],b["trace"]):
            raise ValueError("No earlier admission but request changed")
    else:
        i=first["tick"]
        if (not 0<i<min(len(r["trace"]) for r in (a,b))
                or a["trace"][:i]!=b["trace"][:i] or first["previously_unaccepted"]<=0
                or not .5<=first["newly_accepted"]/first["previously_unaccepted"]<.8):
            raise ValueError("Bad first spawn threshold crossing")
        x,y=a["trace"][i],b["trace"][i]
        if (x["head"],x["q"],x["opportunity"],y["head"],y["q"],y["opportunity"])!=(
                first["head"],256,False,first["head"],512,True):
            raise ValueError("First admission failed to add one actual tail")
    return first


def spawn_pilot(folder):
    m=read(folder/"run.json")
    if (sha(folder/"run.json")!="07cd17f32aa97cf979bf28a3f476a120221256c5d73a8b9c0a9254c8dd7fff09"
            or m["device"]!="cpu" or m["indices"]!=[0,10,20] or not m["quote_costs"]
            or m["samples_sha256"]!=caps.SAMPLES_SHA or m["arms"]!={p:[[512],True,{}] for p in SPAWN}
            or m["spawn_thresholds"]!=dict(zip(SPAWN,(.8,.5)))):
        raise ValueError("Not the frozen spawn preflight")
    for src in ("gemma_unified.py","gemma_quote.py","gemma_unified_audit.py","gemma_wave_kv.py"):
        if m["sources"][src]!=sha(HERE.with_name(src)):
            raise ValueError("Changed request implementation")
    tasks=load_tasks(caps.SAMPLES)
    for arm in SPAWN:
        if read(folder/f"complete-{arm}.json")!={"policy":arm,"requests":3}:
            raise ValueError("Incomplete spawn preflight")
    exposures=[]
    for i in m["indices"]:
        pair=[read(folder/f"{i:02d}-{arm}.json") for arm in SPAWN]
        if any(r["policy"]!=p or any(r[k]!=tasks[i][k] for k in KEYS) for r,p in zip(pair,SPAWN)):
            raise ValueError("Wrong frozen spawn case")
        exposures.append(spawn_witness(*pair))
    if not any(exposures):
        raise ValueError("Earlier admission never changed the CPU pilot")
    return exposures


def exchange_pilot(folder):
    m = read(folder/"run.json")
    if (m["suite"]!="cpu-pilot" or m["device"]!="cpu" or not m["quote_costs"]
            or m["exchange"]!=EX_SPEC or m["indices"]!=[0,10,20] or m["samples_sha256"]!=caps.SAMPLES_SHA
            or m["arms"]!={p:[[128],True,{"128":32}] for p in EXCHANGE}):
        raise ValueError("Not the registered exchange preflight")
    for source in ("gemma_unified.py","gemma_quote.py","gemma_unified_audit.py","gemma_wave_kv.py"):
        if m["sources"][source] != sha(HERE.with_name(source)):
            raise ValueError("Execution source differs from real-weight pilot")
    tasks = load_tasks(caps.SAMPLES)
    changed = []
    for p in EXCHANGE:
        if read(folder/f"complete-{p}.json")!={"policy":p,"requests":3}:
            raise ValueError("Missing complete exchange pilot")
    for i in m["indices"]:
        a,b=(read(folder/f"{i:02d}-{p}.json") for p in EXCHANGE)
        for r,p in zip((a,b),EXCHANGE):
            if (r["policy"]!=p or r["status"]!="success" or any(r[k]!=tasks[i][k] for k in KEYS)
                    or r["projection_modules"]!=85 or r["decoder_layers"]!=30
                    or r["quote_verified_decoder_calls"]!=len(r["trace"])):
                raise ValueError("Pilot quote, geometry or sample mismatch")
        if a["input_sha256"]!=b["input_sha256"] or a["total_exchanged_rows"]!=0:
            raise ValueError("Invalid exchange control")
        first=b["first_exchange_tick"]
        if first!=a["first_exchange_tick"]:
            raise ValueError("Unpaired exchange witness")
        if first is None:
            if b["total_exchanged_rows"] or (a["trace"],a["output_ids"])!=(b["trace"],b["output_ids"]):
                raise ValueError("Unexposed exchange changed the request")
        else:
            if (type(first) is not int or first%2 or not 0<=first<min(len(r["trace"]) for r in (a,b))
                    or a["trace"][:first]!=b["trace"][:first] or not b["total_exchanged_rows"]):
                raise ValueError("Invalid first CPU exchange")
            x,y=(r["trace"][first] for r in (a,b))
            if (any(x[k]!=y[k] for k in ("tick","offer","q","mandatory_q","phase","head"))
                    or not 1<=y["prefix"]-x["prefix"]<=4):
                raise ValueError("Exchange did not preserve paid width")
            changed.append((i, first, b["total_exchanged_rows"]))
    if not changed:
        raise ValueError("No registered pilot request actually swapped seats")
    return changed


def work_row(task, policy, ids, trace, obs, tok, cfg, cap, prompt_digest):
    from hashlib import sha256
    eos = cfg.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    special = set(tok.all_special_ids) | eos | {cfg.pad_token_id}
    if (not ids or len(ids) > cap or any(t in eos for t in ids[:-1])
            or (ids[-1] not in eos and len(ids) != cap)):
        raise ValueError("Returned sequence violates EOS/cap convention")
    prefill = obs["prefill"]
    series, _, _ = ARMS[policy]
    boundary = lambda e: prefill + 256 * (1 + e["head"])
    def opportunity(e):
        return any(p < boundary(e) for p in e["selected"]) and any(p >= boundary(e) for p in e["visible"])
    def sig(obj):
        return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    first = next((i for i, e in enumerate(trace) if opportunity(e)), None)
    ns = len(trace) if first is None else first
    compact = [{k: e[k] for k in ("tick", "offer", "q", "head", "phase", "mandatory_q",
                                   "overflow", "encoded", "promoted")} |
               dict(prefix=len(e["prefix_selected"]), parked=len(e["refresh_only"]),
                    strict_edits=sum(map(len, e["strict_edits"])), opportunity=opportunity(e))
               for e in trace]
    if (not obs["all_projected_widths_match"] or not trace or
            any(e["offer"] != series[i % len(series)] for i, e in enumerate(trace)) or
            obs["actual_decoder_q"] != [e["q"] for e in trace] or
            obs["F"] != len(trace) + obs["reencodes"] or
            obs["Q"] != sum(obs["actual_decoder_q"]) + 256 * obs["reencodes"]):
        raise ValueError("Physical work disagrees with request")
    n = sum(t not in special for t in ids)
    geom = None if first is None else {k: trace[first][k] for k in
        ("tick", "head", "offer", "visible", "selected", "sampled", "dirty_before", "ages_before")}
    return dict({k: task[k] for k in KEYS}, model="diffusiongemma", policy=policy,
                input_sha256=prompt_digest, status="success", output_ids=ids,
                output_text=tok.decode(ids, skip_special_tokens=True), output_tokens=n,
                generation_forwards=obs["F"], generation_physical_padded_rows=obs["Q"],
                decoder_q=sum(obs["actual_decoder_q"]), encoder_q=256 * obs["reencodes"],
                tpf=n/obs["F"], token_efficiency=n/obs["Q"], truncated=ids[-1] not in eos,
                first_opportunity=first, pretrace_sha256=sig(trace[:ns]), first_geometry_sha256=sig(geom),
                prefill_q=prefill, projection_modules=obs["projection_modules"],
                decoder_layers=obs["decoder_layers"], trace=compact)


def rank_witness(trace):
    return [dict(selected=e["selected"], optional=e["prefix_selected"],
                 pre=e["dirty_before"], ids=e["admitted"], edits=e["strict_edits"],
                 revoked=e["revoked"], phase=e["phase"], head=e["head"],
                 prefix_end=e["prefix_end"], ages=e["ages_before"]) for e in trace]


def check_rank_change(a, b, trace_a, trace_b):
    if a["input_sha256"] != b["input_sha256"] or any(a[k]!=b[k] for k in KEYS):
        raise ValueError("Rank inputs differ")
    for tick,(x,y) in enumerate(zip(trace_a,trace_b)):
        for z in (x,y):
            if (len(set(z["selected"])) != len(z["selected"]) or
                    z["optional"] != sorted(set(z["optional"])) or
                    not set(z["optional"]).issubset(z["selected"])):
                raise ValueError("Invalid prefix witness")
        if x["optional"] != y["optional"]:
            if (any(x[k] != y[k] for k in ("pre","phase","prefix_end","head","ages")) or
                    len(x["optional"]) != len(y["optional"]) or
                    set(x["selected"])-set(x["optional"]) != set(y["selected"])-set(y["optional"])):
                raise ValueError("First rank intervention also changed width, dirty rows or non-prefix rows")
            return tick
        if x != y:
            raise ValueError("Rank traces diverged before first rank change")
    if (len(trace_a)!=len(trace_b) or any(a[k]!=b[k] for k in (
            "trace","output_ids","output_tokens","generation_forwards","generation_physical_padded_rows"))):
        raise ValueError("No rank intervention but trajectories differ")
    return None


def rank_pilot(folder):
    """Qualify ranking and price estimates before releasing GPU0 for the sweep."""
    meta = read(folder/"run.json")
    arms = GROUPS[BASE_GROUPS]
    paths = [HERE.with_name(f) for f in ("gemma_unified.py","gemma_unified_audit.py",
                                          "gemma_wave_kv.py","gemma_quote.py")]
    paths.append(ROOT/"benchmarks/checkpoints/diffusiongemma.json")
    if (meta["device"] != "cpu" or meta["suite"] != "cpu-pilot" or meta["indices"] != [0,10,20]
            or meta["samples_sha256"] != caps.SAMPLES_SHA or sha(caps.SAMPLES) != caps.SAMPLES_SHA
            or meta.get("quote_costs") is not True or meta.get("prefix_rank") != {p:RANKS[p] for p in arms}
            or meta["arms"] != {p:[[64],True,{"64":16}] for p in arms}
            or meta["launcher_sha256"] != caps.LAUNCHER_SHA or
            any(meta["sources"][p.name] != sha(p) for p in paths)):
        raise ValueError("Wrong rank CPU preflight or engine")
    for p in arms:
        if read(folder/f"complete-{p}.json") != {"policy":p,"requests":3}:
            raise ValueError("Incomplete rank CPU preflight")
    tasks = load_tasks(caps.SAMPLES)
    changed = []
    for i in meta["indices"]:
        pair = [read(folder/f"{i:02d}-{p}.json") for p in arms]
        for p,r in zip(arms,pair,strict=True):
            if (r["policy"] != p or r["status"] != "success" or
                    any(r[k]!=tasks[i][k] for k in KEYS) or
                    not r["trace"] or r["quote_verified_decoder_calls"] != len(r["trace"])
                    or (r["projection_modules"],r["decoder_layers"]) != (85,30)):
                raise ValueError("Unqualified quote/projection in CPU preflight")
        a,b=pair
        first=a["first_rank_change"]
        if a["input_sha256"] != b["input_sha256"] or first != b["first_rank_change"]:
            raise ValueError("CPU rank arms not paired")
        if first is None:
            if any(a[k]!=b[k] for k in ("output_ids","trace","generation_forwards","generation_physical_padded_rows")):
                raise ValueError("Unchanged CPU selection changed outputs or work")
        else:
            if not (type(first) is int and 0 <= first < min(len(r["trace"]) for r in pair)
                    and a["trace"][:first] == b["trace"][:first]
                    and a["trace"][first]["prefix"] == b["trace"][first]["prefix"] > 0
                    and a["trace"][first]["q"] == b["trace"][first]["q"]):
                raise ValueError("Missing equal-width CPU intervention")
            changed.append(i)
    if not changed:
        raise ValueError("CPU pilot never exercised the rank intervention")
    return changed


def paired(on, off):
    a, b = ARMS[on["policy"]], ARMS[off["policy"]]
    if not (a[0] == b[0] and a[1] and not b[1] and a[2] == b[2]):
        raise ValueError("Not a mask-only intervention")
    for k in (*KEYS, "input_sha256", "first_opportunity", "pretrace_sha256", "first_geometry_sha256"):
        if on[k] != off[k]:
            raise ValueError(f"Different pre-intervention input: {k}")
    if on["first_opportunity"] is None and any(on[k] != off[k] for k in
            ("output_ids", "trace", "output_tokens", "generation_forwards", "generation_physical_padded_rows")):
        raise ValueError("Mask affected an unexposed request")


def run(args, before_cuda=None):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.runtime import diffusiongemma_wave as wave
    from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot
    from experiments.t2t.serving_slots.gemma_unified_audit import measure
    from experiments.t2t.serving_slots.mask_pair_gemma import input_digest

    cpu = before_cuda is None
    if cpu and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU pilot requires hidden GPUs")
    tasks = load_tasks(args.samples)
    native = read(STUDY / "config.json")
    plan = wave.resolve_plan(ROOT / "archive/diffusiongemma-wave/config.json")
    if (sha(args.samples) != caps.SAMPLES_SHA or plan["flowblock_spawn_threshold"] != 0.8
            or plan["cap"] != 2048 or transformers.__version__ != native["transformers"]):
        raise ValueError("Nonregistered suite or runtime")
    group = GROUPS[getattr(args, "pilot_group", 0)] if cpu else GROUPS[int(os.environ["UNIFIED_GROUP"])]
    exchange = group == EXCHANGE or cpu and getattr(args, "exchange", False)
    if exchange:
        group = EXCHANGE
    spawn = group == SPAWN or cpu and getattr(args,"spawn",False)
    if spawn:
        group = SPAWN
    price = exchange or spawn or cpu and getattr(args, "price", False)
    rank = all(arm in RANKS for arm in group)
    transfer = group in (TRANSFER,DWELL)
    masks = group in MASK_GROUPS
    price = price or transfer or masks
    if cpu and group not in (GROUPS[0], GROUPS[1]) and not (rank or exchange or spawn):
        raise ValueError("Unregistered CPU pilot")
    if cpu:
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        tasks = [tasks[i] for i in (0, 10, 20)]
    check_snapshot(args.checkpoint)
    proc = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.bfloat16, device_map={"": "cpu"},
        attn_implementation=native["attention"], experts_implementation=native["experts"]).eval()
    if model.config.canvas_length != 256:
        raise ValueError("Wrong canvas")
    source_paths = [HERE, HERE.with_name("gemma_unified.py"), HERE.with_name("gemma_unified_audit.py"),
                    HERE.with_name("gemma_wave_kv.py"), ROOT / "benchmarks/checkpoints/diffusiongemma.json"]
    if price:
        source_paths.append(HERE.with_name("gemma_quote.py"))
    if transfer:
        source_paths.append(ROOT/"bdlm_serving/methods/t2t/serving_profiles.py")
    args.output.mkdir(mode=0o700, exist_ok=False)
    save_json(args.output / "run.json", dict(
        model=native["model"], revision=native["revision"], torch=str(torch.__version__),
        transformers=transformers.__version__, suite="cpu-pilot" if cpu else "frozen90",
        device="cpu" if cpu else "cuda:0", samples_sha256=sha(args.samples),
        sources={p.name: sha(p) for p in source_paths}, launcher_sha256=caps.LAUNCHER_SHA,
        wave_config=plan, native_config=native, arms={k: ARMS[k] for k in group},
        rng="independent streams seeded by task.seed + canvas ordinal",
        **({"prefix_rank":{p:RANKS[p] for p in group}} if rank else {}),
        **({"exchange": EX_SPEC} if exchange else {}),
        **({"profiles":["economy","q128_prefix_candidate"]} if transfer else {}),
        **({"spawn_thresholds":dict(zip(SPAWN,[0.8,0.5]))} if spawn else {}),
        **({"unmask_only_on_offer":{arm:UNMASK.get(arm) for arm in group}} if masks else {}),
        indices=[t["index"] for t in tasks], **({"quote_costs": True} if price else {})))
    if not cpu:
        before_cuda(model)
        model.to("cuda")
    with torch.inference_mode():
        for task in tasks:
            data = proc.apply_chat_template([{"role": "user", "content": task["prompt"]}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
            digest = input_digest(data)
            rows, witnesses = [], []
            for arm in group:
                series, causal, prefix = ARMS[arm]
                config = dict(plan, prefix_slots=prefix, prefix_mode=RANKS.get(arm,"random" if prefix else "zero"))
                if masks:
                    config.update(audit_offer_masks=True, unmask_offer=UNMASK.get(arm))
                if exchange:
                    config.update(prefix_mode="oldest", exchange_limit=4 if arm == EXCHANGE[1] else 0)
                if spawn and arm==SPAWN[1]:
                    config.update(flowblock_spawn_threshold=0.5)
                if transfer:
                    from bdlm_serving.methods.t2t.serving_profiles import gemma_plan
                    config=gemma_plan(plan,"economy" if arm==group[0] else "q128_prefix_candidate")
                ids, trace, obs = measure(model, data, config, series, task["seed"], causal, price=price)
                row = work_row(task, arm, ids, trace, obs, proc.tokenizer, model.generation_config,
                               plan["cap"], digest)
                if digest != input_digest(data):
                    raise ValueError("Shared input changed")
                if masks:
                    targets = MASK_OFFERS if arm=="replay" else (UNMASK.get(arm) or UNMASK[group[1]],)
                    row["offer_mask_witness"]={str(q):offer_witness(trace,q,UNMASK.get(arm),obs["prefill"])
                                               for q in targets}
                    if arm==group[0] and group not in SWITCH_MASK:
                        old = next(i for i,g in enumerate(GROUPS[:BASE_GROUPS]) if arm in g)
                        twin=read(output(old)/f"{task['index']:02d}-{arm}.json")
                        if any(row[k]!=twin[k] for k in (*KEYS,"input_sha256","output_ids","trace",
                                    "generation_forwards","generation_physical_padded_rows")):
                            raise ValueError("Causal control no longer reproduces historical GPU run")
                    elif arm!=group[0]:
                        compare_offer(rows[0],row,UNMASK[arm])
                if price:
                    row["quote_verified_decoder_calls"] = obs["quoted_decoder_calls"]
                rows.append(row)
                if rank:
                    witnesses.append(rank_witness(trace))
                if exchange or spawn or transfer:
                    witnesses.append(trace)
            if transfer:
                first,n=transfer_witness(*witnesses,*[r["output_ids"] for r in rows],data["input_ids"].shape[1])
                if rows[0]["input_sha256"]!=rows[1]["input_sha256"]:
                    raise ValueError("Unpaired dynamic inputs")
                if group==TRANSFER:
                    twin=read(output(0)/f"{task['index']:02d}-replay.json")
                    if any(twin[k]!=rows[0][k] for k in (*KEYS,"input_sha256","output_ids","trace",
                            "generation_physical_padded_rows","generation_forwards","first_geometry_sha256")):
                        raise ValueError("Original no-prefix replay did not reproduce")
                else:
                    for row,old_arm in zip(rows,TRANSFER,strict=True):
                        twin=read(output(GROUPS.index(TRANSFER))/f"{task['index']:02d}-{old_arm}.json")
                        row["first_different_offer"]=common_offers(twin,row)
                for row in rows:
                    row["first_added_prefix"]=first
                    row["effective_prefix_q"]=n if row["policy"]==group[1] else 0
            if spawn:
                opening=check_early_spawn(*witnesses,*[r["output_ids"] for r in rows])
                for row in rows:
                    row["first_earlier_spawn"]=opening
                    row["dual_ticks"]=sum(e["opportunity"] for e in row["trace"])
                spawn_witness(*rows)
                if not cpu:
                    twin=read(output(1)/f"{task['index']:02d}-q512.json")
                    if any(rows[0][k]!=twin[k] for k in (*KEYS,"input_sha256","output_ids","trace","output_tokens",
                               "generation_forwards","generation_physical_padded_rows","first_geometry_sha256")):
                        raise ValueError(f"Q512 spawn control did not reproduce request {task['index']}")
            if exchange:
                first, swaps = check_exchange(*witnesses, *[r["output_ids"] for r in rows])
                for row in rows:
                    row["first_exchange_tick"] = first
                    row["total_exchanged_rows"] = swaps if row["policy"] == EXCHANGE[1] else 0
                if not cpu:
                    ref = caps.DATA/"gemma-unified90-q128_k32_random-q128_k32_oldest"
                    twin = read(ref/f"{task['index']:02d}-{EXCHANGE[0]}.json")
                    for k in (*KEYS,"input_sha256","output_ids","trace","output_tokens",
                              "pretrace_sha256","generation_forwards","generation_physical_padded_rows"):
                        if rows[0][k]!=twin[k]:
                            raise ValueError(f"Registered oldest control failed replay: {task['index']} {k}")
                    for row, tape in zip(rows,witnesses):
                        row["exchange_by_tick"]=[len(e.get("displaced_mask",())) for e in tape]
                        row["oldest_reproduced"]=True
            if rank:
                first = check_rank_change(*rows, *witnesses)
                for row in rows:
                    row["first_rank_change"] = first
            if group[-1].endswith("_open"):
                paired(*rows)
            for row in rows:
                save_json(args.output / f"{task['index']:02d}-{row['policy']}.json", row)
            print(task["task_id"], [(r["policy"], r["first_opportunity"],
                  r["generation_forwards"], r["generation_physical_padded_rows"]) for r in rows], flush=True)
    for arm in group:
        save_json(args.output / f"complete-{arm}.json", dict(policy=arm, requests=len(tasks)))


def score(folder):
    from bdlm_serving.evaluation.scoring import score as judge
    tasks = load_tasks(caps.SAMPLES)
    meta = read(folder / "run.json")
    pilot = meta["suite"] == "cpu-pilot"
    indices = [0, 10, 20] if pilot else list(range(90))
    if meta["indices"] != indices or meta["samples_sha256"] != caps.SAMPLES_SHA:
        raise ValueError("Incomplete cohort")
    result = []
    for arm in meta["arms"]:
        if read(folder / f"complete-{arm}.json") != dict(policy=arm, requests=len(indices)):
            raise ValueError("Incomplete arm")
        rows = []
        for i in indices:
            task = tasks[i]
            row = read(folder / f"{i:02d}-{arm}.json")
            if row["policy"] != arm or not all(row[k] == task[k] for k in KEYS):
                raise ValueError("Wrong case")
            n, f, q = row["output_tokens"], row["generation_forwards"], row["generation_physical_padded_rows"]
            if q != row["encoder_q"] + row["decoder_q"] or row["tpf"] != n/f or row["token_efficiency"] != n/q:
                raise ValueError("Inconsistent stored metrics")
            rows.append(dict(row, correct=bool(judge(row["output_text"], task)["correct"])))
        cohorts = {"pilot3": rows} if pilot else {"original30": rows[:30], "new60": rows[30:], "full90": rows}
        exchange = meta.get("exchange")==EX_SPEC
        rank = arm in RANKS and not exchange
        spawn = "spawn_thresholds" in meta
        transfer = "profiles" in meta
        key = "first_added_prefix" if transfer else "first_earlier_spawn" if spawn else "first_exchange_tick" if exchange else "first_rank_change" if rank else "first_opportunity"
        yes, no = ("added_prefix","unchanged") if transfer else ("earlier_spawn","unchanged") if spawn else ("swapped","unchanged") if exchange else ("changed_rank","unchanged_rank") if rank else ("exposed","unexposed")
        for name, subset in cohorts.items():
            for scope in ("all",yes,no):
                values = [r for r in subset if scope == "all" or
                          (r[key] is not None) == (scope == yes)]
                n = sum(r["output_tokens"] for r in values)
                f = sum(r["generation_forwards"] for r in values)
                q = sum(r["generation_physical_padded_rows"] for r in values)
                result.append(dict(arm=arm, cohort=name, scope=scope, cases=len(values),
                    correct=sum(r["correct"] for r in values), N=n, F=f, physical_Q=q,
                    macro_TPF=sum(r["tpf"] for r in values)/len(values) if values else None,
                    token_efficiency=n/q if q else None))
    with (folder / "summary.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=result[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    return result


def output(i):
    return caps.DATA / ("gemma-unified90-" + "-".join(GROUPS[i]))


def protected():
    i = int(os.environ["UNIFIED_GROUP"])
    if not 0 <= i < len(GROUPS) or sys.argv[1:] not in (["gemma", "0"], ["gemma", "0", "worker"]):
        raise ValueError("Not registered GPU0 worker")
    if i == 0:
        receipt = read(caps.DATA / caps.OUTPUT / "protected-complete.json")
        if receipt != dict(status="complete", model="gemma", policies=list(caps.POLICIES), requests=90,
                           gpu=0, holder_restored=True, samples_sha256=caps.SAMPLES_SHA, run=caps.OUTPUT):
            raise ValueError("Missing predecessor receipt")
    elif read(output(i-1) / "returned.json") != {"gpu": 0, "restored": True}:
        raise ValueError("Missing preceding group")
    if i == BASE_GROUPS:
        rank_pilot(caps.DATA/"gemma-unified-q64-rank-cpu3-v1")
    if GROUPS[i]==EXCHANGE:
        exchange_pilot(caps.DATA/PILOT_EX)
    if GROUPS[i]==SPAWN:
        spawn_pilot(caps.DATA/PILOT_SPAWN)
        if read(output(1)/"returned.json")!={"gpu":0,"restored":True} or read(output(1)/"complete-q512.json")!={"policy":"q512","requests":90}:
            raise ValueError("Missing independent fixed Q512 replay")
    if GROUPS[i]==TRANSFER and (read(output(0)/"returned.json")!={"gpu":0,"restored":True}
            or read(output(0)/"complete-replay.json")!={"policy":"replay","requests":90}):
        raise ValueError("Missing independent five-slot causal replay")
    if GROUPS[i]==DWELL:
        from experiments.t2t.serving_slots import spawn_mask_gpu as prev
        r=read(prev.OUT/"returned.json")
        if (r["gpu"]!=0 or r["uuid"]!=prev.GPU or r["restored"] is not True or r["pid"]<=0
                or r["full90"]!=prev.sealed(90) or r["original30"]!=prev.sealed(30)):
            raise ValueError("No completed LLaDA mask experiment or GPU0 not returned")
    if GROUPS[i]==DYNAMIC_MASK:
        receipt=read(caps.DATA/"llada21-commit-pair-gpu3-v1"/"returned.json")
        if receipt["restored"] is not True or receipt["gpu"]!=0 or receipt["uuid"]!="GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb":
            raise ValueError("Predecessor did not restore GPU0")
    call = b"run_wave(args, before_cuda=prepare)"
    src = caps.LAUNCHER.read_bytes()
    if sha(caps.LAUNCHER) != caps.LAUNCHER_SHA or src.count(call) != 1:
        raise ValueError("Protected launcher changed")
    os.environ.update(PREFIX_MODE="wave", TRACE_OUT=str(output(i)), FROZEN90_SAMPLES=str(caps.SAMPLES))
    route = b'REPO = RUNTIME / "t2t-experiments"'
    if src.count(route)!=1 or HERE.parents[3].parent != caps.ROOT.parent:
        raise ValueError("Expected one checkout in the registered n164 runtime")
    src = src.replace(route, ("REPO = Path("+repr(str(HERE.parents[3]))+")").encode())
    exec(compile(src.replace(call, b"from experiments.t2t.serving_slots.unified90 import run\n"
                                 b"        run(args, before_cuda=prepare)"), str(caps.LAUNCHER), "exec"),
         {"__file__": str(HERE), "__name__": "__main__"})
    # Only the parent returns here after verified holder restoration.
    for arm in GROUPS[i]:
        if read(output(i) / f"complete-{arm}.json") != dict(policy=arm, requests=90):
            raise ValueError("Incomplete GPU sweep")
    save_json(output(i) / "returned.json", {"gpu": 0, "restored": True})


def queue(*, rank=False, exchange=False, spawn=False, transfer=False, dwell=False):
    indices = (GROUPS.index(DWELL),) if dwell else (GROUPS.index(TRANSFER),) if transfer else (GROUPS.index(SPAWN),) if spawn else (RANK_END,) if exchange else range(BASE_GROUPS,RANK_END) if rank else range(BASE_GROUPS)
    predecessor = "prefix90-llada-q64-progress80-mask-gpu0" if dwell else "prefix90-gemma-q512-spawn50" if transfer else "prefix90-gemma-q128-exchange" if spawn else "prefix90-gemma-rank" if exchange else "prefix90-gemma-unified" if rank else "prefix90-gemma-fixed-caps"
    if any(output(i).exists() for i in indices):
        raise ValueError("Refusing to overwrite an existing registered run")
    for session in ([predecessor,"gemma-rank-cpu3-v1"] if rank else [predecessor]):
        while True:
            check = subprocess.run(["tmux", "has-session", "-t", "="+session], capture_output=True)
            if check.returncode == 1:
                break
            check.check_returncode()
            time.sleep(10)
    if rank:
        print("rank CPU preflight changed requests:",rank_pilot(caps.DATA/"gemma-unified-q64-rank-cpu3-v1"),flush=True)
    if exchange:
        print("Validated exchange exposures:",exchange_pilot(caps.DATA/PILOT_EX),flush=True)
    if spawn:
        print("Validated earlier admission:",spawn_pilot(caps.DATA/PILOT_SPAWN),flush=True)
    for i in indices:
        with output(i).with_suffix(".log").open("x") as log:
            subprocess.run([sys.executable, "-u", str(HERE), "gemma", "0"],
                env=dict(os.environ, UNIFIED_GROUP=str(i)), stdout=log, stderr=subprocess.STDOUT, check=True)
        score(output(i))


if __name__ == "__main__":
    if sys.argv[1:] in (["mask-replay"],["mask-fixed"],["mask-alternate"]):
        for g in {"mask-replay":(DYNAMIC_MASK,),"mask-fixed":FIXED_MASK,"mask-alternate":SWITCH_MASK}[sys.argv[1]]:
            i=GROUPS.index(g)
            with output(i).with_suffix(".log").open("x") as log:
                subprocess.run([sys.executable,"-u",str(HERE),"gemma","0"],env=dict(os.environ,
                               UNIFIED_GROUP=str(i)),stdout=log,stderr=subprocess.STDOUT,check=True)
            score(output(i))
    elif sys.argv[1:] == ["queue"]:
        queue()
    elif sys.argv[1:] == ["queue-prefix-rank"]:
        queue(rank=True)
    elif sys.argv[1:] == ["queue-exchange"]:
        queue(exchange=True)
    elif sys.argv[1:] == ["queue-spawn50"]:
        queue(spawn=True)
    elif sys.argv[1:] == ["queue-transfer"]:
        queue(transfer=True)
    elif sys.argv[1:] == ["queue-dwell"]:
        queue(dwell=True)
    elif sys.argv[1:2] == ["score"]:
        score(Path(sys.argv[2]))
    elif sys.argv[1:2] in (["cpu-pilot"], ["cpu-pilot-q512"], ["cpu-pilot-rank"], ["cpu-pilot-exchange"], ["cpu-pilot-spawn50"]):
        from argparse import Namespace
        folder = Path(sys.argv[2])
        fixed = sys.argv[1] == "cpu-pilot-q512"
        rank = sys.argv[1] == "cpu-pilot-rank"
        exchange = sys.argv[1] == "cpu-pilot-exchange"
        run(Namespace(samples=caps.SAMPLES, output=folder, pilot_group=BASE_GROUPS if rank else int(fixed),
                      exchange=exchange, spawn=sys.argv[1]=="cpu-pilot-spawn50", price=fixed or rank or exchange,
                      checkpoint=ROOT.parent.parent / "models/diffusiongemma-cd40195"))
        if not exchange and sys.argv[1]!="cpu-pilot-spawn50":  # Exposure preflights are not quality results.
            score(folder)
    else:
        protected()
