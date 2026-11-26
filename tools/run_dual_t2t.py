"""Run the paired Q64 two-B32 LLaDA hard-refresh experiment.

The runner is separate from the single-block engine and from ``run_t2t.py``.
It exposes the same ``main(args, prepare=None)`` shape so a holder wrapper can
select this module as its worker without changing the GPU holder protocol.

At nominal Q64 the two B32 blocks have at most 64 editable rows.  The selected
pass is therefore saturated (``effective_q == active_width``); a prompt tail
inside the left B32 block can cap the active width below 64.  ``hard_noedit``
is the saturation/accounting control, not a sparse-policy reference.  There is
no native dual-block archive to compare against; do not invent one.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys


BLOCK = 32
PAIR = 64


@dataclass(frozen=True)
class DualArm:
    budget: int
    fresh: int
    mask: int
    selection: str

    @property
    def edit(self):
        return self.budget - self.fresh - self.mask

    @property
    def policy(self):
        return "t2t_q%d_f%d_m%d_%s" % (self.budget, self.fresh, self.mask, self.selection)


def parse_arm(text):
    q, f, m, selection = text.split(":")
    return DualArm(int(q), int(f), int(m), selection)


def default_arms():
    return [DualArm(PAIR, 0, PAIR, "hard_noedit"),
            DualArm(PAIR, 0, PAIR - 8, "hard_frontier")]


def validate_arms(arms):
    if len(arms) != 2 or len({arm.policy for arm in arms}) != 2:
        raise ValueError("Exactly two distinct paired arms are required")
    by_selection = {arm.selection: arm for arm in arms}
    if set(by_selection) != {"hard_noedit", "hard_frontier"}:
        raise ValueError("Paired arms must be hard_noedit and hard_frontier")
    for arm in arms:
        if arm.budget != PAIR:
            raise ValueError("This experiment is fixed at nominal Q64")
        if arm.fresh != 0:
            raise ValueError("Hard dual arms require fresh=0")
        if arm.edit < 0:
            raise ValueError("Negative edit seats")
    if by_selection["hard_noedit"].edit != 0:
        raise ValueError("hard_noedit requires edit=0")
    if by_selection["hard_frontier"].edit == 0:
        raise ValueError("hard_frontier requires edit>0")
    return by_selection["hard_noedit"], by_selection["hard_frontier"]


def durable(path, data):
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False)
    temp.replace(path)


def write_csv(path, columns, rows):
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def saturation_summary(events):
    selected = [event for event in events if event.get("stage") == "selected"]
    saturated = [event for event in selected if event.get("saturated")]
    full = [event for event in selected if event.get("full_two_block")]
    return dict(
        selected_steps=len(selected),
        saturated_steps=len(saturated),
        unsaturated_steps=len(selected) - len(saturated),
        full_two_block_steps=len(full),
        capped_steps=len(selected) - len(full),
        effective_q_max=max((int(event.get("effective_q", 0)) for event in selected), default=0),
        active_width_max=max((int(event.get("active_width", 0)) for event in selected), default=0),
        selected_q=sum(int(event.get("effective_q", 0)) for event in selected),
        overflow_rows=sum(int(event.get("overflow", 0)) for event in selected),
    )


def _selected_positions(row):
    return tuple(tuple(event.get("absolute_positions", []))
                 for event in row["events"] if event.get("stage") == "selected")


def _stages(row):
    return tuple(event["stage"] for event in row["events"] if event["phase"] != "prefill")


def _reserved_hits(row):
    return sum(len(event.get("reserved_strict_edit_positions", []))
               for event in row["events"] if event.get("stage") == "selected")


def paired_row(task, noedit, frontier):
    noedit_sat, frontier_sat = noedit["dual_saturation"], frontier["dual_saturation"]
    return dict(
        index=task["index"],
        dataset=task["dataset"],
        task_id=task["task_id"],
        seed=task["seed"],
        noedit_N=noedit["output_tokens"],
        noedit_F=noedit["generation_forwards"],
        noedit_Q=noedit["generation_query_rows"],
        noedit_correct=int(bool(noedit["correct"])),
        noedit_selected_steps=noedit_sat["selected_steps"],
        noedit_saturated_steps=noedit_sat["saturated_steps"],
        noedit_full_two_block_steps=noedit_sat["full_two_block_steps"],
        noedit_capped_steps=noedit_sat["capped_steps"],
        noedit_effective_q_max=noedit_sat["effective_q_max"],
        noedit_active_width_max=noedit_sat["active_width_max"],
        noedit_reserved_hits=_reserved_hits(noedit),
        frontier_N=frontier["output_tokens"],
        frontier_F=frontier["generation_forwards"],
        frontier_Q=frontier["generation_query_rows"],
        frontier_correct=int(bool(frontier["correct"])),
        frontier_selected_steps=frontier_sat["selected_steps"],
        frontier_saturated_steps=frontier_sat["saturated_steps"],
        frontier_full_two_block_steps=frontier_sat["full_two_block_steps"],
        frontier_capped_steps=frontier_sat["capped_steps"],
        frontier_effective_q_max=frontier_sat["effective_q_max"],
        frontier_active_width_max=frontier_sat["active_width_max"],
        frontier_reserved_hits=_reserved_hits(frontier),
        same_output=int(noedit["output_ids"] == frontier["output_ids"]),
        same_selected=int(_selected_positions(noedit) == _selected_positions(frontier)),
        same_stages=int(_stages(noedit) == _stages(frontier)),
        delta_correct=int(bool(frontier["correct"])) - int(bool(noedit["correct"])),
    )


def summary_row(model, noedit_arm, frontier_arm, noedit_rows, frontier_rows, paired):
    def total(rows, key):
        return sum(int(row[key]) for row in rows)

    def sat_total(rows, key):
        return sum(int(row["dual_saturation"][key]) for row in rows)

    wins = sum(1 for row in paired if row["delta_correct"] > 0)
    losses = sum(1 for row in paired if row["delta_correct"] < 0)
    both = sum(1 for row in paired if row["noedit_correct"] and row["frontier_correct"])
    neither = sum(1 for row in paired if not row["noedit_correct"] and not row["frontier_correct"])
    return dict(
        model=model,
        requests=len(paired),
        pair="hard_noedit_vs_hard_frontier",
        noedit_policy=noedit_arm.policy,
        frontier_policy=frontier_arm.policy,
        same_output=sum(int(row["same_output"]) for row in paired),
        same_selected=sum(int(row["same_selected"]) for row in paired),
        same_stages=sum(int(row["same_stages"]) for row in paired),
        noedit_correct=total(noedit_rows, "correct"),
        frontier_correct=total(frontier_rows, "correct"),
        wins=wins,
        losses=losses,
        both=both,
        neither=neither,
        delta_correct=wins - losses,
        noedit_N=total(noedit_rows, "output_tokens"),
        frontier_N=total(frontier_rows, "output_tokens"),
        noedit_F=total(noedit_rows, "generation_forwards"),
        frontier_F=total(frontier_rows, "generation_forwards"),
        noedit_Q=total(noedit_rows, "generation_query_rows"),
        frontier_Q=total(frontier_rows, "generation_query_rows"),
        noedit_selected_steps=sat_total(noedit_rows, "selected_steps"),
        frontier_selected_steps=sat_total(frontier_rows, "selected_steps"),
        noedit_saturated_steps=sat_total(noedit_rows, "saturated_steps"),
        frontier_saturated_steps=sat_total(frontier_rows, "saturated_steps"),
        noedit_full_two_block_steps=sat_total(noedit_rows, "full_two_block_steps"),
        frontier_full_two_block_steps=sat_total(frontier_rows, "full_two_block_steps"),
        noedit_capped_steps=sat_total(noedit_rows, "capped_steps"),
        frontier_capped_steps=sat_total(frontier_rows, "capped_steps"),
        noedit_reserved_hits=sum(_reserved_hits(row) for row in noedit_rows),
        frontier_reserved_hits=sum(_reserved_hits(row) for row in frontier_rows),
        note=("Q64 over two B32 is saturated; hard_noedit is the saturation/accounting "
              "control. Selected steps include the mandatory no-MASK verification after a "
              "MASK-filling pass; max_post caps further rechecks. Prompt tails can cap "
              "active_width below 64. No native dual-block archive exists."),
    )


def dual_backend(args, prepare):
    import torch
    import transformers.utils as utils
    if not hasattr(utils, "LossKwargs"):
        from typing import TypedDict
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: int
        utils.LossKwargs = LossKwargs
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.evaluation.frozen import ROOT, sha
    from bdlm_serving.evaluation.scoring import score
    from bdlm_serving.methods.t2t.llada_dual import DualBlockKernel
    from bdlm_serving.methods.t2t.online import package
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask

    for entry in json.loads((ROOT / "benchmarks/checkpoints/llada21.json").read_text()):
        path = args.checkpoint / entry["name"]
        if path.stat().st_size != entry["bytes"] or ("sha256" in entry and sha(path) != entry["sha256"]):
            raise ValueError("Checkpoint mismatch: %s" % path.name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    if prepare:
        for parameter in model.parameters():
            parameter.data = parameter.data.clone()
        prepare(model)
    model.to("cuda")
    eos = [tok.eos_token_id]
    mask = tok.convert_tokens_to_ids("<|mask|>")
    special = set(tok.all_special_ids) | set(eos) | {mask}
    cfg = dict(confidence_threshold=.5, editing_threshold=.5, max_post_steps=16)
    gen_length = int(getattr(args, "gen_length", 2048))

    @torch.inference_mode()
    def run(task, arm):
        prompt = tok.apply_chat_template([dict(role="user", content=task["prompt"])],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        reference = args.reference / ("%02d-native32.json" % task["index"])
        if reference.exists() or getattr(args,"suite","frozen30")=="frozen90":
            ref = json.loads(reference.read_text())
            if ids[0].tolist() != ref["prompt_ids"]:
                raise ValueError("Changed prompt on request %02d" % task["index"])
        kernel = DualBlockKernel(model, arm.budget, arm.fresh, arm.mask,
                                 arm.selection, selection_seed=task["seed"])
        state = kernel.create(ids, cfg, gen_length, task["seed"], mask, eos)
        kernel.run(state)
        output_ids = kernel.output(state)[0].tolist()
        work = [(event["stage"], event["physical_q"]) for event in state.events
                if event["phase"] != "prefill"]
        prefill = sum(event["physical_q"] for event in state.events
                      if event["phase"] == "prefill")
        row = package(
            tok, output_ids, work, eos, special, prefill,
            events=state.events,
            t2t_edits=sum(int(event.get("edited", 0)) for event in state.events))
        row["correct"] = bool(score(row["output_text"], task)["correct"])
        row["dual"] = dict(pair_blocks=2, nominal_q=arm.budget,
                           gen_length=gen_length, mode="rolling_pair")
        row["dual_saturation"] = saturation_summary(state.events)
        return row

    return run


PAIRED_COLUMNS = (
    "index", "dataset", "task_id", "seed",
    "noedit_N", "noedit_F", "noedit_Q", "noedit_correct",
    "noedit_selected_steps", "noedit_saturated_steps",
    "noedit_full_two_block_steps", "noedit_capped_steps",
    "noedit_effective_q_max", "noedit_active_width_max", "noedit_reserved_hits",
    "frontier_N", "frontier_F", "frontier_Q", "frontier_correct",
    "frontier_selected_steps", "frontier_saturated_steps",
    "frontier_full_two_block_steps", "frontier_capped_steps",
    "frontier_effective_q_max", "frontier_active_width_max", "frontier_reserved_hits",
    "same_output", "same_selected", "same_stages", "delta_correct",
)

SUMMARY_COLUMNS = (
    "model", "requests", "pair", "noedit_policy", "frontier_policy",
    "same_output", "same_selected", "same_stages",
    "noedit_correct", "frontier_correct", "wins", "losses", "both", "neither",
    "delta_correct", "noedit_N", "frontier_N", "noedit_F", "frontier_F",
    "noedit_Q", "frontier_Q", "noedit_selected_steps", "frontier_selected_steps",
    "noedit_saturated_steps", "frontier_saturated_steps",
    "noedit_full_two_block_steps", "frontier_full_two_block_steps",
    "noedit_capped_steps", "frontier_capped_steps",
    "noedit_reserved_hits", "frontier_reserved_hits", "note",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("llada21",))
    for key in ("checkpoint", "samples", "output", "reference"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--arm", action="append",
                        help="Q:fresh:mask:selection; default paired Q64 noedit/frontier")
    parser.add_argument("--gen-length", type=int, default=2048)
    parser.add_argument("--suite",choices=("frozen30","frozen90"),default="frozen30")
    parser.add_argument("--reference-pair30",type=Path,help="Required historical true dual64 run for frozen90")
    return parser.parse_args()


def dual_reference90(args,prepare):
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.protocol import verify_work
    archive = args.reference_pair30
    if not archive or args.gen_length!=2048 or getattr(args,"arm",None):
        raise ValueError("frozen90 requires the archived two-B32 run and its fixed noedit configuration")
    tasks = load_tasks(args.samples)
    arm=default_arms()[0]
    read=lambda path:json.loads(path.read_text())
    refmeta=read(archive/"run.json")
    if (refmeta["mode"]!="llada_dual_rolling_pair" or refmeta["model"]!="llada21"
            or refmeta["batch_size"]!=1 or refmeta["cap"]!=2048 or refmeta["pair_blocks"]!=2
            or refmeta["block"]!=32 or refmeta["nominal_q"]!=64 or arm.policy not in refmeta["policies"]
            or read(archive/f"complete-{arm.policy}.json")!=dict(policy=arm.policy,requests=30)):
        raise ValueError("Expected the original true two-B32 reference, not a one-block alias")
    policy="dual64_noedit"
    args.output.mkdir(exist_ok=False,parents=True)
    durable(args.output/"run.json",dict(model="llada21",suite="frozen90",samples_sha256=sha(args.samples),
        block=32,pair_blocks=2,nominal_q=64,mode=refmeta["mode"],cap=2048,batch_size=1,policies=[policy],
        reference_policy30=arm.policy))
    run=dual_backend(args,prepare)
    digests=[]
    for task in tasks:
        i=task["index"]
        row=run(task,arm)
        row.update({k:task[k] for k in ("index","dataset","seed","task_id")},model="llada21",policy=policy)
        verify_work(row)
        audit_dual64(row)
        if i<30:
            src=archive/f"{i:02d}-{arm.policy}.json"
            old=read(src)
            if json.loads(json.dumps(dict(row,policy=arm.policy)))!=old:
                raise ValueError(f"True dual64 trajectory differs from original30 at {i}")
            digests.append(sha(src))
        durable(args.output/f"{i:02d}-{policy}.json",row)
        if i==29:
            durable(args.output/f"reproduced30-{policy}.json",dict(policy=policy,matched=30,reference_sha256=digests))
        print(f"{policy} {i:02d} N={row['output_tokens']} F={row['generation_forwards']} "
              f"physical_Q={row['generation_query_rows']}",flush=True)
    durable(args.output/f"complete-{policy}.json",dict(policy=policy,requests=90))


def audit_dual64(row):
    if (row["dual"]!={"nominal_q":64,"pair_blocks":2,"gen_length":2048,"mode":"rolling_pair"}
            or row["dual_saturation"]!=saturation_summary(row["events"])):
        raise ValueError("Incorrect two-block identity or saturation counts")
    seq=[e for e in row["events"] if e["phase"]!="prefill"]
    work=[]
    for e in seq:
        p=e["absolute_positions"]
        q=e["physical_q"]
        if (e["F"]!=1 or not 0<q<=64 or p!=sorted(set(p)) or len(p)!=q
                or not e["layer_physical_q"] or any(v!=q for v in e["layer_physical_q"])
                or q!=e["actual_q"] or q!=e["effective_q"]
                or (q>32 and len({pos//32 for pos in p})!=2)):
            raise ValueError("Invalid physical two-block trace")
        work.append(q)
    if (len(work)!=row["generation_forwards"] or sum(work)!=row["generation_query_rows"]
            or sum(work)!=row["generation_physical_padded_rows"] or max(work)!=row["max_q"]):
        raise ValueError("True dual64 totals differ from observer")


def main(args, prepare=None):
    from bdlm_serving.evaluation.frozen import load_tasks
    from bdlm_serving.methods.t2t.protocol import verify_work

    if args.model != "llada21":
        raise ValueError("Dual Q64 experiment is LLaDA2.1 only")
    if getattr(args,"suite","frozen30")=="frozen90":
        return dual_reference90(args,prepare)
    arms = [parse_arm(text) for text in args.arm] if getattr(args, "arm", None) else default_arms()
    noedit_arm, frontier_arm = validate_arms(arms)
    tasks = load_tasks(args.samples)
    if len(tasks) != 30:
        raise ValueError("The paired experiment requires all 30 frozen tasks")
    args.output.mkdir(exist_ok=False, parents=True)
    durable(args.output / "run.json", dict(
        model=args.model,
        batch_size=1,
        cap=int(getattr(args, "gen_length", 2048)),
        block=BLOCK,
        pair_blocks=2,
        nominal_q=PAIR,
        mode="llada_dual_rolling_pair",
        policies=[noedit_arm.policy, frontier_arm.policy],
        arms=[dict(budget=arm.budget, fresh=arm.fresh, mask=arm.mask,
                   edit=arm.edit, selection=arm.selection)
              for arm in (noedit_arm, frontier_arm)],
        note=("Q64 over two B32 is saturated; hard_noedit is the saturation/accounting "
              "control. Selected steps include the mandatory no-MASK verification after a "
              "MASK-filling pass; max_post caps further rechecks. Prompt tails can cap "
              "active_width below 64. No native dual-block archive exists."),
    ))

    run = dual_backend(args, prepare)
    noedit_rows, frontier_rows, paired = [], [], []
    for task in tasks:
        rows = {}
        for arm in (noedit_arm, frontier_arm):
            row = run(task, arm)
            row.update({key: task[key] for key in ("index", "task_id", "dataset", "seed")},
                       model=args.model, policy=arm.policy)
            verify_work(row)
            durable(args.output / ("%02d-%s.json" % (task["index"], arm.policy)), row)
            rows[arm.selection] = row
        noedit, frontier = rows["hard_noedit"], rows["hard_frontier"]
        noedit_rows.append(noedit)
        frontier_rows.append(frontier)
        paired.append(paired_row(task, noedit, frontier))
        print("pair %02d %s N=%d/%d F=%d/%d Q=%d/%d same_output=%d" % (
            task["index"], task["dataset"], noedit["output_tokens"],
            frontier["output_tokens"], noedit["generation_forwards"],
            frontier["generation_forwards"], noedit["generation_query_rows"],
            frontier["generation_query_rows"], paired[-1]["same_output"]), flush=True)

    for arm in (noedit_arm, frontier_arm):
        durable(args.output / ("complete-%s.json" % arm.policy),
                dict(policy=arm.policy, requests=len(tasks)))
    write_csv(args.output / "paired.csv", PAIRED_COLUMNS, paired)
    summary = summary_row(args.model, noedit_arm, frontier_arm,
                          noedit_rows, frontier_rows, paired)
    write_csv(args.output / "summary.csv", SUMMARY_COLUMNS, [summary])
    print("summary correct %d/%d -> %d/%d; same_output=%d/%d; selected steps %d/%d" % (
        summary["noedit_correct"], summary["requests"],
        summary["frontier_correct"], summary["requests"],
        summary["same_output"], summary["requests"],
        summary["noedit_selected_steps"], summary["frontier_selected_steps"]))


if __name__ == "__main__":
    main(parse_args())
