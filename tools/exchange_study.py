"""Calibrate before opening the evaluation paths, then audit exchanged physical seats."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST, sha
from bdlm_serving.methods.t2t.exchange import SCHEMA, MODES, choose, decode, position_keys
from tools.t2t_results import table


def states(folder, sample):
    with gzip.open(folder / f"{sample['index']:02d}.trace.jsonl.gz", "rt") as stream:
        rows = map(json.loads, stream)
        meta, prompt = next(rows), next(rows)
        assert all(meta[k] == sample[k] for k in ("index","dataset","task_id","seed"))
        last = prior = None
        mask = prompt["mask_id"]
        for row in rows:
            block, before = row["block"], row["before"]
            new = block != last
            if new:
                valid = list(range(max(0,prompt["prompt_length"]-block),32))
                fresh, strict, writes, clock = set(), set(), [-1]*32, 0
            else:
                assert before == prior
            clock += 1
            resolved = [x != mask for x in before]
            after = row["after"]
            assert len(after) == 32 and row["accepted"] == [x != mask for x in after]
            changed = {i for i in valid if before[i] != after[i]}
            admitted = {i for i in changed if not resolved[i] and after[i] != mask}
            edited = {i for i in changed if resolved[i] and after[i] != mask}
            if not new and any(not resolved[i] for i in valid):
                yield (valid,resolved,fresh,strict,writes,clock), admitted, edited
            for i in changed:
                writes[i] = clock
            fresh, strict, prior, last = admitted | edited, edited, after, block


def fit(calibration, manifest):
    src = calibration/"weights.json"
    train = json.loads(src.read_text())["samples"]
    unique = lambda data,k: {(r["dataset"],r[k]) for r in data}
    assert Counter(s["dataset"] for s in train) == {k:8 for k in ("gsm8k","math500","humaneval")}
    assert len(unique(train,"source_index")) == len({s["prompt_sha256"] for s in train}) == 24
    assert unique(train,"source_index").isdisjoint(unique(manifest,"source_index"))
    assert {s["prompt_sha256"] for s in train}.isdisjoint(s["prompt_sha256"] for s in manifest)
    counts, hits, n, positives = Counter(),Counter(),Counter(),Counter()
    for s in train:
        for (v,r,f,t,w,c),admit,edit in states(calibration,s):
            for i,key in position_keys(v,r,f,t,w,c).items():
                group = key[0]
                event = i in (edit if group == "old" else admit)
                counts[key] += 1
                hits[key] += event
                n[group] += 1
                positives[group] += event
    prior = {k:positives[k]/n[k] for k in n}
    assert positives["old"] == json.loads(src.read_text())["calibration_old_positives"]
    return dict(schema=SCHEMA, calibration_manifest_sha256=sha(src),
        samples=24, prior_strength=16, counts=n, positives=positives, fallback=prior,
        rates=[dict(key=k,n=counts[k],hits=hits[k],probability=(hits[k]+16*prior[k[0]])/(counts[k]+16))
               for k in sorted(counts)])


def replay(folder, manifest, rates):
    assert json.loads((folder/"verified.json").read_text()) == dict(model="llada21",matched_requests=30)
    records, coverage = defaultdict(Counter), defaultdict(Counter)
    for task in manifest:
        for s,admit,edit in states(folder,task):
            v,r,f,t,w,c = s
            for q,cap in ((4,1),(8,1),(16,2)):
                base,_,_ = choose(*s, q, 0, rates, 1.)
                for mode,value in MODES.items():
                    chosen,reserve,offers = choose(*s,q,cap,rates,value)
                    assert set(reserve) == set(chosen)-set(base) and len(chosen)==len(base)
                    target = edit - f
                    cover = coverage[q,cap,mode]
                    cover.update(states=1, old_targets=len(target), base_old_hits=len(target.intersection(base)),
                                 selected_old_hits=len(target.intersection(chosen)), newly_covered=len(target.intersection(reserve)),
                                 lost_old_hits=len(target.intersection(set(base)-set(chosen))),
                                 mask_admissions=len(admit), base_admits=len(admit.intersection(base)),
                                 selected_admits=len(admit.intersection(chosen)), selected_Q=len(chosen),
                                 same_state_base_Q=len(base))
                    for victim_type in ("mask","old"):
                        key = q,cap,mode,victim_type
                        totals = records[key]
                        totals["states"] += 1
                        for candidate,victim,swap in offers:
                            if ("old" if r[victim] else "mask") != victim_type:
                                continue
                            totals["offers"] += 1
                            totals["swaps"] += swap
                            h = int(candidate in edit)
                            cost = int(victim in (edit if r[victim] else admit))
                            totals["candidate_event_swapped"] += swap*h
                            totals["victim_event_swapped"] += swap*cost
                            totals["both_events_swapped"] += swap*h*cost
                            totals["candidate_event_declined"] += (not swap)*h
                            totals["victim_event_declined"] += (not swap)*cost
    cols = ("states","offers","swaps","candidate_event_swapped","victim_event_swapped",
            "both_events_swapped","candidate_event_declined","victim_event_declined")
    for z in coverage.values():
        assert z["old_targets"] == 191 and z["selected_Q"] == z["same_state_base_Q"]
        assert z["newly_covered"]-z["lost_old_hits"] == z["selected_old_hits"]-z["base_old_hits"]
    return ([dict(q=q,cap=e,mode=m,victim=v,**{k:z[k] for k in cols})
             for (q,e,m,v),z in sorted(records.items())],
            [dict(q=q,cap=e,mode=m,**z) for (q,e,m),z in sorted(coverage.items())])


def audit_pair(arm, base):
    for k in ("model","index","seed","task_id","dataset","prefill_rows"):
        if arm[k] != base[k]:
            raise ValueError(f"Pair metadata mismatch: {k}")
    if arm["status"] != "success" or base["status"] != "success":
        raise ValueError("Failed request in paired audit")
    events, controls = arm["events"], base["events"]
    for tick,e in enumerate(events):
        reserved = e.get("reserved_extra_edit_positions", [])
        if tick >= len(controls):
            raise ValueError("Missing control round before first swap")
        b = controls[tick]
        if not reserved:
            if e != b:
                raise ValueError(f"Pre-swap event diverged: {arm['index']} at {tick}")
            continue
        if b["stage"] != "selected" or e["stage"] != "selected":
            raise ValueError("Swap outside sparse selection")
        for field in ("block","context_version","step_in_block","hard_clock","mandatory_fresh",
                      "masked_before","actual_q","physical_q","layer_physical_q","F","prefix_kv_length"):
            if e[field] != b[field]:
                raise ValueError(f"First-intervention mismatch: {field}")
        picked,neutral = set(e["absolute_positions"]),set(b["absolute_positions"])
        if set(reserved) != picked-neutral or len(neutral-picked) != len(reserved):
            raise ValueError("Reserved seats are not exactly the new physical rows")
        displaced = b["selected_masked_before"]-e["selected_masked_before"]
        if displaced not in (0,len(reserved)):
            raise ValueError("Unexpected mixed MASK and old victims")
        return dict(prefix_events=tick, anchor_swaps=len(reserved), victim="mask" if displaced else "old",
                    anchor_mask_lost=displaced, anchor_added_strict=len(e["reserved_strict_edit_positions"]),
                    anchor_net_accept=e["accepted"]-b["accepted"], anchor_net_strict=e["hard_strict_edits"]-b["hard_strict_edits"])
    for k in ("output_ids","output_tokens","generation_forwards","generation_query_rows","events"):
        if arm[k] != base[k]:
            raise ValueError(f"No-swap request diverged: {k}")
    return dict(prefix_events=len(events),anchor_swaps=0,victim="none",anchor_mask_lost=0,
                anchor_added_strict=0,anchor_net_accept=0,anchor_net_strict=0)


def anchors(runs,control,manifest,rates_path):
    records = []
    for run in runs:
        config = json.loads((run/"run.json").read_text())
        assert config["exchange_rates_sha256"] == sha(rates_path)
        assert json.loads((run/"qualified.json").read_text()) == dict(model="llada21",full_width=32,matched=30)
        for arm,name in zip(config["arms"],config["policies"],strict=True):
            q = arm["budget"]
            cname = f"t2t_q{q}_f0_m{q}_hard_noedit"
            assert arm["selection"] in MODES and json.loads((run/f"complete-{name}.json").read_text()) == dict(policy=name,requests=30)
            for sample in manifest:
                i = sample["index"]
                new = json.loads((run/f"{i:02d}-{name}.json").read_text())
                base = json.loads((control/f"{i:02d}-{cname}.json").read_text())
                assert all(new[k] == sample[k] for k in ("index","seed","task_id","dataset"))
                records.append(dict(q=q,policy=name,index=i,dataset=sample["dataset"],task_id=sample["task_id"],
                                    **audit_pair(new,base)))
    return records


def report(folder):
    def read(path):
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    costs = read(folder/"paired-cost.csv")
    coverage = read(folder/"native-coverage.csv")
    uncertainty = {r["policy"]:r for r in read(folder.parent/"budget-paired-intervals.csv")
                   if r["model"] == "llada21"}
    first = read(folder/"first-swap-audit.csv")
    ref = {int(r["nominal_q"]):r for r in read(folder.parent/"hard/llada21-online-summary.csv")
           if r["policy"].endswith("_hard_noedit") and r["dataset"]=="all"}
    assert len(costs)==6 and len(coverage)==6 and len(first)==180
    idx = {(int(r["nominal_q"]),r["policy"].rsplit("_",1)[1]):r for r in costs}
    lines = ["# LLaDA 2.1：预测旧位改写，也预测换走 MASK 的代价", "",
        "Q4 两种交换规则的答对数和 TPF 点估计都超过同预算零预留；Q8 两者都变差，Q16 四倍规则保持判分但 TPF 下降。"
        "这不是跨预算优势。候选能否编辑、实际能否多答对、是否更省物理 Q 是三个不同问题。"
        "运行前确定的规则见[协议](PROTOCOL.md)。", "",
        "## 端到端实测", "",
        "两组分别通过相同 30 题全宽复现，之后实跑六个稀疏配置。冻结三个数据集各 10 题及种子。"
        "TPF 是逐题 N/F 均值，效率是 ΣN/Σ物理 Q；F/Q 含全宽初始化、选位、验证与提交，不含独立 prompt prefill。"
        "必刷超额不截断；表中的 Q4/8/16 不是完整请求的实际 Q/F。", "",
        "| 名义 Q | 分配 | 答对/30 | TPF | token efficiency | 实际 Q/F | 总 F | 物理 Q | 赢/输¹ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for q in (4,8,16):
        for label,r,win in [("零预留",ref[q],"参照")]+[
                (label,idx[q,m],f"{idx[q,m]['only_policy']}/{idx[q,m]['only_reference']}")
                for m,label in (("equal","等权"),("edit4","编辑四倍"))]:
            lines.append(f"| {q} | {label} | {r['correct']} | {float(r['tpf']):.3f} | "
                         f"{float(r['token_efficiency']):.2%} | {float(r['q_per_f']):.2f} | "
                         f"{int(r['F']):,} | {int(r['Q']):,} | {win} |")
    lines += ["", "¹ 同预算逐题配对，只列本策略正确数 / 只列零预留正确数。"
        "这些是同一份反复用于研究的 30 题，不能作独立质量验证或延迟测试。", "",
        "![真实稀疏结果与相同窗口原始 Optimus IBS、OBS 的数值对照](figures/llada21-exchange-study.svg)", "",
        "## 实际换位", "",
        "预留只统计原零预留本来不算的行。先比较两个旧位；只有原选择没有旧回填时才拿候选与末个可换 MASK 比，始终保护首个可算 MASK。"
        "两表是位置和过去写入信息的查找表，不查询 logits 或未算的缓存。", "",
        "| Q | 编辑价值 | 换入旧行 | 换入行严格改写 | 挤占 MASK | 全程必刷超额 Q² | ΔTPF（配对 95% 区间） |",
        "|---:|---:|---:|---:|---:|---:|---|"]
    for (q,m),r in sorted(idx.items()):
        u = uncertainty[r["policy"]]
        lines.append(f"| {q} | {1 if m=='equal' else 4} | {r['reserved_seats']} | "
                     f"{r['reserved_strict_hits']} | {r['displaced_mask_rows']} | {r['mandatory_overflow_Q']} | "
                     f"{float(r['delta_tpf']):+.4f} [{float(u['delta_TPF_lo']):+.4f}, {float(u['delta_TPF_hi']):+.4f}] |")
    c = Counter(r["victim"] for r in first)
    lines += ["", "² 必刷超额已计在总物理 Q，不是换位造成的额外行数。配对区间按数据集、请求重采样，不涵盖重复选策略或换种子的影响。", "",
        f"180 个策略/题目组合：第一次实际交换为旧换旧 {c['old']}、旧换 MASK {c['mask']}、整题未换 {c['none']}。"
        "逐题核对了第一次交换前的全部记录事件、交换当轮的物理行数及新增行集合；没有交换的题还核对了输出 ID、F/Q 和完整事件。"
        "这里核对的是可观测事件，不是逐个比较所有层缓存张量。", "",
        "## 为什么离线召回不足以判断质量", "",
        "在同来源、与评测题不重叠的 24 道校准题，非 fresh 旧行改写 169/16,092，MASK 解码 5,487/18,108。"
        "在 frozen30 的 1,170 个原生选位状态回放，Q16 三种选择均只占 19,800 行："
        "零预留、等权、编辑四倍分别覆盖 7、34、41 个可选旧编辑（共 191 个），同时保留 3,098、3,097、3,074 个当轮会解码的 MASK。"
        "这衡量同一原生历史的交换成本，不是真实稀疏分支的反事实。"
        "在线的 forced MASK 选择、未来的可计算集合和输出长度都可能改变。", "",
        "n164 固定 CPU 核心的 Q16 单步选位中位耗时：零预留 9.28 微秒、等权 47.81、编辑四倍 47.83。"
        "不含 GPU 同步、数据搬运或请求延迟。", "",
        "[逐题评分](llada21-online-requests.csv) · [Q 分账](paired-cost.csv) · [分阶段 F/Q](llada21-online-phases.csv) · "
        "[首次换位审计](first-swap-audit.csv) · [原生覆盖](native-coverage.csv) · [换入/换出事件交叉表](native-physical-swaps.csv) · "
        "[计时](selector-cpu.csv) · [冻结表](rates.json)", "",
        "复算：`tools.exchange_study fit` 使用原校准目录；`replay` 使用原生评测轨迹。"
        "`python -m tools.run_t2t --model llada21` 带 `--exchange-rates` 和 `--arm Q:0:MASK:hard_exchange_equal` 或 "
        "`hard_exchange_edit4`，其中 `MASK` 分别是 3、7、14。用 `tools.t2t_results` 和 `tools.selection_effect` "
        "分别合并评分与分账，`tools.exchange_study anchors` 检查第一处分叉。运行 `python -m tools.budget_index` 更新总表，"
        "`python -m tools.render_hard_study --study exchange --model llada21` 重画图。", ""]
    (folder/"README.md").write_text("\n".join(lines))


def propensity_counts(event, seed, salt):
    """A one-seat randomized action reveals the ranker's reward only on a match."""
    from random import Random
    audit = event["placebo_exchange"]
    base, pool = set(audit["base"]), audit["pool"]
    preferred, = set(audit["ranked"]) - base
    chosen, = set(audit["executed"]) - base
    k = len(pool)
    block = event["block"]
    draw = Random(f"exchange-placebo:{salt}:{seed}:{block}:{event['hard_clock']}").sample(pool, 1)
    if (preferred not in pool or sorted(set(pool)) != pool or draw != [chosen] or
            [32*block+chosen] != event["reserved_extra_edit_positions"]):
        raise ValueError("Unverified one-seat sampling design")
    labels = event["reserved_strict_edit_positions"]
    if labels not in ([], [32*block+chosen]):
        raise ValueError("Invalid observed candidate outcome")
    hit, match = int(bool(labels)), int(chosen == preferred)
    return Counter(decisions=1, uniform_hits=hit, collisions=match, collision_hits=hit*match,
                   weighted_hits=k*match*hit, weighted_support=k*match, squared_weights=k*k*match,
                   expected_collisions=1/k, singleton=int(k==1))


def half_weight(chosen_time, pool_times, oldest):
    """Policy/draw probability ratio; ties have equal fractional inclusion."""
    k = len(pool_times)
    m = (k + 1) // 2
    tied = pool_times.count(chosen_time)
    if not tied:
        raise ValueError("Chosen timestamp absent")
    ahead = sum(x < chosen_time if oldest else x > chosen_time for x in pool_times)
    return (k / m) * min(1.,max(0.,(m-ahead)/tied))


def propensity(runs, folder, manifest):
    from random import Random
    import numpy as np
    groups, seen, request_rows = defaultdict(dict), set(), []
    for run in runs:
        meta = json.loads((run/"run.json").read_text())
        if (meta["model"] != "llada21" or meta.get("seed_offset", 0) or
                json.loads((run/"qualified.json").read_text()) != dict(model="llada21",full_width=32,matched=30)):
            raise ValueError("Native bridge or seed mismatch")
        for arm, policy in zip(meta["arms"],meta["policies"],strict=True):
            q, suffix = arm["budget"], arm["selection"].removeprefix("hard_exchange_random_")
            if suffix not in ("a","b") or q not in (4,8) or (q,suffix) in seen or json.loads(
                    (run/f"complete-{policy}.json").read_text()) != dict(policy=policy,requests=30):
                raise ValueError("Expected distinct, complete preregistered arms")
            seen.add((q,suffix))
            for task in manifest:
                i = task["index"]
                row = json.loads((run/f"{i:02d}-{policy}.json").read_text())
                if row["policy"] != policy or any(row[k] != task[k] for k in ("index","task_id","seed","dataset")):
                    raise ValueError("Wrong request")
                parts = {s:Counter() for s in ("all","mask","backfill")}
                last, block = {}, None
                for e in row["events"]:
                    if e["stage"] == "bootstrap":
                        last, block = {}, e["block"]
                    if "placebo_exchange" in e:
                        c = propensity_counts(e, row["seed"], suffix)
                        a = e["placebo_exchange"]
                        chosen, = set(a["executed"]) - set(a["base"])
                        if e["stage"] != "selected" or block != e["block"]:
                            raise ValueError("Unexpected sparse decision")
                        times = [last[i] for i in a["pool"]]
                        c.update(age_informative=int(min(times) != max(times)),
                                 oldest_hits=c["uniform_hits"]*half_weight(last[chosen],times,True),
                                 newest_hits=c["uniform_hits"]*half_weight(last[chosen],times,False))
                        parts["all"].update(c)
                        parts["mask" if a["masked_victims"] else "backfill"].update(c)
                    elif e.get("reserved_extra_edit_positions"):
                        raise ValueError("Missing randomized-action record")
                    if e["stage"] in ("selected","bootstrap"):
                        last.update((i-32*block,e["hard_clock"]) for i in e["absolute_positions"])
                for s,c in parts.items():
                    groups[q,suffix,s][i] = c
    if seen != {(q,s) for q in (4,8) for s in ("a","b")}:
        raise ValueError("Expected both budgets and both random streams")
    totals = []
    fields = ("decisions","uniform_hits","collisions","collision_hits","weighted_hits",
              "weighted_support","squared_weights","expected_collisions","singleton",
              "age_informative","oldest_hits","newest_hits")
    for (q,salt,stratum), data in sorted(groups.items()):
        rows = [data[t["index"]] for t in manifest]
        for t,c in zip(manifest,rows,strict=True):
            request_rows.append(dict(q=q,salt=salt,stratum=stratum,index=t["index"],
                dataset=t["dataset"],task_id=t["task_id"],**{k:c[k] for k in fields}))
        c = sum(rows,Counter())
        rng = Random(17594)
        samples = [[data[t["index"]] for t in manifest if t["dataset"]==d] for d in ("gsm8k","math500","humaneval")]
        assert all(len(v)==10 for v in samples)
        diffs,age_diffs = [],[]
        for _ in range(2000):
            draw = [r for group in samples for r in rng.choices(group,k=len(group))]
            denom = sum(r["decisions"] for r in draw)
            if denom:
                diffs.append(sum(r["weighted_hits"]-r["uniform_hits"] for r in draw)/denom)
                age_diffs.append(sum(r["oldest_hits"]-r["newest_hits"] for r in draw)/denom)
        n = c["decisions"]
        bounds = np.percentile(diffs,[2.5,97.5]).tolist() if diffs else ["",""]
        age_bounds = np.percentile(age_diffs,[2.5,97.5]).tolist() if age_diffs else ["",""]
        totals.append(dict(q=q,salt=salt,stratum=stratum,**{k:c[k] for k in fields},
            uniform_rate=c["uniform_hits"]/n if n else "", ranked_ipw_rate=c["weighted_hits"]/n if n else "",
            rank_minus_uniform=(c["weighted_hits"]-c["uniform_hits"])/n if n else "",
            delta_lo=bounds[0],delta_hi=bounds[1],
            oldest_rate=c["oldest_hits"]/n if n else "", newest_rate=c["newest_hits"]/n if n else "",
            oldest_minus_newest_lo=age_bounds[0],oldest_minus_newest_hi=age_bounds[1],
            effective_matches=c["weighted_support"]**2/c["squared_weights"] if c["squared_weights"] else 0))
    folder.mkdir(parents=True,exist_ok=True)
    table(folder/"candidate-propensity.csv", totals)
    table(folder/"candidate-propensity-requests.csv", request_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("fit","replay","anchors","report","propensity"))
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--rates", type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--coverage",type=Path)
    p.add_argument("--run",type=Path,action="append")
    a = p.parse_args()
    if a.action not in ("report","propensity") and a.rates is None:
        p.error("--rates is required for calibration, replay and audit")
    frozen = json.loads(MANIFEST.read_text())["samples"]
    if a.action == "propensity":
        if not a.run:
            p.error("propensity requires the randomized run folders")
        propensity(a.run,a.dir,frozen)
    elif a.action == "fit":
        a.rates.write_text(json.dumps(fit(a.dir,frozen),indent=2)+"\n")
    elif a.action == "replay":
        if a.out is None:
            p.error("replay requires --out")
        offers,counts = replay(a.dir,frozen,decode(json.loads(a.rates.read_text())))
        table(a.out,offers)
        if a.coverage:
            table(a.coverage,counts)
    elif a.action == "anchors":
        if not a.run or not a.out:
            p.error("anchors requires --run and --out; --dir is the same-budget zero-reserve run")
        table(a.out,anchors(a.run,a.dir,frozen,a.rates))
    else:
        report(a.dir)
