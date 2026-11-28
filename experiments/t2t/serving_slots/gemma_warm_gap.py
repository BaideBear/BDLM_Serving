"""Account for low-tier forwards and solve a fixed-trajectory IBS break-even."""
import csv
import json
from math import ceil, isclose
from pathlib import Path
from statistics import mean

from .factor_tables import write_csv

ROOT = Path(__file__).resolve().parent / "controls"
PHASES = ("cold_F", "warm_F", "closing_drain_F", "verify_F", "encoder_F")


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def threshold(part, target):
    """Fraction of each individual head-aggregate warm bill removed; no rounding."""
    def tpf(fraction):
        return mean(r["N"]/(r["F"]-fraction*r["warm_F"]) for r in part)
    if tpf(0) >= target:
        return 0.0
    if tpf(1) < target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo+hi)/2
        if tpf(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def account(root, manifest):
    budget_rows = read(root/"fixed-request-comparison/metrics.csv")
    measured = read(root/"gemma-factors/cold-frontier-last-a1/pairs.csv")
    tasks = json.loads(manifest.read_bytes())["samples"]
    assert len(tasks)==90
    output = []
    for q in (32, 64):
        phases = [r for r in measured if int(r["q"])==q]
        eos = read(root/f"gemma-factors/verified-eos/q{q}/pairs.csv")
        if len(phases)!=90 or len(eos)!=90:
            raise ValueError("Incomplete fixed cohorts")
        cases = []
        for i, (r, e, task) in enumerate(zip(phases, eos, tasks)):
            if (not all(r[k]==e[k]==str(task[k]) for k in ("index","seed","dataset","task_id"))
                    or i!=int(r["index"])
                    or not all(r[f"off_{k}"]==e[j] for k,j in
                               (("ids_sha256","ids_sha256"),("N","N"),("correct","correct"),
                                ("F","on_F"),("decoder_Q","on_decoder_Q"),("encoder_F","encoder_F")))):
                raise ValueError("Phase tape does not replay the public request")
            x = {k:int(r[f"off_{k}"]) for k in ("N","F","heads","correct","decoder_Q",*PHASES)}
            # Conditional minimum: every cold first projection actually wrote
            # and at most the final tile can defer the resulting writeback.
            if (sum(x[k] for k in PHASES)!=x["F"]
                    or int(r["off_cold_writes"])!=x["heads"]*256
                    or int(r["off_cold_idle_Q"])!=0
                    or x["cold_F"]!=x["heads"]*ceil((512-q)/q)):
                raise ValueError("Invalid phase accounting or cold-floor assumptions")
            cases.append(x)
        for c,part in (("original30",cases[:30]),("new60",cases[30:]),("frozen90",cases)):
            matching = [r for r in budget_rows if r["model"]=="gemma" and
                        r["cohort"]==c and int(r["budget"])==q]
            refs = {r["method"]:r for r in matching}
            live, ibs = refs["Request"], refs["IBS capped"]
            sums = {k:sum(r[k] for r in part) for k in part[0]}
            macro = mean(r["N"]/r["F"] for r in part)
            if (not isclose(macro,float(live["macro_TPF"]),abs_tol=1e-5) or
                    any(sums[k]!=int(live[j]) for k,j in
                        (("N","N"),("F","F"),("decoder_Q","history_Q"),("correct","correct")))):
                raise ValueError("Reference and phase bills disagree")
            target = float(ibs["macro_TPF"])
            output.append(dict(q=q,cohort=c,n=len(part),**sums,macro_TPF=macro,
                cold_floor_F_per_head=ceil((512-q)/q),capped_IBS_TPF=target,
                free_close_drain_ceiling=mean(r["N"]/(r["F"]-r["closing_drain_F"]) for r in part),
                uniform_warm_F_reduction_fraction=threshold(part,target)))
    return output


if __name__=="__main__":
    write_csv(ROOT/"gemma-factors/cold-frontier-last-a1/warm-gap.csv",
              account(ROOT,ROOT.parents[3]/"benchmarks/frozen90/manifest.json"))
