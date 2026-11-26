"""Stop at the first archived Q8 crowded fork; CPU endpoint splicing, no generation."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from fractions import Fraction
import math
import os
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST, ROOT, sha
from tools import beneficiary_rows as ben
from tools import context_gates_audit as audit
from tools.t2t_results import table


RAW = ROOT.parent / 't2t-native-1125'
SCORED = ROOT / 'experiments/t2t/calibrated/embedding_context/crowded/llada21-online-requests.csv'
FIT = RAW / 'admission-v1-private/victim.json'
OUT = RAW / 'admission-fork-v1'
POLICIES = ('base', 'always_rank', 'hard_admit_low', 'uniform_timing_expectation', 'oracle_diagnostic')
CELLS = {(h,p): f'swaps_hit{h}_paid{p}' for h in (0,1) for p in (0,1)}


@dataclass(frozen=True)
class Endpoint:
    correct: int
    N: int
    F: int
    Q: int

    def metrics(self):
        return dict(correct=self.correct, N=self.N, F=self.F, physical_Q=self.Q, tpf=Fraction(self.N,self.F))


def marks(d, p):
    values = audit.load(d / f'marks-{p}.json')
    audit.check(set(values) == {str(i) for i in range(30)} and all(type(x) is bool for x in values.values()),
                'Incomplete or invalid archived marks')
    return values


def scored_rows(path, policies):
    with path.open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['policy'] in policies]
    by = {(r['policy'], int(r['index'])):r for r in rows}
    audit.check(len(by)==len(rows)==30*len(policies) and set(by)=={(p,i) for p in policies for i in range(30)},
                'Incomplete or duplicate published endpoints')
    return by


def endpoint(raw, correct, ref):
    expected = dict(index=raw['index'], policy=raw['policy'], model=raw['model'], dataset=raw['dataset'],
                    task_id=raw['task_id'], correct=int(correct), failed=0, truncated=int(raw['truncated']),
                    N=raw['output_tokens'], F=raw['generation_forwards'], Q=raw['generation_query_rows'],
                    physical_Q=raw['generation_physical_padded_rows'], max_q=raw['max_q'])
    audit.check(all(str(v)==ref[k] for k,v in expected.items()) and expected['Q']==expected['physical_Q'],
                'Raw branch and published endpoint differ')
    return Endpoint(expected['correct'],expected['N'],expected['F'],expected['physical_Q'])


@dataclass(frozen=True)
class Case:
    index: int
    dataset: str
    base: Endpoint
    rank: Endpoint
    anchor: bool
    approved: bool
    risk: float | None
    hit: bool
    paid: bool


def first_decision(raw, anchor, fit, decide):
    if anchor is None:
        return False, None
    clock, block = anchor
    # Only the common pre-forward screen enters the parent's frozen gate. Never retry.
    move = raw['events'][clock-1]['crowded_screen']
    audit.check(move['reason']=='swap_mask', 'Not a crowded first fork')
    approved, risk = decide(move, fit, 'hard_admit_low', raw['seed'], block, clock)
    audit.check(type(approved) is bool and type(risk) is float and math.isfinite(risk) and 0 <= risk <= 1,
                'Unexpected admission decision or score')
    return approved, risk


def cases(fit, decide):
    runs = [RAW / f'context-once-{s}-q8-v1' for s in ('pair','random')]
    history = RAW / 'hard-v1-llada21'
    owners = ben.crowded_cohort(runs, history)
    keys = audit.ONCE
    correct = {p:marks(owners[p], p) for p in keys}
    noedit_marks = marks(history, ben.NOEDIT)
    published = scored_rows(SCORED, keys)
    historical = scored_rows(ROOT / 'experiments/t2t/hard/llada21-online-requests.csv', [ben.NOEDIT])
    samples = audit.load(MANIFEST)['samples']
    audit.check([s['index'] for s in samples]==list(range(30)) and
                Counter(s['dataset'] for s in samples)==dict.fromkeys(ben.DATASETS[1:],10), 'Not original frozen30')
    for s in samples:
        i = s['index']
        def load(d, p):
            r = audit.load(d / f'{i:02d}-{p}.json')
            ben._identity(r, s, 'llada21', p)
            return r
        try:
            full = load(history, 'fullwidth')
            audit.check(all(audit.same(load(d,'fullwidth'),full) for d in runs), 'Fullwidth qualifiers differ')
            raw = [load(owners[p],p) for p in keys]
            control = load(history, ben.NOEDIT)
            ben.compare_base(raw[0],control)
            results = [audit.audit_crowded_request(r,s,p) for r,p in zip(raw,keys,strict=True)]
            counts, anchors = zip(*results, strict=True)
            audit.audit_crowded_match(raw,list(anchors))
            ends = [endpoint(r, correct[p][str(i)], published[p,i]) for r,p in zip(raw,keys,strict=True)]
            audit.check(ends[0] == endpoint(control, noedit_marks[str(i)], historical[ben.NOEDIT,i]),
                        'Historical Q8 noedit endpoint differs')
            audit.check(all(a['output_ids'] != b['output_ids'] or ea.correct == eb.correct
                            for a,ea in zip(raw,ends) for b,eb in zip(raw,ends)), 'Identical outputs scored differently')
            approve, risk = first_decision(raw[0], anchors[0], fit, decide)
            yield Case(i,s['dataset'],ends[0],ends[1],anchors[0] is not None,approve,risk,
                       bool(counts[1]['rank_direct_hits']),bool(counts[0]['victim_accepted']))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f'Frozen request {i}: {exc}') from exc


def evaluate(cohort):
    """Exact inclusion marginals for uniform subsets without replacement, never sampled placements."""
    eligible = Counter(c.dataset for c in cohort if c.anchor)
    approved = Counter(c.dataset for c in cohort if c.approved)
    requests, totals = [], {}
    for c in cohort:
        audit.check(c.anchor or (not (c.approved or c.hit or c.paid) and c.base==c.rank), 'Invalid no-anchor case')
        p_uniform = Fraction(approved[c.dataset], eligible[c.dataset]) if c.anchor else Fraction()
        a,b = c.base.metrics(),c.rank.metrics()
        for mode,p in zip(POLICIES,(0, int(c.anchor), int(c.approved), p_uniform, int(c.hit and not c.paid)),strict=True):
            vals = {k:a[k]+p*(b[k]-a[k]) for k in a}
            vals.update(requests=1, anchors=int(c.anchor), approved=p, vetoed=int(c.anchor)-p,
                        retained_direct_edits=p*c.hit, paid_admissions=p*c.paid)
            vals.update({name:p*int((c.hit,c.paid)==cell) for cell,name in CELLS.items()})
            vals.update({name.replace('swaps_', 'vetoes_'):(int(c.anchor)-p)*int((c.hit,c.paid)==cell)
                         for cell,name in CELLS.items()})
            choice = ('base' if not c.anchor else 'expected_mixture' if mode==POLICIES[3] else 'rank' if p else 'base')
            requests.append(dict(index=c.index, dataset=c.dataset, policy=mode, endpoint_source=choice,
                                 victim_risk=c.risk, rank_direct_strict=int(c.hit) if c.anchor else None,
                                 base_victim_admit=int(c.paid) if c.anchor else None, **vals))
            for d in ('all', c.dataset):
                totals.setdefault((mode,d),Counter()).update(vals)
    summary = []
    for (mode,d), sums in totals.items():
        vals = dict(sums)
        vals['macro_tpf'] = vals.pop('tpf') / vals['requests']
        eff = Fraction(vals['N'], vals['physical_Q'])
        expectation = mode==POLICIES[3]
        vals.update(pooled_efficiency=None if expectation else eff,
                    ratio_of_expected_N_to_expected_physical_Q=eff if expectation else None)
        summary.append(dict(policy=mode,dataset=d,**vals))
    return requests,summary


def numeric(rows):
    return [{k:(v.numerator if v.denominator==1 else float(v)) if isinstance(v,Fraction) else v
             for k,v in row.items()} for row in rows]


def run(fit_path=FIT, out=OUT):
    from bdlm_serving.methods.t2t.admission_gate import admission_decision, read_admission_fit

    audit.check(not out.resolve().is_relative_to(ROOT), 'Choose a private output directory outside the checkout')
    digest = sha(fit_path)
    fit = read_admission_fit(fit_path)
    requests, summary = evaluate(list(cases(fit,admission_decision)))
    audit.check(sha(fit_path)==digest, 'Admission fit changed during evaluation')
    out.mkdir(mode=0o700,parents=True,exist_ok=True)
    table(out / 'requests.csv', numeric(requests))
    table(out / 'summary.csv', numeric([dict(r, victim_fit_sha256=digest) for r in summary]))
    for r in numeric(summary):
        if r['dataset']=='all':
            print(f"{r['policy']}: {r['correct']:g}/30, TPF={r['macro_tpf']:.5f}, swaps={r['approved']:g}, "
                  f"strict={r['retained_direct_edits']:g}, paid={r['paid_admissions']:g}")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,default=FIT)
    parser.add_argument('--out',type=Path,default=OUT)
    args = parser.parse_args()
    os.umask(0o077)
    run(args.fit,args.out)
