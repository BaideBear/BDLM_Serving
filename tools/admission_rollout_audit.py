"""Audit the two frozen30 first-approved crowded rollouts, without a model or GPU."""
import argparse
from collections import Counter
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST, sha
from tools import context_gates_audit as cg
from tools.context_gates_audit import check, load, same
from tools.t2t_results import table


PREFIX = 't2t_q8_f0_m7_'
MODES = ('hard_admit_low', 'hard_admit_coin')
POLICIES = tuple(PREFIX + mode for mode in MODES)
BASELINE = 't2t_q8_f0_m8_hard_noedit'
ROOT = Path('../t2t-native-1125')
DECISION = 'admission_decision'
SIDE = {'intervention', 'beneficiary', 'crowded_screen', DECISION}
DIGESTS = {'trajectory_sha256', 'accepted_sha256'}


def without(obj, keys):
    return {k:v for k,v in obj.items() if k not in keys}


class AdmissionLedger(cg.CrowdedLedger):
    def __init__(self, mode, seed, fit, decide):
        # Both policies physically execute the ranked placement, never the coin's row.
        super().__init__(cg.ONCE[1], seed)
        self.mode, self.fit, self.decide = mode, fit, decide
        self.offers = []

    def select(self, e, eligible, picked, reserve):
        if DECISION in e:
            move = e['crowded_screen']
            keep, risk = self.decide(move, self.fit, self.mode, self.seed, e['block'], e['hard_clock'])
            check(type(keep) is bool and type(risk) is float and 0 <= risk <= 1 and
                  same(e[DECISION], dict(keep=keep,risk_probability=risk)), 'wrong cached decision or risk')
            self.offers.append(e['hard_clock'])
            if not keep:
                masks, fresh, base, _, _ = cg.replay_move(move,e,eligible,cg.POLICIES[2],self.seed)
                check(picked == base and not reserve and 'intervention' not in e,
                      'rejected offer did not defer on noedit')
                self.counts.update(screened=1, threshold_checks=1, screen_swap_mask=1)
                return masks, fresh
        return super().select(e,eligible,picked,reserve)


def audit_arm(raw, sample, policy, fit, decide):
    check(policy in POLICIES and raw['policy'] == policy, 'wrong admission policy')
    ledger = AdmissionLedger(policy.removeprefix(PREFIX),sample['seed'],fit,decide)
    for e in raw['events']:
        offer = 'crowded_screen' in e and e['crowded_screen']['reason'] == 'swap_mask'
        check((DECISION in e) == offer, 'missing or misplaced admission decision')
    # Reuse the verified targeted-seat wire contract. This is a shallow view:
    # all physical events, identities, outcomes and counters are untouched.
    c = cg.audit_request(dict(raw,policy=cg.ONCE[1]),sample,cg.ONCE[1],once=ledger)
    anchor = ledger.anchor[0] if ledger.anchor else None
    n = len(ledger.offers)
    check(c['swaps'] == int(anchor is not None) and ledger.counts['screen_swap_mask'] == n,
          'offer or actual swap accounting mismatch')
    counts = Counter({k:c[k] for k in ('requests','F','Q','N','reserved_hits','strict_edits','admissions',
                      'revocations','selected_fresh','fresh_overflow_rows',
                      *('followup_'+s for s in cg.FOLLOWUP),
                      *(s+'_'+metric for s in cg.STAGES for metric in ('F','Q')))})
    counts.update({k:v for k,v in ledger.counts.items() if k.startswith('screen') or k in
                   ('threshold_checks','anchor_admissions','anchor_revocations','post_anchor_selected')})
    counts.update(opportunities=n, skipped=n-int(anchor is not None), triggered=int(anchor is not None),
                  no_offer=int(n==0), never_approved=int(anchor is None),
                  first_approved=int(anchor is not None and n==1), delayed_approved=int(anchor is not None and n>1))
    return counts, anchor, ledger.offers[0] if n else None


def equal_endpoint(left,right):
    check(same(without(left,{'policy','events'}),without(right,{'policy','events'})),
          'identical physical paths have unequal output or work')


def baseline_parity(noedit, base, sample):
    check(noedit['policy']==BASELINE and base['policy']==cg.ONCE[0] and
          all(same(noedit[k],sample[k]) for k in ('index','dataset','task_id','seed')), 'wrong baseline identity')
    equal_endpoint(noedit,base)
    check(len(noedit['events']) == len(base['events']) and all(
        same(x, without(y,SIDE|DIGESTS)) for x,y in zip(noedit['events'],base['events'])),
        'archived base differs from original physical noedit')


def same_history(left, right, split=None, *, identical_observers=False):
    """Compare through the independently predicted first physical fork, or the whole path."""
    if split is None:
        check(len(left['events'])==len(right['events']), 'identical paths have unequal length')
        equal_endpoint(left,right)
    else:
        check(0 < split <= min(len(left['events']),len(right['events'])), 'missing common fork')
    for x,y in zip(left['events'],right['events']):
        if 'crowded_screen' in x and 'crowded_screen' in y:
            check(same(x['crowded_screen'],y['crowded_screen']), 'cached screen differs on shared input')
        if DECISION in x and DECISION in y:
            check(same(x[DECISION]['risk_probability'],y[DECISION]['risk_probability']),
                  'shared offer has different victim risk')
        if split is not None and x['hard_clock']==split:
            check(x['absolute_positions'] != y['absolute_positions'] and
                  all(same(x[k],y[k]) for k in cg.FORK_FIELDS), 'unexplained fork location, state or width')
            if 'intervention' in x and 'intervention' in y:
                check(all(same(x['intervention'][k],y['intervention'][k]) for k in
                          ('pre_sha256','post_rng_sha256')), 'fork prestate or RNG differs')
            return
        # Observers can legitimately start/stop at different clocks. Compare
        # shared evidence, including digests, on every physically identical step.
        ignored = {DECISION} if identical_observers else {DECISION}|((x.keys() ^ y.keys()) & SIDE)
        check(same(without(x,ignored),without(y,ignored)), 'unequal event on shared physical path')
    check(split is None, 'predicted fork was not reached')


def audit_references(noedit, hist, sample):
    anchors = [cg.audit_crowded_request(r,sample,p)[1] for r,p in zip(hist,cg.ONCE,strict=True)]
    cg.audit_crowded_match(hist,anchors)
    baseline_parity(noedit,hist[0],sample)
    return anchors[0][0] if anchors[0] else None


def compare_arm(raw, anchor, first, hist, historical_first):
    check(first == historical_first, 'first candidate differs from archived stop-on-first clock')
    base, rank, placebo = hist
    same_history(raw,base,anchor,identical_observers=anchor==first)
    if anchor is not None:
        e = raw['events'][anchor-1]
        old = base['events'][anchor-1]
        check([cg.B*e['block']+i for i in e['crowded_screen']['base']] == old['absolute_positions'],
              'anchor counterfactual Q differs from measured base')
    # If no candidate passes, the endpoint is measured base. If the first one
    # passes, it is measured rank (also measured random on a candidate collision).
    # A later fire has no corresponding stop-on-first endpoint.
    same_history(raw,rank,first if first != anchor else None,identical_observers=anchor==first)
    collision = first is not None and rank['events'][first-1]['absolute_positions'] == placebo['events'][first-1]['absolute_positions']
    random_same = first is None or (anchor==first and collision)
    same_history(raw,placebo,None if random_same else first,identical_observers=anchor==first)


def baseline_dir(d):
    m = load(d/'run.json')
    check(same([m[k] for k in ('model','batch_size','cap','block')]+[m.get('seed_offset',0)],
               ['llada21',1,2048,cg.B,0]), 'wrong baseline run configuration')
    check(m['policies'].count(BASELINE)==1 and len(m['policies'])==len(m['arms']) and
          same(m['arms'][m['policies'].index(BASELINE)],dict(budget=8,fresh=0,mask=8,selection='hard_noedit')),
          'wrong baseline spec')
    check(same(load(d/'qualified.json'),dict(model='llada21',full_width=32,matched=30)) and
          same(load(d/f'complete-{BASELINE}.json'),dict(policy=BASELINE,requests=30)) and
          {f.name for f in d.glob(f'[0-9]*-{BASELINE}.json')} == {f'{i:02d}-{BASELINE}.json' for i in range(30)},
          'incomplete or unqualified baseline')


def audit(runs, baseline, historical, fit_path, out):
    from bdlm_serving.methods.t2t.admission_gate import admission_decision, read_admission_fit

    arms = cg.discover(runs,POLICIES)
    refs = cg.discover(historical,cg.ONCE)
    baseline_dir(baseline)
    digest = sha(fit_path)
    fit = read_admission_fit(fit_path)
    check(sha(fit_path)==digest and all(load(d/'run.json')['admission_fit_sha256']==digest for d in runs),
          'run fit digest differs from supplied fit')
    samples = load(MANIFEST)['samples']
    check([s['index'] for s in samples]==list(range(30)), 'expected frozen30 manifest')
    rows, totals = [], {p:Counter() for p in POLICIES}
    for s in samples:
        i = s['index']
        def read(d,p):
            return load(d/f'{i:02d}-{p}.json')
        hist = [read(refs[p],p) for p in cg.ONCE]
        try:
            historical_first = audit_references(read(baseline,BASELINE),hist,s)
            raws, anchors = [], []
            for p in POLICIES:
                raw = read(arms[p],p)
                c, anchor, first = audit_arm(raw,s,p,fit,admission_decision)
                compare_arm(raw,anchor,first,hist,historical_first)
                rows.append(dict(index=i,policy=p,first_offer_clock=first,trigger_clock=anchor,**c))
                totals[p].update(c)
                raws.append(raw)
                anchors.append(anchor)
            split = min(a for a in anchors if a is not None) if anchors[0]!=anchors[1] else None
            same_history(*raws,split,identical_observers=True)
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError(f'admission audit request {i}: {exc}') from exc
    out.mkdir(parents=True,exist_ok=True)
    table(out/'admission-rollout-requests.csv',rows)
    table(out/'admission-rollout-summary.csv',[dict(policy=p,**c) for p,c in totals.items()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',type=Path,required=True,help='disjoint admission shard, repeatable')
    parser.add_argument('--fit','--admission-fit',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'hard-v1-llada21')
    parser.add_argument('--historical',type=Path,action='append',help='disjoint once shards (default: archived pair and random)')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    audit(args.run,args.baseline,args.historical or [ROOT/f'context-once-{part}-q8-v1' for part in ('pair','random')],
          args.fit,args.out)
