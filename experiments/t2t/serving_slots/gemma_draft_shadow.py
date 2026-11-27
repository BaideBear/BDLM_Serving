"""Read-only comparison on the real verification trajectory. Retain no logits."""
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
import json
from hashlib import sha256


def compare_block(draft, verified, eos):
    if len(draft)!=256 or len(verified)!=256:
        raise ValueError('Expected two complete physical canvases')
    def end(ids):
        return next((j for j,t in enumerate(ids) if t in eos),256)
    a,b=end(draft),end(verified)
    changed=[j for j in range(256) if draft[j]!=verified[j]]
    return dict(changed=len(changed),first_change=next(iter(changed),None),
                draft_eos=a,verify_eos=b,delivered_equal=a==b and draft[:min(a+1,256)]==verified[:min(b+1,256)])


class DraftShadow:
    def __init__(self, observer, eos):
        self.observer,self.eos=observer,set(eos)
        self.native=observer.criterion
        self.last=self.draft=self.verified=None
        self.blocks=[]

    def reset(self):
        self.native.reset()

    def __call__(self, draft, scores):
        result=self.native(draft,scores)
        self.last=self.observer.last
        self.draft=draft[0].tolist()
        return result

    @contextmanager
    def install(self,model):
        self.observer.criterion=self
        try:
            def hook(_m,_args,output):
                self.verified=output.logits[0].argmax(-1).tolist()
            with model.register_forward_hook(hook):
                yield
        finally:
            self.observer.criterion=self.native

    def after(self,e,origin):
        if e['route']=='dense_single':
            self.last_latched=e['fullstop']['latched']
            return
        ordinal=e['ordinal']
        if (e['route'] not in ('verify','verify_paid') or self.last is None
                or (self.last.tick,self.last.ordinal,self.last.epoch)!=(e['tick']-1,ordinal,origin+256*ordinal)
                or self.verified is None or len(e['parts'])!=1 or e['sampled_absolute']
                or e['parts'][0]['rows']!=list(range(256)) or ordinal!=len(self.blocks)):
            raise ValueError('Terminal comparison requires the preceding measured full draft and ordered verify')
        self.blocks.append(dict(ordinal=ordinal,latched=self.last_latched,encoded=e['encoded'],
            dirty_paid=len(e['dirty_before_absolute']),**compare_block(self.draft,self.verified,self.eos)))


def summarize(root, references, manifest):
    tasks=manifest['samples']
    if {t['index'] for t in tasks}!=set(range(90)):
        raise ValueError('Wrong manifest')
    out={}
    for name,indices in (('original30',range(30)),('new60',range(30,90)),('frozen90',range(90))):
        counts=Counter()
        for i in indices:
            path=root/f'fixed256-{i:02}.json'
            if not path.exists():
                continue
            r=json.loads(path.read_bytes())
            blob=(references/f'C256-dense-stop1-{i:02}.json').read_bytes()
            ref=json.loads(blob)
            blocks=r['draft_shadow']
            if (r['profile']!='dense_b_stop_fused' or r['identical'] is not True or r['suite']!='fixed256'
                    or r['reference_sha256']!=sha256(blob).hexdigest() or ref['fullstop'] is not True
                    or r['samples_sha256']!=manifest['samples_sha256'] or r['index']!=i
                    or any(r[k]!=tasks[i][k] for k in ('index','task_id','dataset','seed'))
                    or any(r[k]!=ref[k] for k in ('input_sha256','decoder_Q','encoder_Q','F'))
                    or r['output_digest']!=sha256(json.dumps(ref['output_ids'],separators=(',',':')).encode()).hexdigest()
                    or len(blocks)!=r['encoder_F']+1 or len(blocks)!=ref['fused_F']
                    or sum(b['dirty_paid'] for b in blocks)!=ref['fused_dirty_Q']):
                raise ValueError(f'Unmatched control or incomplete terminal writes {i}')
            counts['requests']+=1
            counts['requests_with_delivery_difference']+=any(not b['delivered_equal'] for b in blocks)
            counts['requests_with_encoder_difference']+=any(b['encoded'] and b['changed'] for b in blocks)
            for j,b in enumerate(blocks):
                if (b['ordinal']!=j or b['encoded']!=(j<len(blocks)-1) or b['dirty_paid']<=0
                        or not 0<=b['changed']<=256 or (b['changed']==0)!=(b['first_change'] is None)
                        or not b['changed'] and not b['delivered_equal']):
                    raise ValueError(f'Broken canvas ledger {i}:{j}')
                counts['blocks']+=1
                counts['latched_blocks']+=b['latched']
                counts['different_blocks']+=bool(b['changed'])
                counts['different_tokens']+=b['changed']
                counts['different_first_eos']+=b['draft_eos']!=b['verify_eos']
                counts['different_deliveries']+=not b['delivered_equal']
                counts['different_latched_deliveries']+=b['latched'] and not b['delivered_equal']
                counts['different_encoder_inputs']+=bool(b['encoded'] and b['changed'])
                counts['terminal_writes']+=b['dirty_paid']
        out[name]={'expected':len(indices),'observed':counts['requests']}
        if counts['requests']==len(indices):
            out[name]['counts']=dict(counts)
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('receipts','references','manifest'):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    print(json.dumps(summarize(a.receipts,a.references,json.loads(a.manifest.read_bytes())),indent=2,sort_keys=True))

