"""Exact named-arm supplement to the component-level B4 parity receipt."""
import hashlib
import json
import subprocess
import sys
from . import pf_pairs_validation as parent

s=parent.s
DEST=parent.DEST.parent/'paired_v3_arm'
REVIEW=parent.REVIEW.parent/'paired_v3_arm'
SOURCES=['pf_pairs_v3_arm.py','tests/test_ladder_pairs_v3_arm.py']

def declare():
    parent.check();DEST.mkdir(exist_ok=True)
    d=dict(as_of=s.now(),source_commit=s.command(['git','rev-parse','HEAD'],cwd=s.a.HERE),sources={p:s.sha(s.a.HERE/p) for p in SOURCES},
        parent_receipt_sha256=s.sha(parent.DEST/'receipt.json'),parent_declaration_sha256=s.sha(parent.DEST/'declaration.json'),
        scope='Exact MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240|mid|AM:5:25:95 on full300s wholly synthetic inputs. No-token frozen parity, fixed-size no-fill parity, variable resize positive, CAP10 and RO60 controls, native request/pair trace replay.',
        population='Six exact-v3 traces (two sides times no_trade/reduce_fill/cap_fill) plus four fixed-size controls;16 explicit Python checks. No market-data reads or exclusions.',
        correction='The earlier component resize controls used AM sizing, PM90 and ladder rules but JOIN policy and a1s stop, not the complete named v3 arm. They remain component evidence. This supplement checks the complete arm; no blanket fixed-size rejection parity claim is added.',
        limitations=['Synthetic function outputs only; no historical parquet-output parity or calibrated fill/PnL result.','Existing LZ rejection-accounting correction intentionally differs from frozen fixed-size rejection-stress paths.'],
        no_market_data_read=True,no_protected_dates_read=True,no_live_change=True)
    s.write(DEST/'declaration.json',d);s.write(REVIEW/'declaration.json',d)

def check():
    parent.check();d=json.loads((DEST/'declaration.json').read_text())
    for p,h in d['sources'].items():
        if s.sha(s.a.HERE/p)!=h:raise ValueError('exact-arm source changed')
    if s.sha(parent.DEST/'receipt.json')!=d['parent_receipt_sha256']:raise ValueError('parent receipt changed')
    return d

def controls():
    check();subprocess.run([sys.executable,'-m','pytest','-q',str(s.a.HERE/'tests/test_ladder_pairs_v3_arm.py'),'--tb=short'],check=True)

def run_cases():
    check()
    from .tests.test_ladder_pairs_v3_arm import exact_case
    rows=[];traces=[]
    def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
    for side in [0,1]:
        for kind in ['no_trade','reduce_fill','cap_fill']:
            paired,trace=exact_case(side,kind);generated,_=exact_case(side,kind,paired=False);frozen,_=exact_case(side,kind,paired=False,frozen=True)
            a,b=digest(generated),digest(frozen)
            if a!=b:raise ValueError('exact v3 no-token outputs differ')
            positive=paired['live_ctr'].get('lz_resize',0)>0 and paired['requests']!=generated['requests']
            if not positive:raise ValueError('exact v3 resize control did not fire')
            rows.append(dict(side=side,kind=kind,generated_sha256=a,frozen_sha256=b,paired_sha256=digest(paired),no_token_byte_identical=a==b,resize_positive=positive,resize_count=paired['live_ctr']['lz_resize']))
            traces.append(dict(name=f'v3/{side}/{kind}',events=trace,minimum=1.,rpm=500.))
        for size in [5.,25.]:
            paired,_=exact_case(side,size=size);single,_=exact_case(side,paired=False,size=size)
            a,b=digest(paired),digest(single)
            if a!=b:raise ValueError('exact v3 fixed-size control differs')
            rows.append(dict(side=side,size=size,kind='fixed_no_trade',paired_sha256=a,single_sha256=b,byte_identical=a==b))
    s.write(DEST/'parity.json',rows)
    with (DEST/'synthetic.jsonl').open('x') as f:
        for t in traces:f.write(json.dumps(t,separators=(',',':'))+'\n')
    subprocess.run([str(s.TARGET/f'release/examples/{parent.NAME}'),str(DEST/'synthetic.jsonl'),str(DEST/'native.jsonl')],check=True)

if __name__=='__main__':
    {'declare':declare,'controls':controls,'run':run_cases}[sys.argv[1]]()
