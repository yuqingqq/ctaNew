"""Resolve post-hoc intent ambiguity with two-market, behavior-identical replays."""
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import pandas as pd
from . import pf_inflight_engine as engine
from .pf_inflight_study import HERE, DATA, DEST, ARMS, sha, write
from .pf_sizing_run import COMMON, stem
from .pf_inflight_run import location

MIDS=[3814150,3814579]
DAY='20261026'


def folder(tm):
    return DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_inflight_tie_{tm}'/DAY


def replay(tm):
    if tm not in ['J','X'] or folder(tm).exists():raise ValueError('invalid/overwrite trace')
    s=engine.source()
    s=engine.parent.replace(s,'for mid in ORDER:',f'for mid in ORDER:\n    if mid not in {MIDS!r}: continue')
    s=engine.parent.replace(s,"            requests.append((t, side, 'place', lane))",
        "            ENTRY_AUDIT.append(dict(kind='send',variant=ENTRY_KIND,mid=int(mid),t_ms=float(t),side=side,lane=lane,pt=p,size=float(zsz),known_net=float(knet)))\n            requests.append((t, side, 'place', lane))")
    for k in list(os.environ):
        if k.startswith('PF_'):del os.environ[k]
    os.environ.update(COMMON,PF_FORWARD='20261104',PF_PREDS='preds_basis_study',PF_TAG=f'inflight_tie_{tm}',
        PF_MAX_WINDOWS='0',PF_ARMS=';'.join(ARMS[k] for k in ['reduce_reserved','both_reserved']),
        PF_TXJIT='frame' if tm=='J' else '',PF_TXSHIFT='0' if tm=='J' else '-50',PF_QACT='' if tm=='J' else 'max')
    engine.source=lambda:s
    sys.argv=[sys.argv[0],DAY,'tokyoreal','custom'];engine.main()


def resolve(records,mid,variant,side,lane,pt,size,t_ms):
    matches=[r for r in records if r.get('kind')=='send' and r['mid']==mid and r['variant']==variant
        and r['side']==side and r['lane']==lane and r['pt']==pt and abs(r['size']-size)<1e-7
        and round(r['t_ms']*4096)==round(t_ms*4096)]
    states={r['known_net'] for r in matches}
    if len(states)!=1:raise ValueError('missing/ambiguous placement identity')
    return states.pop()


def main():
    toy=[dict(kind='send',mid=1,variant='v',side=1,lane=1,pt=50,size=10,t_ms=200,known_net=0.)]
    assert resolve(toy,1,'v',1,1,50,10,200)==0
    try:resolve(toy,1,'v',1,1,50,10,201)
    except ValueError:pass
    else:raise AssertionError('wrong event accepted')
    old=json.loads((DEST/'overshoot_diagnostic.json').read_text())
    examples=[x for x in old['examples'] if x['variant'] in ['reduce_reserved','both_reserved']]
    if {x['day'] for x in examples}!={DAY} or {x['mid'] for x in examples}!=set(MIDS):raise ValueError('undeclared trace scope')
    resolved=[];evidence={}
    for tm in ['J','X']:
        subprocess.run([sys.executable,'-m',__package__+'.pf_inflight_ties','replay',tm],cwd=HERE.parents[1],check=True)
        out=folder(tm)
        for label in ['reduce_reserved','both_reserved']:
            for kind in ['win','fills','requests']:
                a=pd.read_parquet(out/kind/stem(ARMS[label]));b=pd.read_parquet(location(label,DAY,tm)/kind/stem(ARMS[label]))
                pd.testing.assert_frame_equal(a,b[b.mid.isin(MIDS)].reset_index(drop=True),check_exact=True)
        records=json.loads((out/'entry_audit.json').read_text())['examples']
        evidence[tm]=dict(trace_sha=sha(out/'entry_audit.json'),summary_sha=sha(out/'summary.json'),exact_parity=['win','fills','requests'],markets=MIDS)
        for ex in [x for x in examples if x['timing']==tm]:
            f=pd.read_parquet(out/'fills'/stem(ARMS[ex['variant']]));w=pd.read_parquet(out/'win'/stem(ARMS[ex['variant']]))
            hit=f[(f.mid==ex['mid'])&(f.side==ex['side'])&((f.t-ex['t']).abs()<1e-9)&((f.q-ex['q']).abs()<1e-9)]
            if len(hit)!=1:raise ValueError('ambiguous tranche identity')
            r=hit.iloc[0];ws=float(w.loc[w.mid==ex['mid'],'ws'].iloc[0]);sent=ws+r.t*1000-r.age_ms-40
            known=resolve(records,ex['mid'],ex['variant'],int(r.side),int(r.lane),int(r.pt),float(r.osz),sent)
            resolved.append(dict(ex,exact_known_net=known,reducing_at_send=bool(known*(1 if r.side==0 else -1)<-1e-7)))
    totals=copy.deepcopy(old['totals'])
    for row in totals:
        if row['variant'] not in ['reduce_reserved','both_reserved']:continue
        ex=[x for x in resolved if x['timing']==row['timing'] and x['variant']==row['variant']]
        if row['tranches_lower']!=0 or len(ex)!=row['tranches_upper'] or abs(sum(x['opening'] for x in ex)-row['opening_shares_upper'])>1e-7:
            raise ValueError('not all uncertain opening volume resolved')
        actual=[x for x in ex if x['reducing_at_send']]
        row.update(opening_shares_lower=sum(x['opening'] for x in actual),opening_shares_upper=sum(x['opening'] for x in actual),
            tranches_lower=len(actual),tranches_upper=len(actual),windows_lower=len({x['mid'] for x in actual}),windows_upper=len({x['mid'] for x in actual}))
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),kind='superseding_mechanism_audit',
        supersedes_sha=sha(DEST/'overshoot_diagnostic.json'),main_receipt_sha=sha(DEST/'report.json'),source_sha=sha(__file__),
        totals=totals,resolved=resolved,evidence=evidence,controls=dict(exact_lookup_positive=True,wrong_event_refused=True),
        interpretation='same population as v1; ambiguous reducing/opening classification resolved at exact engine send events; replay artifacts match original exactly')
    write(DEST/'overshoot_diagnostic_v2.json',result);write(HERE.parent/'reviews/inflight_budget_20261109/overshoot_diagnostic_v2.json',result)
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='replay':replay(sys.argv[2])
    else:main()
