"""Read-only observational diagnosis of saved R4E first-live reviews.
No fit, replay, new labels, threshold selection, or causal attribution.
Counts are generation events, not independent observations or sampling weights.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv, gzip, hashlib, io, json, math, subprocess

ROOT = Path('/home/yuqing/ctaNew')
PIN = '141e820e4b736c6a8c4f27b0a0dbf53789e95e89'
DATA = ROOT/'data/predict_fun/research/adv2_entry_smooth_v1_20261121'
OUT = Path(__file__).resolve().parent
COLS = ('fair_edge', 'o_edge', 'fp_queue', 'fp_remaining', 'pf_mid', 'pf_spread', 'adv_pf_imb1', 't_to_end')
GROUPS = dict(book=('pf_mid','pf_spread','adv_pf_imb1'), recorded_market=('pf_mid','pf_spread','adv_pf_imb1','fair_edge'),
              all_recorded_except_clock=COLS[:-1])

def sha(b): return hashlib.sha256(b).hexdigest()
def git_show(path): return subprocess.check_output(['git','-C',str(ROOT),'show',PIN+':'+path])
def classification(row, cols):
    vals=[row.get('d_'+c) for c in cols]
    if any(v is None or not math.isfinite(v) for v in vals): return 'unknown'
    return 'changed' if any(v != 0 for v in vals) else 'unchanged'
def first_policy_live(trace, fallbacks):
    fb={(float(x[0]),int(x[1]),x[2]) for x in fallbacks}
    first={}
    for x in trace:
        if x[2]=='live' and x[4] is not None and (float(x[0]),int(x[1]),x[2]) not in fb:
            first.setdefault(int(x[4]),x)
    return first
def synthetic():
    r={'d_'+c:0. for c in COLS}
    assert classification(r,GROUPS['recorded_market'])=='unchanged'
    for c in GROUPS['recorded_market']:
        bad=dict(r); bad['d_'+c]=.01
        assert classification(bad,GROUPS['recorded_market'])=='changed'
    for v in (None,float('nan')):
        bad=dict(r);bad['d_pf_mid']=v
        assert classification(bad,GROUPS['recorded_market'])=='unknown'
    # Changed own queue alone is not relabelled as changed market proxies.
    bad=dict(r);bad['d_fp_queue']=10
    assert classification(bad,GROUPS['recorded_market'])=='unchanged'
    assert classification(bad,GROUPS['all_recorded_except_clock'])=='changed'
    tr=[[100.,0,'live','KEEP',1],[150.,0,'live','PULL',1]]
    assert first_policy_live(tr,[[100.,0,'live','feature_unavailable']])[1]==tr[1]
    assert first_policy_live(tr,[])[1]!=tr[1]
    return True

def run():
    manifest=git_show('predictfun/reviews/r4e_entry_smooth_20261121/checkpoint/outputs.sha256')
    hashes=dict((line.split()[1],line.split()[0]) for line in manifest.decode().splitlines())
    seen=[]; statuses=Counter(); parts=[]; by=defaultdict(list)
    for k in range(12):
        rp=f'replay_v1/parts/part_{k:03d}.json'; bp=f'replay_v1/parts/part_{k:03d}.jsonl.gz'
        rb=(DATA/rp).read_bytes(); b=(DATA/bp).read_bytes()
        assert sha(rb)==hashes[rp] and sha(b)==hashes[bp]
        rec=json.loads(rb); assert rec['part']==k and rec['sha256']==sha(b)
        rows=[json.loads(line) for line in gzip.decompress(b).splitlines()]
        assert [(r['m'],r['t_ms']) for r in rows]==[(r['m'],r['t_ms']) for r in rec['members']]
        parts.append(dict(part=k,record_sha256=sha(rb),data_sha256=sha(b)))
        for w in rows:
            statuses[w['status']]+=1
            if w['status']!='ok': continue
            for policy,e in w['episodes'].items():
                trace=e['trace']['decision_list']
                posts={int(x[4]):(float(x[0]),int(x[1])) for x in trace if x[2]=='absent' and x[3]=='POST'}
                first=first_policy_live(trace,e['trace']['fallback_list'])
                assert len(e['churn'])==e['reviewed_generations']==len(first)
                assert sum(r['action']=='PULL' for r in e['churn'])==e['first_review_pulls']
                for r in e['churn']:
                    assert (r['t_post'],r['side'])==posts[r['gid']]
                    assert (r['t_review'],r['side'],r['action'])==(float(first[r['gid']][0]),int(first[r['gid']][1]),first[r['gid']][3]), (w['day'],w['m'],w['t_ms'],policy,r,first[r['gid']])
                    assert r['dt']==r['t_review']-r['t_post']
                    x=dict(day=w['day'],m=w['m'],t_ms=w['t_ms'],policy=policy,**r)
                    seen.append(x)
                    for side in ('all',str(r['side'])): by[(w['day'],policy,r['action'],side)].append(x)
    csv_rel='replay_v1/churn.csv'; cb=(DATA/csv_rel).read_bytes()
    assert cb==gzip.decompress(git_show('predictfun/reviews/r4e_entry_smooth_20261121/replay/churn.csv.gz'))
    saved=list(csv.DictReader(io.StringIO(cb.decode())))
    def key(r):return (str(r['day']),int(r['m']),int(r['t_ms']),r['policy'],int(r['gid']))
    raw={key(r):r for r in seen};assert len(raw)==len(seen)==len(saved)
    for r in saved:
        x=raw[key(r)]
        for c,v in x.items():
            if c not in r: continue
            if v is None: assert r[c]=='',(key(r),c)
            elif isinstance(v,(int,float)): assert float(r[c])==v,(key(r),c)
            else: assert r[c]==str(v),(key(r),c)
    summaries=[]
    for (day,policy,action,side),g in sorted(by.items()):
        z=dict(day=day,policy=policy,action=action,side=side,n=len(g),worlds=len({(r['m'],r['t_ms']) for r in g}),
               status=dict(Counter(r['status_review'] for r in g)),dt_ms=dict(Counter(str(r['dt']) for r in g)))
        for name,cols in GROUPS.items():z[name]=dict(Counter(classification(r,cols) for r in g))
        z['fields']={c:dict(Counter('unknown' if r.get('d_'+c) is None or not math.isfinite(r['d_'+c]) else
                     'up' if r['d_'+c]>0 else 'down' if r['d_'+c]<0 else 'unchanged' for r in g)) for c in COLS}
        z['same_mid_and_spread']=sum(classification(r,('pf_mid','pf_spread'))=='unchanged' for r in g)
        z['same_mid_and_spread_share']=z['same_mid_and_spread']/len(g)
        z['imbalance_changed_share']=sum(z['fields']['adv_pf_imb1'].get(d,0) for d in ('up','down'))/len(g)
        z['fair_edge_increased_share']=z['fields']['fair_edge'].get('up',0)/len(g)
        z['fair_edge_increased_in_majority']=z['fair_edge_increased_share']>.5
        z['same_book_fair_not_down_queue_not_up']=sum(classification(r,GROUPS['book'])=='unchanged' and r.get('d_fair_edge') is not None and
               r['d_fair_edge']>=0 and r.get('d_fp_queue') is not None and r['d_fp_queue']<=0 for r in g)
        z['policy_handoffs']=dict(Counter(str(r['by_post'])+' -> '+str(r['by_review']) for r in g))
        summaries.append(z)
    result=dict(pinned_source=PIN,asof=datetime.now(timezone.utc).isoformat(),input_manifest_sha256=sha(manifest),csv_sha256=sha(cb),
                selftest=synthetic(),rows=len(seen),world_statuses=dict(statuses),parts=parts,all_saved_churn_fields_equal=True,
                summaries=summaries,limitations=['Eight retained fields only; full market/order-flow inputs absent.',
                'Exact input changes do not measure materiality, forecast correctness, or causal contributions.',
                'All counts are unweighted generation events, repeated within worlds; two consumed days, no intervals.',
                'POST advantage and retention value are different targets; their difference is not a forecast revision.'])
    (OUT/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','parts')},sort_keys=True))
    for r in summaries:
        if r['side']=='all' and r['action']=='PULL':
            print(json.dumps({k:r[k] for k in ('day','policy','n','worlds','status','dt_ms','book','recorded_market','all_recorded_except_clock','same_mid_and_spread','same_book_fair_not_down_queue_not_up','policy_handoffs')}))

if __name__=='__main__':run()
