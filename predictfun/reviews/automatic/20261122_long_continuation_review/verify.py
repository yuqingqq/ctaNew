"""Pinned-source and saved-artifact review only; synthetic tapes, no historical replay."""
from pathlib import Path
import collections, copy, gzip, hashlib, importlib.util, json, math, resource, subprocess, sys
import numpy as np
import pandas as pd
resource.setrlimit(resource.RLIMIT_CPU, (85, 90))
ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-lc-review-6b806c19'); OUT=Path(__file__).resolve().parent
TARGET='6b806c192980de106da80fa97ce4993d6553ec66'; REL='predictfun/reviews/ladder_long_continuation_20261122'
sys.path.insert(0,str(PIN/'predictfun/research'))
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def blob(ref,p):return subprocess.check_output(['git','-C',str(PIN),'show',f'{ref}:{p}'])
def sha(b):return hashlib.sha256(b).hexdigest()
assert subprocess.check_output(['git','-C',str(PIN),'rev-parse','HEAD'],text=True).strip()==TARGET
import pf_ladder_long_continuation as X
import pf_ladder_long_continuation_selftest as ST
import pf_ladder_long_continuation_mutants as MH
import pf_mutant_one as MO
import pf_mutant_summary as MS
ST.main(X)
ld=PIN/REL/'logs/mutants_try2'
records={'control':MS.last_record(ld/'llc2_control.log')}
records.update({i:MS.last_record(ld/f'llc2_{i}.log') for i in range(len(MH.MUTANTS))})
ok,bad=MS.judge(len(MH.MUTANTS),records,MO.sources(MH),[m[0] for m in MH.MUTANTS]);assert ok,bad
C=mod('review_lc_checks',PIN/REL/'lc_checks.py');C.REPO=PIN;C.R=PIN/REL
E=C.load();E['my']=C.my_pairs(E);base=C.run_checks(E);assert base['all_ok'],base
fz=C.falsify(E,base);assert fz['falsifier_ok'],fz
(OUT/'checks.json').write_text(json.dumps(base,indent=2)+'\n');(OUT/'falsifier.json').write_text(json.dumps(fz,indent=2)+'\n')
gm=E['gm'];assert X.sources()==gm['sources']==E['audit']['sources']
for p in C.COMMITTED+['predictfun/reviews/ladder_long_continuation_20261122/tools/rehearse_lc.py']:
    assert blob('7407bdf1',p)==blob(TARGET,p),p
assert sha(blob(TARGET,f'{X.AF.RECEIPT}/manifest.json'))==gm['af_manifest_sha256']
for day,want in gm['pilot_records'].items():
    b=(C.AF/'pilot'/f'{day}.json').read_bytes();r=json.loads(b);assert sha(b)==want
    assert sha((C.AF/'pilot'/f'{day}.jsonl.gz').read_bytes())==r['sha256']['gz']
    assert sha((C.AF/'pilot'/f'{day}_arrays.npz').read_bytes())==r['sha256']['npz']
for (part,day),r in E['days'].items():
    assert r['sources']==gm['sources'] and r['day']==day and r['part']==part
    assert sorted(r['markets'])==sorted(gm['pilot_markets'][day])
    # The saved records in data are the exact receipt copies.
    cp=PIN/REL/'outputs'/f'{day}.json'
    if cp.exists():assert cp.read_bytes()==(C.D/part/f'{day}.json').read_bytes()
for p in (PIN/REL/'outputs').glob('*.csv.gz'):
    assert gzip.decompress(p.read_bytes())==(C.D/'measure'/p.name[:-3]).read_bytes(),p
assert (PIN/REL/'outputs/audit.json').read_bytes()==(C.D/'measure/audit.json').read_bytes()

arms=pd.read_csv(C.D/'measure/arms.csv',dtype={'day':str},float_precision='round_trip')
pairs=pd.read_csv(C.D/'measure/pairs.csv',dtype={'day':str},float_precision='round_trip')
reps=pd.read_csv(C.D/'measure/replacements.csv',dtype={'day':str},float_precision='round_trip')
summary=pd.read_csv(C.D/'measure/summary.csv',dtype={'day':str,'side':str},float_precision='round_trip')
assert len(arms)==128 and not arms.duplicated(['key','arm']).any()
assert set(zip(arms.key,arms.arm))=={(s['key'],a) for s in gm['selected'] for a in ('retain','alternative')}
sel={s['key']:s for s in gm['selected']};pr=pairs.set_index('key');ar=arms.set_index(['key','arm'])
af={(d,m):C.af_record(d,m) for d,ms in gm['pilot_markets'].items() for m in ms}
needs=[];coverage={};price_constant=True;no_missing_queue=0;missing_mids=0
checked=0;replacement_changes=[];fifo_censored=[]

def near(a,b):return bool(np.isclose(float(a),float(b),rtol=1e-10,atol=1e-7,equal_nan=True))
for x in C.pairs_led(E):
    s=sel[x['key']];k=x['key'];t=s['t'];reference=af[(s['day'],s['m'])];end=reference['end']
    assert x['kind']==s['kind'] and x['day']==s['day'] and x['m']==s['m']
    for name,action in zip(('retain','alternative'),X.ACTIONS[s['kind']]):
        a=x[name];row=ar.loc[(k,name)];checked+=1
        assert len(a['applied'])==1
        ap=a['applied'][0]
        assert ap[:4]==(['live',t,s['gid'],action] if s['kind']=='REVIEW' else ['slot',t,s['rung'],action]),(k,name,ap)
        assert np.isfinite(a['mark']) and np.isfinite(a['M_fork']) and a['mark']==reference['mark']
        assert a['pre_fills']==[f for f in reference['fills'] if f[0]<=t]
        assert all(t<f[0]<=end for f in a['post_raw'])
        fill=a['pre_fills']+a['post_raw'];by=collections.defaultdict(set)
        for f in fill:by[f[1]].add((f[2],f[4]))
        assert all(len(v)==1 for v in by.values()),(k,name,'same gid changed side/price')
        val=sum(C.cval(f,a['mark']) for f in fill);assert near(val,a['value'])
        mine=C.my_arm(a);assert near(mine['Q'],row.Q) and near(mine['focal'],row.focal_c) and near(mine['other'],row.other_c)
        assert near(row.edge+row.movement,row.Q) and row.edge_missing==0
        missing_mids+=int(row.edge_missing)
        assert mine['seq']==row.seq and near(row.early,X.sequence_axis(a['post_raw'],set(a['lineage']),a['focal'],a['mark'])[1])
        # Inventory and FIFO quantities use original saved fills, not exported scalars.
        inv=mine['I0'];peak=abs(inv);tw=0.;last=t
        for f in sorted(a['post_raw'],key=lambda f:f[0]):
            tw+=abs(inv)*(f[0]-last);inv+=(1 if f[2]==0 else -1)*f[3];peak=max(peak,abs(inv));last=f[0]
        tw+=abs(inv)*(end-last)
        assert near(row.tw_abs_I,tw/(end-t)) and near(row.peak_abs_I,peak) and near(row.terminal_I,inv)
        assert near(row.buy_q,sum(f[3] for f in a['post_raw'] if f[2]==0)) and near(row.sell_q,sum(f[3] for f in a['post_raw'] if f[2]==1))
        matched,hold,openlots=X.fifo([(f[0],f[2],f[3]) for f in sorted(a['post_raw'],key=lambda f:f[0])],mine['I0'],t)
        assert near(matched,row.matched) and near(hold,row.hold_ms) and near(openlots,row.open_lots)
        fifo_censored.append(abs(mine['I0'])>1e-9)
        assert a['suppressed']==len(a['suppressed_log'])==row.suppressed
        assert all(g in a['lineage'] and tt>=t for tt,g in a['suppressed_log'])
        assert all(g not in a['lineage'] for _,g in a['pp_log'])
        assert len(a['pp_log'])==row.pressure_pulls
        assert all(r['side']==s['side'] and r['submit']>=t and r['delay']==r['submit']-t for r in a['replacements'])
        for r in a['replacements']:
            if not r['refused']:
                assert r['ahead0'] is not None,(k,name,'missing activation queue')
                no_missing_queue+=1;assert near(r['ahead0'],C.shown(r['book'],r['side'],r['price']))
        # Branch-local gid reuse after WAIT cannot suppress the first replacement.
        if s['kind']=='ADMISSION' and name=='alternative':
            first=next(iter(a['replacements']),None)
            correct=first['delay'] if first else np.nan
            if not near(row.first_rep_delay,correct):
                replacement_changes.append(dict(key=k,day=s['day'],m=s['m'],kind=s['kind'],arm=name,stored=float(row.first_rep_delay),correct=correct,
                                                factual_gid=s['gid'],actual_first_gid=first['gid']))
        assert near(row.first_rep_delay,next((r['delay'] for r in a['replacements'] if r['gid']!=s['gid']),np.nan))
        rep_table=reps[(reps.key==k)&(reps.arm==name)].drop(columns=['day','m','key','arm']).to_dict('records')
        assert len(rep_table)==len(a['replacements'])
        for r0,r1 in zip(rep_table,a['replacements']):
            for col in r1:
                if col=='book':assert json.loads(r0[col])==r1[col]
                elif r1[col] is None:assert pd.isna(r0[col])
                else:assert near(r0[col],r1[col]),(k,col)
    assert x['retain']['M_fork']==x['alternative']['M_fork']
    assert near(pr.loc[k,'Y'],x['retain']['value']-x['alternative']['value'])

# Full summary fields and exact groups, supplementing the narrower checker.
new,rd=X.summarize(pairs,arms)
assert len(summary)==len(new)
for col in new:
    if col in ('kind','day','side'):assert summary[col].astype(str).tolist()==new[col].astype(str).tolist()
    else:assert np.allclose(summary[col],new[col],rtol=1e-10,atol=1e-7,equal_nan=True),col

# Three small known-bad controls show current checker limits. No data edited.
for name,col in [('wrong_callback_action','action'),('wrong_callback_time','time')]:
    e=dict(E);e['led']=copy.deepcopy(E['led']);p=next(x for x in e['led'] if x['kind']!='BASE')
    p['alternative']['applied'][0][3 if col=='action' else 1]='KEEP' if col=='action' else -999
    coverage[name]=C.fam_G(e)[0]['G_gates']
e=dict(E);e['sum']=E['sum'].copy();e['sum'].loc[0,'retain_edge']+=10000
coverage['changed_reported_execution_edge']=C.fam_S(e)[0]['S_summary']

# A fixed-price order can have an early adverse fill and a profitable opposite
# close, while its END-signed own-order sequence cannot turn positive.
hand=[(100.,7,0,10.,50.,1),(1000.,7,0,10.,50.,2),(2000.,8,1,20.,51.,3)]
seq=X.sequence_axis(hand,{7},7,49.)
assert seq==('early_loss_no_later_benefit',-10.,-10.) and sum(C.cval(f,49.) for f in hand)==20.
plate={}
for kind,g in pairs.groupby('kind'):
    plate[kind]=dict(pairs=len(g),by_day={d:float(h.Y.sum()) for d,h in g.groupby('day')},
                    Y=float(g.Y.sum()),abs_Y=float(g.Y.abs().sum()),abs_focal=float(g.Y_focal.abs().sum()),abs_other=float(g.Y_other.abs().sum()),
                    other_abs_larger=int((g.Y_other.abs()>g.Y_focal.abs()+1e-6).sum()))
old_alt=arms[arms.arm=='alternative'].first_rep_delay.copy()
fixed_alt=arms[arms.arm=='alternative'].copy()
for r in replacement_changes:fixed_alt.loc[fixed_alt.key==r['key'],'first_rep_delay']=r['correct']
ev=dict(target=TARGET,selftest='6 groups PASS',saved_mutants_rejudged=len(MH.MUTANTS),checks=base['all_ok'],falsifiers=len(fz['cases']),
        census=len(E['census']),pairs=len(pairs),arms_checked=checked,base_markets=len(af),finite_common_marks=True,
        queue_checks_nonrefused=no_missing_queue,missing_activation_queue=0,carried_inventory_arms=int(sum(fifo_censored)),
        summary_full_field_reproduction=True,kind_readings=plate,replacement_delay_changes=replacement_changes,
        delay_alternative_mean_ms_before=float(old_alt.mean()),delay_alternative_mean_ms_after=float(fixed_alt.first_rep_delay.mean()),
        checker_corruptions_still_pass=coverage,
        fixed_price_sequence_example=dict(sequence=seq,full_book_END_value=20.,meaning='Opposite-side capture is not same-side END-sign reversal'),
        resources=base['details']['I'],limitations=['no historical replay','arm orders/commands/fill-time mids not retained in paired ledgers; ancillary recomputation limited to saved facts',
                                                 'same-price sequence example synthetic, not evidence of realized queue necessity'])
(OUT/'evidence.json').write_text(json.dumps(ev,indent=2,default=str)+'\n')
pd.DataFrame(replacement_changes).to_csv(OUT/'replacement_delay_corrections.csv',index=False)
print(json.dumps(ev,indent=2,default=str))
