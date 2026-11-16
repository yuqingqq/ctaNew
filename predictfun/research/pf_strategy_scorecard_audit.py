"""Independent raw-book/outcome audit of the scorecard's exported fill ledger.

Descriptive source/age partitions are exported for all declared cases. This
does not choose a policy or modify the frozen accounting/replay pipeline.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_strategy_scorecard as S
np,pd,L,A=S.np,S.pd,S.L,S.A


def verify_up_index(value):
    L.require(isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_)) and int(value)==1,'Up token mapping changed')


def verify_values(f,times,mids,outcome,terminal):
    times=np.asarray(times,float);mids=np.asarray(mids,float)
    t=f.fill_ms.to_numpy();sign=1-2*f.side.to_numpy();p=f.price_cents.to_numpy()/100.;q=f.shares.to_numpy()
    ix=np.searchsorted(times,t,side='left')-1
    L.require((ix>=0).all(),'missing pre-fill source')
    pre=mids[ix];age=t-times[ix]
    L.require(np.allclose(f.pre_mid_cents,pre,atol=1e-9,rtol=0),'pre-fill mid provenance')
    L.require(np.allclose(f.pre_mid_age_ms,age,atol=1e-8,rtol=0),'pre-fill age provenance')
    expected=q*sign*(pre/100.-p)
    L.require(np.allclose(f.spread_capture_usd,expected,atol=1e-9,rtol=0),'spread decomposition from raw source')
    for h in (100,500,1000,5000):
        ix=np.searchsorted(times,t+h,side='right')-1
        L.require((ix>=0).all(),'missing mark source')
        expected=q*sign*(mids[ix]/100.-p)
        L.require(np.allclose(f[f'markout_{h}_usd'],expected,atol=1e-9,rtol=0),'markout from raw source')
        drift=q*sign*((mids[ix]-pre)/100.)
        L.require(np.allclose(f[f'mid_drift_{h}_usd'],drift,atol=1e-9,rtol=0),'midpoint movement from raw source')
    L.require(np.all(f.settlement_yes==outcome),'registry outcome identity')
    # Independent long-token accounting: buy Up at p, or buy Down at 1-p.
    token_cost=np.where(f.side.to_numpy()==0,p,1-p)
    token_payout=np.where(f.side.to_numpy()==0,outcome,1-outcome)
    L.require(np.allclose(f.settlement_pnl_usd,q*(token_payout-token_cost),atol=1e-9,rtol=0),'independent token settlement')
    ti=int(np.searchsorted(times,terminal,side='right'))-1
    L.require(ti>=0 and np.isfinite(mids[ti]),'missing terminal source')
    L.require(np.allclose(f.terminal_mid_cents,mids[ti],atol=1e-9,rtol=0),'terminal mid from raw source')
    L.require(np.allclose(f.terminal_pnl_usd,q*sign*(mids[ti]/100.-p),atol=1e-9,rtol=0),'terminal value from raw source')


def controls():
    verify_up_index(np.int64(1));mapping_refused=[]
    for value in (0,[1],True):
        caught=False
        try:verify_up_index(value)
        except ValueError:caught=True
        L.require(caught,'bad token mapping not refused');mapping_refused.append(repr(value))
    t=np.array([0.,10.,100.,500.,1000.,5000.]);m=np.array([50.,51.,49.,48.,52.,53.])
    f=pd.DataFrame(dict(fill_ms=[10.,10.],side=[0,1],price_cents=[49.,52.],shares=[10.,4.],pre_mid_cents=[50.,50.],pre_mid_age_ms=[10.,10.],
        spread_capture_usd=[.10,.08],settlement_yes=[1.,1.],settlement_pnl_usd=[5.1,-1.92],terminal_mid_cents=[53.,53.],terminal_pnl_usd=[.4,-.04]))
    for h,mark in [(100,49.),(500,48.),(1000,52.),(5000,53.)]:
        f[f'markout_{h}_usd']=f.shares*(1-2*f.side)*(mark-f.price_cents)/100
        f[f'mid_drift_{h}_usd']=f.shares*(1-2*f.side)*(mark-50)/100
    verify_values(f,t,m,1.,5000.);refused=[]
    for column in ['fill_ms','side','pre_mid_cents','settlement_yes','markout_1000_usd','terminal_mid_cents','terminal_pnl_usd']:
        bad=f.copy();bad.loc[0,column]+=1
        caught=False
        try:verify_values(bad,t,m,1.,5000.)
        except ValueError:caught=True
        L.require(caught,'audit corruption not refused '+column);refused.append(column)
    return dict(positive_bid_ask_and_clock=True,corruptions_refused=refused,scalar_up_index_positive=True,bad_token_mappings_refused=mapping_refused)


def run():
    S.verify();out=S.REVIEW/'results_v1';p=out/'audit_v1.json';L.require(not p.exists(),'audit exists')
    result=dict(asof=A.stamp(),controls=controls(),report_sha256=L.sha(out/'report.json'),days={},limits='Raw-input identity and arithmetic audit, not physical fill/timing validation or inference.')
    parts=[]
    for day in L.DAYS:
        d=S.OUT/day;r=json.loads((d/'receipt.json').read_text());fp=d/'fills.parquet'
        L.require(L.sha(fp)==r['outputs']['fills.parquet'],'changed ledger')
        books={p:h for p,h in r['input_hashes'].items() if f'/step1/{day}/books/' in p}
        L.require(bool(books),'no raw-book provenance')
        for path,h in books.items():L.require(L.sha(path)==h,'changed valuation book')
        # The loader includes the frozen admissibility rules for input parts.
        D=S.K.load_day(day);f=pd.read_parquet(fp)
        regpath=Path(S.K.IO)/'step1'/day/'registry.parquet'
        L.require(L.sha(regpath)==r['input_hashes'][str(regpath)],'changed resolution source')
        reg=pd.read_parquet(regpath).set_index('mid');L.require(reg.index.is_unique,'duplicate registry identity')
        n=0
        for mid,g in f.groupby('m',sort=False):
            b=D['B'].iloc[D['BI'][mid]].sort_values('t_ms',kind='stable')
            valid=(b.b1_p>=0)&(b.a1_p>b.b1_p);times=b.loc[valid,'t_ms'].to_numpy(float);mids=((b.loc[valid,'b1_p']+b.loc[valid,'a1_p'])/2).to_numpy(float)
            outcome=reg.loc[mid,'outcome'];L.require(outcome in ('Up','Down'),'unresolved outcome')
            verify_up_index(reg.loc[mid,'up_index'])
            verify_values(g,times,mids,float(outcome=='Up'),float(reg.loc[mid,'ws'])*1000.+60200.);n+=len(g)
        L.require(n==len(f),'unverified fills')
        f['source']=np.where(f.ws_timed,'WS','E2');f['age_group']=np.where(f.age_ms<100,'lt100',np.where(f.age_ms<500,'100to500','ge500'))
        fields=['shares','spread_capture_usd','terminal_pnl_usd','settlement_pnl_usd']+[f'markout_{h}_usd' for h in (100,500,1000,5000)]
        grouped=f.groupby(['day','case','source','age_group'],observed=True)
        z=grouped[fields].sum();z['fills']=grouped.size();z=z.reset_index();parts.append(z)
        L.require(int(z.fills.sum())==len(f),'source/age fill-count reconciliation')
        for field in fields:S.Q.close(z[field].sum(),f[field].sum(),'source/age reconciliation')
        result['days'][day]=dict(fills_verified=n,markets_with_fills=int(f.m.nunique()),cases=int(f.case.nunique()),book_files_verified=len(books),receipt_sha256=L.sha(d/'receipt.json'),ledger_sha256=L.sha(fp),registry_sha256=L.sha(regpath))
        print(json.dumps(dict(day=day,fills_verified=n)),flush=True)
    attribution=pd.concat(parts,ignore_index=True);cp=out/'source_age_attribution.csv';attribution.to_csv(cp,index=False)
    result.update(completed_asof=A.stamp(),fills_verified=sum(x['fills_verified'] for x in result['days'].values()),source_sha256=L.sha(__file__),source_age_attribution_sha256=L.sha(cp))
    A.dump(p,result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
