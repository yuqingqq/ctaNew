"""Audit and visualize the declared minute experiment; no policy selection.

Uses the frozen primary outputs. Decompositions are descriptive identities,
not counterfactual removal replays. All partitions are retained.
"""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_stop_ev_minutes as R
A,L,Q,np,pd=R.A,R.L,R.Q,R.np,R.pd


def reconcile(r, tables):
    for case,z in r['totals'].items():
        for minute,v in z.items():
            row=tables[(tables.case==case)&(tables.minute==int(minute))]
            L.require(len(row)==1,'published row identity')
            for key,value in v.items():
                if value is not None:Q.close(value,float(row.iloc[0][key]),'published JSON/CSV '+key)
            for h in R.V.HORIZONS:
                Q.close(sum(v[f'markout_{h}_{s}_shares'] for s in ['ok','expiry_censored','missing_book']),v['shares'],'published coverage')
                if int(minute)<5:Q.close(v[f'markout_{h}_expiry_censored_shares'],0.,'early minute expiry coverage')
    return True


def ratio(frame,h=1000):
    den=float(frame[f'markout_{h}_ok_shares'].sum())
    return 100*float(frame[f'markout_{h}_usd'].sum())/den if den else None


def main():
    R.verify();out=R.REVIEW/'results_v2';path=out/'report.json';r=json.loads(path.read_text())
    L.require(not (out/'audit_v1.json').exists(),'audit output exists')
    table=pd.read_csv(out/'minute_totals.csv');reconcile(r,table)
    bad=copy.deepcopy(r);bad['totals']['STOP_EV']['2']['shares']+=1;caught=False
    try:reconcile(bad,table)
    except ValueError:caught=True
    L.require(caught,'published corruption not refused')
    sources={};status={};checks={};age=pd.read_csv(out/'attribution.csv');depth=pd.read_csv(out/'depth_attribution.csv')
    for day in R.L.DAYS:
        p=R.OUT/day;rec=json.loads((p/'receipt.json').read_text())
        L.require(L.sha(p/'receipt.json')==r['receipts'][day]['sha256'],'receipt identity')
        for file,h in rec['outputs'].items():L.require(L.sha(p/file)==h,'day output identity')
        for file,h in rec['input_hashes'].items():L.require(L.sha(file)==h,'raw input identity')
        st={}
        for batch in rec['loader_batches']:
            for k,v in batch.items():
                if k not in ['batch','registry','outside_day','sealed_or_forward','no_outcome','parts_skipped']:st[k]=st.get(k,0)+v
        f=pd.read_parquet(p/'fills.parquet',columns=['case','shares','spread_capture_usd','pre_mid_cents','price_cents','side']+
            [f'{k}_{h}_usd' for h in R.V.HORIZONS for k in ['markout','mid_drift']])
        makers=f[f.case!='STOP_EV'];selected=st['raw_selected_legs'];excluded=sum(st[x+'_legs'] for x in ['untimed_outside_context','anchored_outside_context','before_contract','after_contract'])
        L.require(selected==len(makers)+excluded,'raw maker leg population reconciliation')
        excluded_q=sum(st[x+'_shares'] for x in ['untimed_outside_context','anchored_outside_context','before_contract','after_contract'])
        Q.close(st['raw_selected_shares'],float(makers.shares.sum())+excluded_q,'raw maker share population reconciliation')
        for h in R.V.HORIZONS:
            v=f[f'markout_{h}_usd'];d=f[f'mid_drift_{h}_usd'];s=f.spread_capture_usd
            mask=np.isfinite(v)&np.isfinite(s)
            L.require(np.allclose(v[mask],s[mask]+d[mask],rtol=0,atol=1e-8),'retained edge decomposition')
        status[day]=st;checks[day]=dict(exported_fills=len(f),maker_legs=len(makers),excluded_maker_legs=excluded,
            feature_and_60s_parity=rec['counts']['first_minute_replay_parity'],continuous_prefix=rec['counts']['continuous_prefix_parity'])
        sources[day]=L.sha(p/'receipt.json')
    diagnostics={}
    for m in range(1,6):
        z={}
        for case in ['STOP_EV','TOP5','OTHER_MAKERS']:
            v=r['totals'][case][str(m)];den=v['markout_1000_ok_shares'];g=age[(age.case==case)&(age.minute==m)]
            d=depth[(depth.case==case)&(depth.minute==m)]
            z[case]=dict(shares=v['shares'],fills=v['fills'],markout_1s_c=ratio(g),
                postfill_drift_1s_c=100*v['mid_drift_1000_usd']/den if den else None,
                pre_mid_stale_share_fraction=v['pre_mid_stale_shares']/v['shares'] if v['shares'] else None,
                markout_1s_stale_share_fraction=v['markout_1000_stale_shares']/den if den else None,
                ws_share_fraction=v['ws_shares']/v['shares'] if v['shares'] else None,
                imputed_share_fraction=v['imputed_shares']/v['shares'] if v['shares'] else None,
                age_lt100_share_fraction=v['young_shares']/v['shares'] if v['shares'] and case=='STOP_EV' else None,
                at_best_1s_c=ratio(d[d.depth_group=='at_best']),inside_1s_c=ratio(d[d.depth_group=='inside']),
                resting_ge500ms_1s_c=ratio(g[g.age_group=='ge500']) if case=='STOP_EV' else None,
                ws_only_1s_c=ratio(g[g.source=='WS']))
        z['stop_above_top5_days_1s']=sum(r['by_day'][day]['STOP_EV'][str(m)]['markout_1000_cents_per_share']>
            r['by_day'][day]['TOP5'][str(m)]['markout_1000_cents_per_share'] for day in R.L.DAYS)
        diagnostics[str(m)]=z
    audit=dict(asof=A.stamp(),source_commit=A.commit(),source_sha256=L.sha(Path(__file__)),report_sha256=L.sha(path),
        checks=dict(published_positive=True,published_corruption_refused=caught,raw_maker_population=True,
            retained_edge_identity=True,input_and_output_identity=True,per_day=checks),
        receipts=sources,loader_status=status,diagnostics=diagnostics,
        scope='Five consumed dates; 1434 markets; no inference. Clock-aligned simulated vs real fills. Depth/source/age subsets are descriptive, not counterfactual strategies.')
    # Static figure, exported as a research artifact.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    minutes=list(range(1,6))
    for case,label,color in [('STOP_EV','STOP_EV · simulated','#1464a0'),('TOP5','Top five · observed','#d46919')]:
        for ax,h,title in zip(axes[:2],[1000,5000],['1-second markout','5-second markout']):
            vals=[r['totals'][case][str(m)][f'markout_{h}_cents_per_share'] for m in minutes]
            ax.plot(minutes,vals,marker='o',label=label,color=color);ax.set_title(title);ax.set_ylabel('Cents per covered share')
        axes[2].plot(minutes,[r['totals'][case][str(m)]['shares']/1000 for m in minutes],marker='o',label=label,color=color)
    for ax in axes:ax.set_xlabel('Minute of contract');ax.set_xticks(minutes);ax.grid(alpha=.2)
    for ax in axes[:2]:ax.axhline(0,color='gray',linewidth=.7)
    axes[2].set_title('Filled volume');axes[2].set_yscale('log');axes[2].set_ylabel('Thousand shares · log scale')
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('STOP_EV transfer across five minutes · 5 dates / 1,434 markets',fontsize=13)
    fig.text(.5,-.05,'58/37ms order clocks · common M50 valuation · minute-five expiry coverage is incomplete · gross, descriptive',ha='center',fontsize=8)
    chart=out/'minutes.png';fig.savefig(chart,dpi=150,bbox_inches='tight');plt.close(fig)
    audit['chart_sha256']=L.sha(chart);A.dump(out/'audit_v1.json',audit)
    print(json.dumps(dict(asof=audit['asof'],checks=audit['checks'],diagnostics=diagnostics),indent=2))


if __name__=='__main__':main()
