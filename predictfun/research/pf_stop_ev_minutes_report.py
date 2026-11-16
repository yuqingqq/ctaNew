"""Reconcile and summarize the frozen continuous-minute experiment."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_stop_ev_minutes as R
I,V,A,L,Q,np,pd=R.I,R.V,R.A,R.L,R.Q,R.np,R.pd


def aggregate(g):
    fields=[c for c in g if c not in ['day','m','case','minute']]
    out={c:float(g[c].sum()) for c in fields};out['market_windows']=len(g)
    for key in ['spread_capture','terminal_pnl','settlement_pnl']:
        den=out[key+'_covered_shares'];out[key+'_cents_per_share']=100*out[key+'_usd']/den if den else None
    for h in V.HORIZONS:
        den=out[f'markout_{h}_ok_shares'];out[f'markout_{h}_cents_per_share']=100*out[f'markout_{h}_usd']/den if den else None
        out[f'markout_{h}_coverage']=den/out['shares'] if out['shares'] else None
    return out


def check_day(w,f,inventory,markets):
    L.require(len(w)==len(markets)*15 and not w.duplicated(['m','case','minute']).any(),'minute population')
    L.require(set(w.m)==set(markets),'market identity')
    L.require(set(w.case)=={'STOP_EV','TOP5','OTHER_MAKERS'},'case identity')
    L.require(not f.duplicated(['m','case','side','rung','fill_seq']).any(),'duplicate fills')
    groups=f.groupby(['m','case','minute']).indices
    for row in w.itertuples(index=False):
        g=f.iloc[groups.get((row.m,row.case,row.minute),np.zeros(0,int))]
        z=V.summarize_fills(g)
        for key,value in z.items():Q.close(value,getattr(row,key),'fill/minute reconciliation '+key)
        for h in V.HORIZONS:
            Q.close(sum(z[f'markout_{h}_{state}_shares'] for state in ['ok','expiry_censored','missing_book']),z['shares'],'coverage quantity partition')
            L.require(sum(z[f'markout_{h}_{state}_fills'] for state in ['ok','expiry_censored','missing_book'])==z['fills'],'coverage fill partition')
    inv=inventory.groupby('m')
    for m,g in inv:
        L.require(g.minute.tolist()==[1,2,3,4,5],'inventory minute sequence')
        Q.close(g.incremental_marked_pnl_usd.sum(),g.pnl_at_minute_end_usd.iloc[-1],'minute increment telescoping')
        mine=w[(w.m==m)&(w.case=='STOP_EV')]
        Q.close(mine.terminal_pnl_usd.sum(),g.pnl_at_minute_end_usd.iloc[-1],'minute fill/portfolio value')
    return len(f)


def scalar(v,fmt='.3f'):
    return 'n/a' if v is None else format(v,fmt)


def above(a,b):
    return bool(a>b) if a is not None and b is not None else None


def render(result,path):
    t=result['totals'];inv=result['inventory']
    lines=['# STOP_EV across the five-minute contract', '',
        f"As of {result['asof']}. Five consumed dates: Oct27, Oct31, Nov1-3; 1,434 common market windows, six Oct31 boundary exclusions. Frozen corrected-v3 STOP_EV, place58/cancel37ms, notice100ms, review50ms, M50/queueM, six30-share rungs per side, no cap/cooldown. Quoting ends at299.8s and the cancellation tail ends at300s. Orders and inventory carry across minute boundaries. No model retraining or inventory-control change.", '',
        'Top five are ranked per date by maker volume during0-240s on these same markets, not by profitability. Their public executed quantities and our simulated fills are valued on the same M50/imputation convention. This aligns accounting and market coverage; it does not make the two fill populations interchangeable or validate simulated execution.', '',
        'All tables retain fills. Markout denominators include only fills with that horizon available before expiry; censored and missing quantities are explicit. All dollar columns are five-date totals. Positive markout is not an executable liquidation profit. The first-minute-trained model is being applied to an untrained later phase.', '',
        '## Fill quality and volume by minute', '',
        '| Minute | Population | Filled shares | Initial edge c/sh | +100ms c/sh | +1s c/sh | +5s c/sh | +1s value $ | 1s / 5s coverage % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for minute in range(1,6):
        for case in ['STOP_EV','TOP5','OTHER_MAKERS']:
            z=t[case][str(minute)]
            lines.append(f"| {minute} | {case} | {z['shares']:,.0f} | {scalar(z['spread_capture_cents_per_share'])} | {scalar(z['markout_100_cents_per_share'])} | {scalar(z['markout_1000_cents_per_share'])} | {scalar(z['markout_5000_cents_per_share'])} | {z['markout_1000_usd']:+,.2f} | {scalar(None if z['markout_1000_coverage'] is None else 100*z['markout_1000_coverage'],'.2f')} / {scalar(None if z['markout_5000_coverage'] is None else 100*z['markout_5000_coverage'],'.2f')} |")
    lines+=['','## STOP_EV inventory and continuous marked value', '',
        '| Minute | Incremental marked P&L $ | Cumulative marked P&L $ | Mean abs inventory | Mean abs ending inventory | Max abs inventory | Any / both quote % | Submissions / guard pulls |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for minute in range(1,6):
        z=inv[str(minute)]
        lines.append(f"| {minute} | {z['incremental_marked_pnl_usd']:+,.2f} | {z['pnl_at_minute_end_usd']:+,.2f} | {z['mean_abs_inventory']:.2f} | {z['mean_abs_position_end']:.2f} | {z['max_abs_inventory']:.2f} | {100*z['quote_any_fraction']:.2f} / {100*z['quote_both_fraction']:.2f} | {z['orders']:,.0f} / {z['guard_pulls']:,.0f} |")
    lines+=['','These inventory measures describe only the simulated strategy. Observed maker fills do not reveal their starting inventory, taker executions, hedges or complete account P&L. Minute marked increments include the movement of inventory carried from earlier minutes; they are not sums of independent minute restarts. Guard pulls exclude routine price and final cancellations.', '',
        '## Per-date one-second markout c/share', '', '| Date | Population | Minute1 | Minute2 | Minute3 | Minute4 | Minute5 |','|---|---|---:|---:|---:|---:|---:|']
    for day in R.L.DAYS:
        for case in ['STOP_EV','TOP5']:
            cells=[scalar(result['by_day'][day][case][str(m)]['markout_1000_cents_per_share']) for m in range(1,6)]
            lines.append('| '+day+' | '+case+' | '+' | '.join(cells)+' |')
    lines+=['','## Coverage, controls and limits', '',
        f"All {result['checks']['first_minute_replay_parity']:,} independent60.2s control replays match the prior scorecard including fill identities; first-minute market inputs match bit-for-bit. All {result['checks']['continuous_prefix_parity']:,} continuous trajectories match the stopped control up to59.2s, when its clipped final horizon can first differ. The continuous first minute can therefore differ near its end from the original stopped-first-minute scorecard.", '',
        f"All {result['checks']['audited_fills']:,} exported records reconcile to the minute tables and raw-book markouts. Inventory sums reconcile to engine cash, position and terminal value. Corrupted values and horizon statuses are refused. The full comparison still depends on the inherited queue, trade-time and imputation assumptions. No inference, untouched validation or promotion claim is made.", '',
        'Near-expiry unavailable markouts are not replaced with settlement or the last midpoint. Midpoints older than1s are counted and retained as a separate coverage diagnostic. Source, order-age and actual-depth partitions are in attribution.csv; horizon-specific denominators and missing/censored counts are in the CSV/JSON. Values attributed to a subset of fills are not a replay that removes the other fills.', '',
        'Artifacts: `reviews/stop_ev_minutes_20261116/results_v1/`; per-fill ledgers and per-market inputs/statuses: `data/predict_fun/research/stop_ev_minutes_20261116_v1/`. The original scorecard, model files and historical maker study remain unchanged.']
    path.write_text('\n'.join(lines)+'\n')


def run():
    declaration=R.verify();out=R.REVIEW/'results_v1';L.require(not out.exists(),'report exists');out.mkdir()
    frames=[];invframes=[];featframes=[];parts=[];receipts={};checks=R.Counter()
    for day in R.L.DAYS:
        p=R.OUT/day;r=json.loads((p/'receipt.json').read_text());L.require(not r['smoke_only'],'smoke in report')
        for name,h in r['outputs'].items():L.require(L.sha(p/name)==h,'changed day artifact')
        L.require(r['freeze_sha256']==L.sha(R.REVIEW/'freeze_v1.json'),'wrong freeze identity')
        w=pd.read_parquet(p/'minute_values.parquet');f=pd.read_parquet(p/'fills.parquet');i=pd.read_parquet(p/'inventory.parquet')
        check_day(w,f,i,r['selected_markets'])
        bad=w.copy();bad.loc[bad.index[0],'shares']+=1;caught=False
        try:check_day(bad,f,i,r['selected_markets'])
        except ValueError:caught=True
        L.require(caught,'corrupted minute quantity not refused')
        for key in ['windows','first_minute_replay_parity','continuous_prefix_parity','feature_rows_first_minute_parity','audited_fills','strategy_fills','maker_legs']:checks[key]+=r['counts'][key]
        frames.append(w);invframes.append(i);featframes.append(pd.read_parquet(p/'feature_support.parquet'))
        f['source']=np.where(f.imputed,'IMPUTED',np.where(f.ws_timed,'WS','E2'))
        f['age_group']=np.where(f.case!='STOP_EV','unobserved',np.where(f.age_ms<100,'lt100',np.where(f.age_ms<500,'100to500','ge500')))
        for keys,g in f.groupby(['case','minute','depth_group','source','age_group'],sort=False):
            parts.append(dict(day=day,**dict(zip(['case','minute','depth_group','source','age_group'],keys)),**V.summarize_fills(g)))
        receipts[day]=dict(sha256=L.sha(p/'receipt.json'),asof=r['asof'],counts=r['counts'],top5_ids=r['top5_ids'])
    L.require(checks['windows']==1434,'wrong complete population')
    w=pd.concat(frames,ignore_index=True);i=pd.concat(invframes,ignore_index=True);totals={};byday={d:{} for d in R.L.DAYS}
    for case,g in w.groupby('case',sort=False):
        totals[case]={str(m):aggregate(h) for m,h in g.groupby('minute')}
        for day,z in g.groupby('day'):byday[day][case]={str(m):aggregate(h) for m,h in z.groupby('minute')}
    inventories={}
    for minute,g in i.groupby('minute'):
        z={c:float(g[c].sum()) for c in g if c not in ['day','m','minute']}
        z['mean_abs_inventory']=z['abs_inventory_share_ms']/z['observed_ms']
        z['mean_abs_position_end']=z['abs_position_end']/len(g);z['max_abs_inventory']=float(g.max_abs_inventory.max())
        z['quote_any_fraction']=z['quote_any_ms']/z['observed_ms'];z['quote_both_fraction']=z['quote_both_ms']/z['observed_ms']
        inventories[str(minute)]=z
    result=dict(asof=A.stamp(),declaration=declaration,receipts=receipts,checks=dict(checks),totals=totals,by_day=byday,inventory=inventories,
        predicates={str(m):dict(stop_ev_1s_positive=totals['STOP_EV'][str(m)]['markout_1000_usd']>0,
            stop_ev_5s_positive=totals['STOP_EV'][str(m)]['markout_5000_usd']>0,
            simulated_1s_above_observed_top5=above(totals['STOP_EV'][str(m)]['markout_1000_cents_per_share'],totals['TOP5'][str(m)]['markout_1000_cents_per_share'])) for m in range(1,6)},
        limits='Descriptive consumed-data transfer; simulated vs observed fills; no inventory policy, costs, significance or promotion.')
    A.dump(out/'report.json',result)
    pd.DataFrame([dict(case=c,minute=m,**v) for c,t in totals.items() for m,v in t.items()]).to_csv(out/'minute_totals.csv',index=False)
    pd.DataFrame([dict(day=d,case=c,minute=m,**v) for d,z in byday.items() for c,t in z.items() for m,v in t.items()]).to_csv(out/'by_day.csv',index=False)
    pd.DataFrame([dict(minute=m,**v) for m,v in inventories.items()]).to_csv(out/'inventory.csv',index=False)
    attribution=pd.DataFrame(parts);attribution.to_parquet(R.OUT/'attribution.parquet',index=False)
    # Compact published partitions, with complete detailed cells retained outside git.
    keys=['case','minute','source','age_group']
    numeric=[c for c in attribution if c not in ['day','depth_group']+keys]
    attribution.groupby(keys,as_index=False)[numeric].sum().to_csv(out/'attribution.csv',index=False)
    depth_keys=['case','minute','depth_group'];numeric=[c for c in attribution if c not in ['day','source','age_group']+depth_keys]
    attribution.groupby(depth_keys,as_index=False)[numeric].sum().to_csv(out/'depth_attribution.csv',index=False)
    pd.concat(featframes).groupby(['minute','side'],as_index=False)[['valid_rows','missing_cells','market_cells','outside_training_phase']].sum().to_csv(out/'feature_support.csv',index=False)
    render(result,R.HERE.parent/'handoffs/20261116_stop_ev_minutes.md')
    print(json.dumps(dict(asof=result['asof'],checks=checks,headline={case:{m:{k:v[k] for k in ['shares','markout_1000_cents_per_share','markout_5000_cents_per_share','terminal_pnl_usd','settlement_pnl_usd']} for m,v in z.items()} for case,z in totals.items()}),indent=2))


if __name__=='__main__':run()
