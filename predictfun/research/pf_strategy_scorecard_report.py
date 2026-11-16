"""Verify and render the frozen strategy scorecard from replay/fill artifacts."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_strategy_scorecard as S
Q,L,A,np,pd=S.Q,S.L,S.A,S.np,S.pd

ADDITIVE=(['shares','fills','buy_shares','sell_shares','spread_capture_usd','terminal_pnl_usd','settlement_pnl_usd','settlement_carry_usd',
    'orders','pulls','rejected_orders','filled_orders','terminal_abs_position','quote_any_ms','quote_both_ms','quote_bid_ms','quote_ask_ms','quote_observed_ms',
    'abs_inventory_share_ms','inventory_observed_ms','legacy_anchor_markout1_usd','legacy_anchor_markout5_usd','pre_mid_stale_1000ms_shares']+
    [f'{p}_{h}_usd' for h in Q.HORIZONS for p in ['markout','mid_drift']]+
    [f'mark_{h}_stale_1000ms_shares' for h in Q.HORIZONS]+
    [f'{p}_{suffix}' for p in ['young','older','age100_500','age500','ws','e2','pending','imputed','through'] for suffix in ['terminal_usd','shares']])


def summarize(g):
    d={c:float(g[c].sum()) for c in ADDITIVE};d['windows']=len(g)
    sh=d['shares']
    for c in ['spread_capture_usd','terminal_pnl_usd','settlement_pnl_usd']+[f'markout_{h}_usd' for h in Q.HORIZONS]:d[c.replace('_usd','_cents_per_share')]=100*d[c]/sh if sh else None
    for name in ['any','both','bid','ask']:d[f'quote_{name}_fraction']=d[f'quote_{name}_ms']/d['quote_observed_ms']
    d.update(mean_abs_inventory=d['abs_inventory_share_ms']/d['inventory_observed_ms'],max_abs_inventory=float(g.max_abs_position.max()),
        mean_terminal_abs_inventory=d['terminal_abs_position']/len(g),mean_legacy_rung_exposure=float(g.legacy_mean_rung_exposure.mean()),
        pull_per_order=d['pulls']/d['orders'] if d['orders'] else None,filled_order_fraction=d['filled_orders']/d['orders'] if d['orders'] else None,
        worst_window_terminal=float(g.terminal_pnl_usd.min()),worst_window_settlement=float(g.settlement_pnl_usd.min()))
    daily=g.groupby('day')[['terminal_pnl_usd','settlement_pnl_usd']].sum()
    d.update(days=len(daily),terminal_positive_days=int((daily.terminal_pnl_usd>0).sum()),settlement_positive_days=int((daily.settlement_pnl_usd>0).sum()),
        settlement_day_std=float(daily.settlement_pnl_usd.std(ddof=1)) if len(daily)>1 else None,
        terminal_day_std=float(daily.terminal_pnl_usd.std(ddof=1)) if len(daily)>1 else None)
    return d


def verify_ledger(w,f):
    L.require(len(f)==int(w.fills.sum()),'fill count mismatch')
    L.require(not f.duplicated(['day','m','case','side','rung','fill_seq']).any(),'duplicate ledger fill')
    groups=f.groupby(['m','case'],sort=False)
    for row in w.itertuples(index=False):
        key=(row.m,row.case);g=groups.get_group(key) if key in groups.indices else f.iloc[:0]
        for field in ['shares','spread_capture_usd','terminal_pnl_usd','settlement_pnl_usd']+[f'markout_{h}_usd' for h in Q.HORIZONS]+[f'mid_drift_{h}_usd' for h in Q.HORIZONS]:
            Q.close(g[field].sum(),getattr(row,field),'ledger/window reconciliation '+field)
        Q.reconcile(row._asdict())
    if len(f):
        sg=np.where(f.side.to_numpy()==0,1.,-1.);q=f.shares.to_numpy();p=f.price_cents.to_numpy()
        L.require(np.allclose(f.spread_capture_usd,q*sg*(f.pre_mid_cents-p)/100,rtol=0,atol=1e-9),'ledger spread sign')
        L.require(np.allclose(f.settlement_pnl_usd,q*sg*(f.settlement_yes-p/100),rtol=0,atol=1e-9),'ledger settlement sign')
        L.require(np.allclose(f.fill_ms,f.anchor_ms-50,rtol=0,atol=1e-8),'ledger fill clock')


def render(result,path):
    totals=result['totals'];lines=['# Predict strategy scorecard', '',f"As of {result['asof']}. Five consumed dates: 2026-10-27, 10-31, 11-01, 11-02, 11-03. Each case covers 1,434 first-minute windows; six Oct31 boundary windows excluded. All fills retained. Gross simulated values, no fees/rewards added. Quoting stops at 60s; final cancel tail ends at 60.2s. Settlement means holding that residual inventory to the market’s recorded binary resolution, with no further trading.", '',
        '## Names: strategy and execution settings are separate', '',
        '| Strategy ID | Plain name | Entry | Resting-order decision | Former name |','|---|---|---|---|---|']
    for key,v in S.CATALOG.items():lines.append(f"| {key} | {v['name']} | {v['entry']} | {v['resting']} | {v['old']} |")
    lines+=['','STOP-EV v3 is the **model**. STOP_EV is one **strategy using that model** for both entry and cancellation. FILL_GUARD uses only its fill head. ARRIVAL_EV replaces entry and keeps the STOP_EV cancellation rule. FUTURE_ENTRY uses future information and is not deployable. ORDER_EV is the separate order-state-only model.', '',
        'The earlier STOP-EV-PLAY v2 / QR replay is **FILL_GUARD_250_RECALIBRATED**: 250ms decisions and separately calibrated thresholds. It is not the FILL_GUARD 50ms fixed-threshold case below. Older Huber/local one-second model tests are prediction diagnostics, not strategy P&L rows. Their negative selected-label sums must never be compared directly with this table.', '',
        '| Clock ID | Place / cancel ms | Pending cancel / notice ms | Feed |','|---|---|---|---|']
    for key,v in S.CLOCKS.items():lines.append(f"| {key} | {v['placement_ms']} / {v['cancel_ms']} | {v['pending_cancel_ms']} / {v['notification_ms']} | {v['feed']} |")
    lines+=['','Every case uses a 50ms review grid, M50 trade-time assumption, queue M with unknown-depth queue zero, six 30-share rungs per side, no inventory cap or fixed cooldown. Clock IDs are scenarios, not measured present production latency. Compare models within a clock block; compare clocks within STOP_EV. No new model or threshold was fitted.', '',
        '## Metric definitions', '',
        'All price references are YES-equivalent cents, with side sign +1 for buys and -1 for sells. Let q be filled shares, p the execution price, m0 the last valid book midpoint strictly before modeled fill time, mh the last valid midpoint at or before fill+h, and y the recorded settlement payout (0 or 1 dollars).', '',
        '- **Spread capture / execution edge:** sum(q × side × (m0 − p))/100 dollars. This is an edge against the pre-fill public midpoint, not necessarily a matched round-trip spread or executable liquidation value.',
        '- **Post-fill midpoint movement:** sum(q × side × (mh − m0))/100 dollars. Negative values measure adverse movement after fills; they are not mechanically deducted a second time.',
        '- **Markout at h:** spread capture + post-fill midpoint movement = sum(q × side × (mh − p))/100 dollars. Report h=100ms, 500ms, 1s and 5s, in dollars and cents per filled share.',
        '- **Terminal P&L:** cash + signed remaining inventory × midpoint at 60.2s. This is the number previously called P&L in the latency/arrival reports.',
        '- **Settlement P&L:** cash + signed remaining inventory × y. **Settlement carry** = settlement P&L − terminal P&L. It isolates the effect of carrying the final inventory to resolution.',
        '- **Quote availability:** fraction of the quoting interval with any physically live, unfilled order; “both” requires one on each side. Overlapping intervals are merged. This differs from summed per-rung exposure or keep-decision frequency.',
        '- **Inventory:** time-weighted mean absolute inventory and maximum absolute inventory during replay. Different markets settle independently; their signed positions are not netted across windows.', '',
        'Marks use the **modeled fill clock**, which is the recorded/imputed trade anchor minus 50ms in these cases. Previous reports used anchor+1s/5s; those old measures are retained separately in the CSV/JSON and must reproduce exactly. Missing required marks refuse the report; stale-mid shares are counted without dropping fills.', '']
    for clock,title in [('recorded58','Recorded-median execution: current entry comparison'),('legacy40','Legacy execution: model/rule comparison'),('zero','Zero-delay STOP_EV diagnostics')]:
        lines += [f'## {title}', '', '| Strategy / clock | Shares | Spread c/sh | +1s c/sh | +5s c/sh | Terminal $ | Settlement $ | Settlement carry $ |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for c in S.CASES:
            if not (c['clock'].startswith('zero') if clock=='zero' else c['clock']==clock):continue
            v=totals[c['id']];lines.append(f"| {c['strategy']} / {c['clock']} | {v['shares']:,.0f} | {v['spread_capture_cents_per_share']:+.3f} | {v['markout_1000_cents_per_share']:+.3f} | {v['markout_5000_cents_per_share']:+.3f} | {v['terminal_pnl_usd']:+.2f} | {v['settlement_pnl_usd']:+.2f} | {v['settlement_carry_usd']:+.2f} |")
        lines+=['']
    lines+=['## Economic decomposition', '', '| Case | Spread $ | Midpoint movement to +1s $ | +1s markout $ | +100ms c/sh | +500ms c/sh |','|---|---:|---:|---:|---:|---:|']
    for c in S.CASES:
        v=totals[c['id']];lines.append(f"| {c['id']} | {v['spread_capture_usd']:+.2f} | {v['mid_drift_1000_usd']:+.2f} | {v['markout_1000_usd']:+.2f} | {v['markout_100_cents_per_share']:+.3f} | {v['markout_500_cents_per_share']:+.3f} |")
    lines+=['','## Activity and inventory', '', '| Case | Orders / pulls | Filled orders % | Any / both quote % | Mean / max abs inventory | Terminal / settlement positive dates |','|---|---:|---:|---:|---:|---:|']
    for c in S.CASES:
        v=totals[c['id']];lines.append(f"| {c['id']} | {v['orders']:,.0f} / {v['pulls']:,.0f} | {100*v['filled_order_fraction']:.2f} | {100*v['quote_any_fraction']:.2f} / {100*v['quote_both_fraction']:.2f} | {v['mean_abs_inventory']:.1f} / {v['max_abs_inventory']:.1f} | {v['terminal_positive_days']}/5 / {v['settlement_positive_days']}/5 |")
    lines+=['','## Attribution and mark coverage', '', '| Case | Terminal excluding <100ms fills $ | WS / E2 terminal $ | Pending-cancel volume % | Pre-fill midpoint older than 1s: shares |','|---|---:|---:|---:|---:|']
    for c in S.CASES:
        v=totals[c['id']];lines.append(f"| {c['id']} | {v['older_terminal_usd']:+.2f} | {v['ws_terminal_usd']:+.2f} / {v['e2_terminal_usd']:+.2f} | {100*v['pending_shares']/v['shares']:.2f} | {v['pre_mid_stale_1000ms_shares']:.2f} |")
    lines+=['','These are additive fill partitions, not reruns that remove those fills. Full age/source/horizon coverage, per-day values, gross fill ledgers and hashes are retained in the artifacts.', '',
        '## Per-date terminal / settlement dollars', '', '| Case | Oct27 | Oct31 | Nov1 | Nov2 | Nov3 |','|---|---:|---:|---:|---:|---:|']
    for c in S.CASES:
        cells=[result['by_day'][d][c['id']] for d in L.DAYS];lines.append('| '+c['id']+' | '+' | '.join(f"{v['terminal_pnl_usd']:+.2f} / {v['settlement_pnl_usd']:+.2f}" for v in cells)+' |')
    lines+=['','## Checks, provenance and interpretation limits', '',
        f"All {result['checks']['prior_window_parity']:,} strategy-window reproductions pass against their named previous artifacts, including existing terminal values, shares, order/pull counts, fill counts, inventory maxima, exposure, age/source splits and old anchor markouts. Every fill ledger reconciles to portfolio cash/inventory and settlement; pre-fill timing, bid/ask signs, overlap-safe availability and known-bad controls pass.", '',
        'Settlement can be dominated by a few inventory outcomes. It is a separate endpoint, not a substitute for fill quality. Day dispersion and worst-window values are included in JSON/CSV; no confidence interval, significance claim, selection of a new policy, untouched validation, or live-profit claim is made. Queue/timing assumptions and public-book counterfactual fills remain unvalidated; simulated volume and dollars can be inflated. No M0/M250/queue-L robustness reruns are added.', '',
        'Canonical registry: `predictfun/reviews/strategy_scorecard_20261116/freeze_v1.json`. Results: `results_v1/report.json`, `strategy_totals.csv`, `strategy_by_day.csv`. Fill and window ledgers: `data/predict_fun/research/strategy_scorecard_20261116_v1/`. Historical result files are unchanged. Always cite strategy ID + clock ID + mark horizon + included fill population.']
    path.write_text('\n'.join(lines)+'\n')


def run():
    freeze=S.verify();out=S.REVIEW/'results_v1';L.require(not out.exists(),'report exists');out.mkdir(parents=True)
    receipts={};frames=[];checks=dict(prior_window_parity=0,ledger_windows=0,ledger_fills=0)
    for day in L.DAYS:
        d=S.OUT/day;receipt=json.loads((d/'receipt.json').read_text());L.require(not receipt['smoke_only'],'smoke output in report')
        for name,h in receipt['outputs'].items():L.require(L.sha(d/name)==h,'ledger output changed')
        L.require(receipt['freeze_sha256']==L.sha(S.REVIEW/'freeze_v1.json'),'wrong frozen accounting')
        w=pd.read_parquet(d/'windows.parquet');f=pd.read_parquet(d/'fills.parquet');verify_ledger(w,f)
        if len(f):
            bad=f.copy();bad.loc[bad.index[0],'shares']+=1;caught=False
            try:verify_ledger(w,bad)
            except ValueError:caught=True
            L.require(caught,'corrupted ledger not refused')
        checks['prior_window_parity']+=receipt['counts']['prior_window_parity'];checks['ledger_windows']+=len(w);checks['ledger_fills']+=len(f)
        frames.append(w);receipts[day]=dict(path=str(d/'receipt.json'),sha256=L.sha(d/'receipt.json'),asof=receipt['asof'],counts=receipt['counts'],loader=receipt['loader'])
    w=pd.concat(frames,ignore_index=True);totals={};byday={day:{} for day in L.DAYS}
    for case,g in w.groupby('case',sort=False):
        L.require(len(g)==1434 and g.day.nunique()==5,'unequal comparison population')
        totals[case]=summarize(g)
        for day,h in g.groupby('day'):byday[day][case]=summarize(h)
    result=dict(asof=A.stamp(),freeze=freeze,receipts=receipts,checks=checks,totals=totals,by_day=byday,source_sha256=L.sha(__file__),
        limits='Descriptive gross replay accounting; previously consumed dates; no inference or selection; settlement is passive carry after first-minute trading.')
    A.dump(out/'report.json',result)
    pd.DataFrame([dict(case=k,**v) for k,v in totals.items()]).to_csv(out/'strategy_totals.csv',index=False)
    pd.DataFrame([dict(day=d,case=k,**v) for d,z in byday.items() for k,v in z.items()]).to_csv(out/'strategy_by_day.csv',index=False)
    render(result,S.HERE.parent/'handoffs/20261116_strategy_scorecard.md')
    print(json.dumps(dict(asof=result['asof'],checks=checks,totals={k:{f:v[f] for f in ['shares','spread_capture_usd','markout_1000_usd','terminal_pnl_usd','settlement_pnl_usd','settlement_carry_usd']} for k,v in totals.items()}),indent=2))


if __name__=='__main__':run()
