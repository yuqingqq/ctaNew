"""Descriptive reconciliation and clock-phase probe; no forecast-skill claim.

Primary two-clock replay was frozen before outcomes. The synthetic review-phase
probe is a post-hoc mechanism check motivated by source inspection during the
run; it is not an added empirical strategy candidate.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_ev_lag_gap as G


def first_review(place):
    times=[];w=G.M.toy_lane_window([],[],[])
    def record(k,t,order):
        times.append((float(t),float(t-order['ta'])));return False
    with G.clocks(place,22.):G.L.simulate(w,dec={(0,0):record},shift=0.)
    G.L.require(bool(times),'no toy review')
    return dict(place_ms=place,first_review_ms=times[0][0],active_age_at_first_review_ms=times[0][1])


def compare(a,b):
    for field in ['pnl','shares','orders','pulls','fill_events']:
        G.L.require(abs(a[field]-b[field])<1e-6,'baseline aggregate mismatch')


def run(src,out):
    G.L.require(not out.exists(),'output exists');out.mkdir(parents=True)
    freeze=json.loads((G.HERE.parent/'reviews/ev_lag_gap_20261116/freeze_v1.json').read_text())
    old_path=G.PREVIOUS/'report.json';old=json.loads(old_path.read_text())
    G.L.require(G.L.sha(old_path)==freeze['prior_report_sha256'],'prior report changed')
    rows={d:json.loads((src/d/'receipt.json').read_text()) for d in freeze['days']}
    G.L.require(len({r['source_commit'] for r in rows.values()})==1,'mixed replay source commits')
    for day,r in rows.items():
        G.L.require(r['day']==day and r['source_sha256']==freeze['source_sha256'],'replay identity')
        G.L.require(r['prior_receipt_sha256']==G.L.sha(G.PREVIOUS/f'receipt_{day}_v3.json'),'prior receipt changed')
        G.L.require(r['reference_windows_sha256']==G.L.sha(G.DATA/day/'windows.parquet'),'prior windows changed')
        G.L.require(r['stats']['windows']==r['stats']['baseline_window_parity'],'incomplete baseline parity')
        compare(r['totals']['baseline_40_22'],old['days'][day]['totals']['current']['EV'])
        frame=G.pd.read_parquet(src/day/'windows.parquet')
        G.L.require(not frame.duplicated(['day','m','cfg']).any(),'duplicate output windows')
        G.L.require(set(frame.day)=={day} and set(frame.cfg)=={c[0] for c in G.CASES},'output population mismatch')
        for cfg,group in frame.groupby('cfg'):
            G.L.require(len(group)==r['stats']['windows'],'output window count mismatch')
            compare({k:float(group[k].sum()) for k in ['pnl','shares','orders','pulls','fill_events']},r['totals'][cfg])
    positive=rows[freeze['days'][0]]['totals']['baseline_40_22'];compare(positive,positive)
    bad=dict(positive,pnl=positive['pnl']+1.);caught=False
    try:compare(positive,bad)
    except ValueError:caught=True
    G.L.require(caught,'corrupt aggregate not refused')
    totals={}
    for cfg,_,_ in G.CASES:
        rr=[r['totals'][cfg] for r in rows.values()]
        z={k:sum(r[k] for r in rr) for k in ['windows','pnl','shares','orders','pulls','fill_events','markout1_cents','markout5_cents','imputed_pnl','imputed_shares','pend','pend_shares','qunk','qunk_shares']}
        z.update(cents_per_share=100*z['pnl']/z['shares'],positive_dates=sum(r['pnl']>0 for r in rr),
            live_share=sum(r['live_share']*r['windows'] for r in rr)/z['windows'],max_abs_position=max(r['max_abs_position'] for r in rr))
        totals[cfg]=z
    phase=[first_review(x) for x in [0.,40.,50.,51.,58.]]
    G.L.require([(x['first_review_ms'],x['active_age_at_first_review_ms']) for x in phase]==[(50.,50.),(50.,10.),(50.,0.),(100.,49.),(100.,42.)],'unexpected review phase')
    wrong=all(x['first_review_ms']==50. for x in phase)
    G.L.require(not wrong,'constant first-review assumption not refused')
    cur=old['totals']['current']['EV'];oz=old['totals']['order_zero']['EV'];zero=old['totals']['modeled_zero']['EV'];qr=old['totals']['current']['QR'];measured=totals['medians_58_37']
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_sha256=G.L.sha(__file__),freeze=freeze,days=rows,totals=totals,
        receipt_hashes={d:G.L.sha(src/d/'receipt.json') for d in rows},
        window_hashes={d:G.L.sha(src/d/'windows.parquet') for d in rows},
        daily_median_minus_baseline={d:r['totals']['medians_58_37']['pnl']-r['totals']['baseline_40_22']['pnl'] for d,r in rows.items()},
        pooled=dict(median_minus_baseline=measured['pnl']-cur['pnl'],zero_order_minus_baseline=oz['pnl']-cur['pnl'],zero_feed_additional=zero['pnl']-oz['pnl'],
            full_zero_minus_medians=zero['pnl']-measured['pnl'],full_ev_minus_fill_only_same_environment=cur['pnl']-qr['pnl'],
            baseline_terminal_minus_mark1=cur['pnl']-cur['markout1_cents']/100,median_terminal_minus_mark1=measured['pnl']-measured['markout1_cents']/100,
            full_zero_terminal_minus_mark1=zero['pnl']-zero['markout1_cents']/100),
        review_phase_toy=phase,controls=dict(corrupt_aggregate_refused=caught,constant_review_phase_refused=not wrong,
            baseline_windows=sum(r['stats']['baseline_window_parity'] for r in rows.values()),state_comparisons=sum(r['stats']['state_parity'] for r in rows.values()),
            trajectory_comparisons=sum(r['stats']['trajectory_parity'] for r in rows.values())),
        limits='No new forecast tested. No largest-cause attribution established; different-policy comparison has endogenous exposure. Median-only execution sensitivity, not full physical latency validation. Toy establishes review-phase mechanism, not its share of empirical PnL. Consumed data, no inference or promotion.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(asof=result['asof'],totals=totals,pooled=result['pooled'],review_phase_toy=phase,controls=result['controls']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--src',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.src,a.out)
