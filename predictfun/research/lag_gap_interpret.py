"""Computed interpretation of the completed median sensitivity; no new replay."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_ev_lag_gap as G


def derive(r):
    a=r['totals']['baseline_40_22'];b=r['totals']['medians_58_37']
    daily={d:v['totals']['medians_58_37']['pnl'] for d,v in r['days'].items()}
    G.L.require(abs(sum(daily.values())-b['pnl'])<1e-8,'daily value reconciliation')
    best=max(daily,key=daily.get)
    return dict(median_pnl_reduction_fraction=(a['pnl']-b['pnl'])/a['pnl'],
        median_volume_ratio=b['shares']/a['shares'],median_value_per_share_ratio=b['cents_per_share']/a['cents_per_share'],
        best_median_date=best,best_median_date_pnl=daily[best],
        other_four_median_dates_pnl=sum(v for d,v in daily.items() if d!=best),
        mean_rung_exposure_ratio=b['live_share']/a['live_share'],
        median_pull_order_ratio=b['pulls']/b['orders'])


if __name__=='__main__':
    p=G.HERE.parent/'reviews/ev_lag_gap_20261116/results_v1/report.json'
    r=json.loads(p.read_text());values=derive(r)
    toy=dict(totals=dict(baseline_40_22=dict(pnl=10.,shares=10.,cents_per_share=100.,live_share=.1),
        medians_58_37=dict(pnl=4.,shares=12.,cents_per_share=100/3,live_share=.2,pulls=2,orders=4)),
        days={'a':dict(totals=dict(medians_58_37=dict(pnl=6.))),
              'b':dict(totals=dict(medians_58_37=dict(pnl=-2.)))})
    z=derive(toy);G.L.require(z['median_pnl_reduction_fraction']==.6 and z['other_four_median_dates_pnl']==-2.,'arithmetic control')
    toy['totals']['medians_58_37']['pnl']=5.;caught=False
    try:derive(toy)
    except ValueError:caught=True
    G.L.require(caught,'corrupt daily reconciliation not refused')
    out=p.parent/'interpretation_v1.json';G.L.require(not out.exists(),'interpretation exists')
    result=dict(report_sha256=G.L.sha(p),source_sha256=G.L.sha(__file__),asof=r['asof'],windows=1434,days=5,
        derived=values,controls=dict(arithmetic_positive_control=True,corrupt_daily_refused=caught),
        limits='Descriptive concentration and ratios only. All windows remain in primary PnL; no post-hoc day exclusion or causal attribution.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
