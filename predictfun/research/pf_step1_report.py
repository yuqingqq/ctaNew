"""Step 1 report across days: completeness, trade timing, book structure, trade activity.
Usage: python3 -m predictfun.research.pf_step1_report 20261023 20261024 20261025"""
import sys, json
import numpy as np, pandas as pd
from . import pf_io as io

days = sys.argv[1:]
rep = dict(days=days, mode='EXPLORATORY', per_day={})
Ws = []
for d in days:
    S1 = f'{io.OUT}/step1/{d}'
    s = json.load(open(f'{S1}/summary.json')); t = json.load(open(f'{S1}/timing.json')); iso = json.load(open(f'{S1}/isolated_recon.json'))
    W = pd.read_parquet(f'{S1}/windows.parquet'); W['day'] = d; Ws.append(W)
    rep['per_day'][d] = dict(
        windows=s['n_windows_registry'], with_live_frames=s['n_windows_with_live_frames'], with_outcome=s['n_windows_with_outcome'],
        with_start_price=s['n_windows_with_start_price'], frames=s['n_frames'], live_legs=s['legs_by_phase'].get('live'),
        pre_legs=s['legs_by_phase'].get('pre'), live_makers=s['n_distinct_makers_live'], rest_first_seen_lag_s=s['first_seen_lag_s_live'],
        timed_leg_share=t['timed_leg_share'], timed_volume_share=t['timed_volume_share'], timing_offset_ms=t['offset_ms_pct'],
        offset_within_500ms=t['share_offset_in_m500_500'], iso_found_same=iso['same']['share_found'],
        collector_events={k: v for k, v in s['collector_events'].items() if k in ('MATCH_POLL_INCOMPLETE', 'WS_FAILURE', 'CATEGORY_FETCH_FAILED', 'RESNAPSHOT_FETCHED')})
W = pd.concat(Ws, ignore_index=True)
q = lambda c: dict(zip(['p10', 'p50', 'p90'], W[c].quantile([.1, .5, .9]).round(3).tolist()))
rep['structure_per_window'] = {c: q(c) for c in (
    'frames_per_s', 'max_frame_gap_s', 'tw_two_sided_ok', 'tw_locked', 'tw_one_or_empty', 'tw_spread1', 'tw_spread2', 'tw_spread3p',
    'tw_gap_behind_bid', 'tw_gap_behind_ask', 'med_touch_bid_sh', 'med_touch_ask_sh', 'live_shares', 'live_yes_notional', 'live_makers',
    'top_maker_share', 'aggr_buy_share', 'pre_shares', 'start_known_after_ws_s', 'outcome_known_after_we_s')}
rep['by_minute'] = {f'{b}-{b + 60}s': dict(tw_spread1=round(float(W[f'tw_spread1_{b}'].mean()), 3), shares_mean=round(float(W[f'shares_{b}'].mean()), 0)) for b in range(0, 300, 60)}
rep['outcomes'] = W.outcome.value_counts().to_dict()
rep['n_windows'] = int(len(W)); rep['windows_max_gap_over_5s'] = int((W.max_frame_gap_s > 5).sum())
io.write_json(f'{io.OUT}/step1/report_{"_".join(days)}.json', rep)
print(json.dumps(rep, indent=1, default=str))
