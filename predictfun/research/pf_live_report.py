"""Report for the live-deviation tests (run_live_dev.sh; request from the pmmmrust execution-port session, 2026-11-02).

Six read days 10-23..27, 10-31; C2's model spec (preds_lodo6, out-of-day); tokyo20, E2, fill model v2, mid timing.
  1 no amend (XNA) and removal lock (XL<ms>A all cancels / XL<ms>R A1 size-down only), P and S, vs the amend baseline
  2 minimum order value 1 USD: share of baseline P fills / shares / P&L from orders under 1 USD, and the XMIN1 reruns
  3 request rate (place + cancel + amend, at decision time) per UTC second / minute while quoting, P baseline vs XNA
  4 total-loss cap 150 USD, carried across the six days in date order: refusals, halts, cost vs the same arm uncapped;
    plus every-start-window analysis on the uncapped live-like arm (first bind = realised since start + the window's
    worst case at a place attempt < -150; until the first bind the capped and uncapped paths are identical)
Usage: python3 -m predictfun.research.pf_live_report
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
D = f'{io.OUT}/step4/live_tokyo20_custom_e2_preds_lodo6_livedev'
P = 'MICRO+G4+G2+G6+G5+G3+G1+M025'; S = 'MICRO+G5+G1+M025'; A1 = '/mid@A1:5:25:0'
SUM = {d: json.load(open(f'{D}/{d}/summary.json')) for d in DAYS}


def stem(name):
    return name.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def arm(base, x):
    return f'{base}{x}{A1}'


def fills(name):
    return pd.concat([pd.read_parquet(f'{D}/{d}/fills/{stem(name)}.parquet').assign(day=d) for d in DAYS], ignore_index=True)


def wins(name):
    return pd.concat([pd.read_parquet(f'{D}/{d}/win/{stem(name)}.parquet').assign(day=d) for d in DAYS], ignore_index=True).sort_values('ws')


def reqs(name):
    return pd.concat([pd.read_parquet(f'{D}/{d}/requests/{stem(name)}.parquet').assign(day=d) for d in DAYS], ignore_index=True)


def row(name):
    v = [SUM[d]['arms'][name] for d in DAYS]; sh = sum(x['shares'] for x in v); g = [x['gross_usd'] for x in v]
    return dict(by_day=g, sum=round(sum(g), 2), pos=sum(x > 0 for x in g), fills=sum(x['fills'] for x in v), shares=round(sh),
                mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3), inv_p95_max=max(x['abs_net_end_p95'] for x in v),
                zombie_fills=sum(x['live']['zombie_fills'] for x in v), requests={k: sum(x['live']['requests'].get(k, 0) for x in v) for k in ('place', 'cancel', 'amend')})


def vs(a, b):
    d = [round(x - y, 2) for x, y in zip(a['by_day'], b['by_day'])]
    return dict(by_day=d, sum=round(sum(d), 2), pos=sum(x > 0 for x in d))


out = dict(days=DAYS, model='preds_lodo6 (C2 spec, out-of-day)', settings='tokyo20, E2, fill v2, mid, MICRO, A1:5:25:0')
# 1 no amend + lock
T1 = {}
for lab, base, xs in (('P', P, ['', '+XNA', '+XNA+XL500A', '+XNA+XL1000A', '+XNA+XL500R', '+XNA+XL1000R']),
                      ('S', S, ['', '+XNA', '+XNA+XL500A', '+XNA+XL1000A'])):
    ref = row(arm(base, ''))
    for x in xs:
        r = row(arm(base, x)); r['vs_amend_baseline'] = vs(r, ref); T1[f'{lab}{x or " amend (sim)"}'] = r
out['t1_no_amend_lock'] = T1
# 2 minimum order value
f = fills(arm(P, ''))
f['notional'] = np.where(f.side == 0, f.pt * f.osz, (100 - f.pt) * f.osz) / 100.0
small = f.notional < 1.0 - 1e-9
out['t2_min_order'] = dict(
    baseline_from_small_orders=dict(fills=int(small.sum()), fills_share=round(float(small.mean()), 4), shares=round(float(f.q[small].sum())),
                                    shares_share=round(float(f.q[small].sum() / f.q.sum()), 4), settle_usd=round(float(f.settle[small].sum()), 2),
                                    settle_share_of_total=round(float(f.settle[small].sum() / f.settle.sum()), 4),
                                    mo1_c_sh=round(float(100 * f.mo1[small].sum() / max(f.q[small].sum(), 1e-9)), 3),
                                    bid_side_share=round(float((f.side[small] == 0).mean()), 3) if small.any() else None),
    rerun_amend_min1=dict(row(arm(P, '+XMIN1')), vs_amend=vs(row(arm(P, '+XMIN1')), row(arm(P, '')))),
    rerun_noamend_min1=dict(row(arm(P, '+XNA+XMIN1')), vs_noamend=vs(row(arm(P, '+XNA+XMIN1')), row(arm(P, '+XNA')))))
# 3 request rate
T3 = {}
for lab, name in (('P amend (sim)', arm(P, '')), ('P no-amend', arm(P, '+XNA')), ('P no-amend + min 1 USD', arm(P, '+XNA+XMIN1')),
                  ('S no-amend', arm(S, '+XNA'))):
    q = reqs(name); W = wins(name)
    sec = (q.t_ms // 1000).astype(np.int64); mn = (q.t_ms // 60000).astype(np.int64)
    # quoting seconds/minutes = [ws, ws + 240 s) of every replayed window (zero-request bins included)
    qs = np.concatenate([np.arange(int(w // 1000), int(w // 1000) + 240) for w in W.ws.values])
    qm = np.concatenate([np.arange(int(w // 60000), int(w // 60000) + 4) for w in W.ws.values])
    cs = pd.Series(sec).value_counts().reindex(qs, fill_value=0).values
    cm = pd.Series(mn).value_counts().reindex(qm, fill_value=0).values
    pw = pd.Series(mn).value_counts().reindex(qm, fill_value=0).groupby(np.repeat(np.arange(len(W)), 4)).max().values
    T3[lab] = dict(requests=int(len(q)), by_kind={k: int(c) for k, c in q.kind.value_counts().items()},
                   per_second=dict(mean=round(float(cs.mean()), 2), p50=float(np.percentile(cs, 50)), p90=float(np.percentile(cs, 90)),
                                   p99=float(np.percentile(cs, 99)), max=int(cs.max()), share_gt4=round(float((cs > 4).mean()), 3)),
                   per_minute=dict(mean=round(float(cm.mean()), 1), p50=float(np.percentile(cm, 50)), p90=float(np.percentile(cm, 90)),
                                   p99=float(np.percentile(cm, 99)), max=int(cm.max()), share_gt240=round(float((cm > 240).mean()), 3)),
                   window_peak_minute=dict(p50=float(np.percentile(pw, 50)), p90=float(np.percentile(pw, 90)), max=int(pw.max()),
                                           windows_peak_gt240=round(float((pw > 240).mean()), 3)))
out['t3_request_rate'] = T3
# 4 loss cap
T4 = {}
for lab, capped, free in (('P amend + cap', arm(P, '+XCAP150'), arm(P, '')), ('P no-amend + min 1 USD + cap', arm(P, '+XNA+XMIN1+XCAP150'), arm(P, '+XNA+XMIN1'))):
    c, u = row(capped), row(free); W = wins(capped)
    T4[lab] = dict(capped=c, uncapped_sum=u['sum'], cost=vs(c, u), refused_events=int(W.refused.sum()), windows_with_refusal=int((W.refused > 0).sum()),
                   halted_windows=int(W.halted.sum()), halted_at=SUM[DAYS[-1]]['arms'][capped]['live']['cap_state_end']['halted_at'],
                   realised_end=round(SUM[DAYS[-1]]['arms'][capped]['live']['cap_state_end']['realised'], 2))
W = wins(arm(P, '+XNA+XMIN1')).reset_index(drop=True)            # uncapped live-like arm, every start window
pnl = W.pnl.values; mw = np.nan_to_num(W.min_worst.values, nan=np.inf); n = len(W); cum = np.r_[0.0, np.cumsum(pnl)]
first = np.full(n, -1); halt = np.full(n, -1)
for s in range(n):
    rb = cum[s:n] - cum[s]                                       # realised before each window w >= s
    b = np.flatnonzero(rb + mw[s:] < -150.0); h = np.flatnonzero(rb + pnl[s:] <= -150.0)
    first[s] = b[0] if len(b) else -1; halt[s] = h[0] if len(h) else -1
dd = float(np.min(cum - np.maximum.accumulate(cum)))
T4['every_start_window (P no-amend + min 1 USD, uncapped path)'] = dict(
    starts=n, bind_ever=round(float((first >= 0).mean()), 3), bind_within_288_windows=round(float(((first >= 0) & (first < 288)).mean()), 3),
    halt_ever=round(float((halt >= 0).mean()), 3), halt_within_288_windows=round(float(((halt >= 0) & (halt < 288)).mean()), 3),
    windows_to_first_bind_p50=float(np.median(first[first >= 0])) if (first >= 0).any() else None,
    worst_window_min_worst=round(float(np.min(W.min_worst)), 2), worst_window_pnl=round(float(pnl.min()), 2),
    max_drawdown_usd=round(dd, 2), note='first bind is exact (paths identical until it); what happens after a bind needs the capped replay')
out['t4_loss_cap'] = T4
out['data_diag'] = {d: SUM[d]['live_data_diag'] for d in DAYS}
io.write_json(f'{D}/report_live.json', out)

print('1 NO AMEND / LOCK (6-day sum gross $, vs amend baseline, +1 s c/sh, zombie fills)')
for k, r in T1.items():
    print(f"  {k:26s} {r['sum']:9.2f} vs {r['vs_amend_baseline']['sum']:8.2f} ({r['vs_amend_baseline']['pos']}/6) sh {r['shares']:7d} mo1 {r['mo1_c_sh']:6.3f} z {r['zombie_fills']}")
print('2 MIN ORDER', json.dumps(out['t2_min_order']['baseline_from_small_orders']))
print('  rerun amend+min1', out['t2_min_order']['rerun_amend_min1']['sum'], out['t2_min_order']['rerun_amend_min1']['vs_amend'])
print('  rerun noamend+min1', out['t2_min_order']['rerun_noamend_min1']['sum'], out['t2_min_order']['rerun_noamend_min1']['vs_noamend'])
print('3 REQUESTS'); [print('  ', k, json.dumps(v)) for k, v in T3.items()]
print('4 LOSS CAP'); [print('  ', k, json.dumps({kk: vv for kk, vv in v.items() if kk != 'capped'})) for k, v in T4.items()]
print('DIAG'); [print('  ', d, v) for d, v in out['data_diag'].items()]
