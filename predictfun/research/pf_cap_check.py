"""Would the live $150 total-loss cap have bound in the round-i replays of pm_ladder_v1? (read-only; no replay)
The frozen arm has no loss cap (the port adds it live-only, pmmmrust docs/PREDICT_FUN_PLAN.md). The engine's XCAP rule refuses a place
when realised (all earlier windows, start-time order) + this window's worst case < -cap. Round i ran without XCAP, so its path is the
capped path up to the first refusal: the first window with realised_before + min_worst < -cap is exactly where XCAP150 would start
refusing (min_worst is recorded at every place attempt, capped or not: pf_step4_ladder.py l. 958-959 / 1048-1049). Past that window the
paths differ and only a capped replay tells the cost. Realised = settlement P&L (win.pnl), not mark@stop.
Days in order: 10-23..27, 10-31, 11-01..03 (10-28..30 sealed; the path is joined across the gap).
Selftest (positive and negative control) runs first.
Usage: python3 -m predictfun.research.pf_cap_check
"""
import sys
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102', '20261103']
STEM = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'
CAP = 150.0; MIN_ORDER = 1.0; PER_DAY = 288


def first_refusal(pnl, mw, cap=CAP):
    """Index of the first window whose place would be refused (realised before + worst case < -cap), else None."""
    r0 = np.r_[0.0, np.cumsum(pnl)][:-1]; b = np.nonzero(r0 + np.nan_to_num(mw, nan=0.0) < -cap)[0]
    return int(b[0]) if len(b) else None


def first_halt(pnl, cap=CAP):
    """Port halt (headroom below the smallest order): realised <= -(cap - 1). On the uncapped path only."""
    h = np.nonzero(np.cumsum(pnl) <= -(cap - MIN_ORDER))[0]
    return int(h[0]) if len(h) else None


def mdd(x):
    c = np.r_[0.0, np.cumsum(x)]; return float((c - np.maximum.accumulate(c)).min())


def selftest():
    assert first_refusal(np.array([-100.0, 0.0]), np.array([-10.0, -60.0])) == 1        # positive: -100 - 60 < -150 at window 1
    assert first_refusal(np.array([-100.0, 0.0]), np.array([-10.0, -50.0])) is None     # negative: -150 is not < -150
    assert first_refusal(np.array([0.0]), np.array([np.nan])) is None                   # no place attempt -> no refusal
    assert first_halt(np.array([-100.0, -49.0])) == 1 and first_halt(np.array([-100.0, -48.0])) is None
    assert mdd(np.array([10.0, -30.0, 5.0, -10.0])) == -35.0


def main():
    selftest(); out = dict(cap_usd=CAP, days=DAYS, as_of=pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%MZ'), timings={})
    for tm in ('J', 'X'):
        w = pd.concat([pd.read_parquet(f'{io.OUT}/step4/ladder_tokyoreal_custom_e2_preds_t60k_l{tm}i/{d}/win/{STEM}').assign(day=d) for d in DAYS])
        w = w.sort_values('ws').reset_index(drop=True); n = len(w)
        if w.halted.any() or (w.refused != 0).any():
            sys.exit(f'REFUSED: {tm} round-i run has cap activity (expected none)')
        pnl, mw = w.pnl.values, w.min_worst.values
        per_day = {d: dict(settle=round(float(g.pnl.sum()), 1), mark_at_stop=round(float(g.mtm_stop.sum()), 1), worst_window=round(float(g.pnl.min()), 1),
                           intraday_max_drawdown=round(mdd(g.pnl.values), 1), worst_case_p50=round(float(g.min_worst.median()), 1),
                           worst_case_min=round(float(g.min_worst.min()), 1)) for d, g in w.groupby('day')}
        ref = [first_refusal(pnl[s:], mw[s:]) for s in range(n)]; hlt = [first_halt(pnl[s:]) for s in range(n)]
        def share(a, k):                       # starts with >= k windows of tape left
            idx = [s for s in range(n) if s + k <= n]; return dict(starts=len(idx), pct=round(100.0 * np.mean([a[s] is not None and a[s] < k for s in idx]), 1))
        starts00 = {}
        for d in DAYS:
            s = int(np.nonzero(w.day.values == d)[0][0]); r = ref[s]
            starts00[d] = None if r is None else dict(window=r, day=w.day.values[s + r], realised_before=round(float(pnl[s:s + r].sum()), 1),
                                                     worst_case=round(float(mw[s + r]), 1))
        out['timings'][tm] = dict(windows=n, settle_sum=round(float(pnl.sum()), 1), mark_at_stop_sum=round(float(w.mtm_stop.sum()), 1),
                                  max_drawdown_all_days=round(mdd(pnl), 1), worst_case_pct=dict(zip(('p50', 'p10', 'min'), np.round(np.nanpercentile(mw, [50, 10, 0]), 1).tolist())),
                                  refusal_within_1d=share(ref, PER_DAY), refusal_within_3d=share(ref, 3 * PER_DAY),
                                  halt_within_1d_uncapped=share(hlt, PER_DAY), first_refusal_from_00Z=starts00, per_day=per_day)
    io.write_json(f'{io.OUT}/step4/report_cap150_round_i.json', out)
    for tm, T in out['timings'].items():
        print(tm, {k: v for k, v in T.items() if k not in ('per_day', 'first_refusal_from_00Z')})
        print(' first refusal from 00Z:', T['first_refusal_from_00Z'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
