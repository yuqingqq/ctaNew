"""When does the Binance futures best level shrink before the early competitor arrivals, and how far does that lead the futures mid move?
Diagnostic on four consumed days. Events from lead_venues.py (early class: no futures mid change in (A - 250, A]); placebos at A - 10 s
and A + 10 s. Per event: the size of the best level on the side the move takes out (ask for up, bid for down) at A - 500 ms is the base;
the first time in (A - 500, A + 300] it is at or below 50 % of the base at the same price (or the level is gone) = the depletion time.
Reported: share of events with a depletion before A; A - depletion (their reaction = detection + decision + network to Predict);
the futures mid move (first change our way, lead_venues tf_after) - depletion; the same shares at placebo times.
Output (write-once): lead_depletion_timing.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh lead_depl -- python3 predictfun/reviews/take_exit_venue_20261125/lead_depletion_timing.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import lead_flow as LF  # noqa: E402

LV, C, TR = LF.LV, LF.C, LF.TR
BASE, WIN_END, FRAC = 500., 300., 0.5
SAME_PRICE_ONLY = False
if len(sys.argv) > 1 and sys.argv[1] == 'strict':                       # the lead_flow definition: base 300 ms, same price, before A
    BASE, WIN_END, SAME_PRICE_ONLY = 300., 0., True
SFX = '_strict' if SAME_PRICE_ONLY else ''
SHIFTS = {'ev': 0., 'pm': -10000., 'pp': 10000.}


def depletion(bt, b, bq, ask, aq, a, d):
    """First time in (a - BASE, a + WIN_END] the take-out side's best level is at or below FRAC of its size at a - BASE at the same price,
    or that price is no longer the best on that side. None if neither happens or the base is not available."""
    j0 = int(np.searchsorted(bt, a - BASE, 'right') - 1)
    if j0 < 0:
        return None
    px, q0 = (ask[j0], aq[j0]) if d == 1 else (b[j0], bq[j0])
    if not q0 > 0:
        return None
    j1 = int(np.searchsorted(bt, a + WIN_END, 'right'))
    p, q = (ask[j0 + 1:j1], aq[j0 + 1:j1]) if d == 1 else (b[j0 + 1:j1], bq[j0 + 1:j1])
    hit = np.flatnonzero(((p == px) & (q <= FRAC * q0)) if SAME_PRICE_ONLY else ((p != px) | (q <= FRAC * q0)))
    return float(bt[j0 + 1 + hit[0]]) if len(hit) else None


def main():
    fn = H / f'lead_depletion_timing{SFX}.json'
    C.require(not fn.exists(), f'{fn.name} exists (write-once)')
    E = pd.read_parquet(LV.OUT / 'lead_events.parquet', columns=['day', 'A', 'd', 'cls', 'tf_after'])
    dep = {s: np.full(len(E), np.nan) for s in SHIFTS}
    skipped = collections.Counter()
    for day in LV.DAYS:
        lo, hi = LV.day_ms(day)
        pairs = collections.defaultdict(list)
        for i in np.flatnonzero(E.day.values == day):
            for s, sh in SHIFTS.items():
                a = E.A.values[i] + sh
                if a - BASE < lo + 1000. or a + WIN_END >= hi:
                    skipped['window_outside_day'] += 1; continue
                pairs[int((a - lo) // 3600000)].append((i, s, a))
        carry = None
        for h in range(24):
            D, carry = LF.load_hour(day, h, carry)
            bt, b, bq, ask, aq = D[3:]
            for i, s, a in pairs.get(h, []):
                if a + WIN_END > lo + (h + 1) * 3600000:                        # window runs into the next hour: read it there
                    pairs[h + 1].append((i, s, a)) if h < 23 else skipped.update(['window_outside_day'])
                    continue
                t = depletion(bt, b, bq, ask, aq, a, int(E.d.values[i]))
                dep[s][i] = np.nan if t is None else t - a                     # relative to the (placebo) arrival
        print(day, 'done', flush=True)
    out = dict(asof='2026-11-25', base_ms_before_arrival=BASE, window_end_ms_after_arrival=WIN_END, frac=FRAC, same_price_only=SAME_PRICE_ONLY, skipped=dict(skipped), groups={})
    for c in ('early', 'late'):
        X = E.cls.values == c
        ev = dep['ev'][X]; pl = np.r_[dep['pm'][X], dep['pp'][X]]
        before = np.isfinite(ev) & (ev <= 0); pbefore = np.isfinite(pl) & (pl <= 0)
        react = -ev[before]
        lead_mid = (E.tf_after.values[X] - (E.A.values[X] + ev))[before & np.isfinite(E.tf_after.values[X])]
        out['groups'][c] = dict(events=int(X.sum()), depleted_before_arrival=float(before.mean()), placebo_rate=float(pbefore.mean()),
                                lift=float(before.mean() / pbefore.mean()) if pbefore.mean() > 0 else None,
                                reaction_ms_arrival_minus_depletion_p10_p25_p50_p75_p90=[float(np.quantile(react, q)) for q in (.1, .25, .5, .75, .9)] if len(react) else None,
                                depletion_leads_futures_mid_ms_p25_p50_p75=[float(np.quantile(lead_mid, q)) for q in (.25, .5, .75)] if len(lead_mid) else None)
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
