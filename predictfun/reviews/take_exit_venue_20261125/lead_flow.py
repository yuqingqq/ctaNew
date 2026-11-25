"""Binance futures order flow before the early competitor arrivals (no futures mid change yet). Diagnostic on four consumed days.
Events from lead_venues.py (OUT/lead_events.parquet): arrival A at Predict, direction d, class early / late; placebos at A - 10 s and A + 10 s.
In (A - 300, A] from the Binance futures streams (transaction time T): aggressive volume in direction d (buyer-initiated for up) and against,
their imbalance, and the depletion of the best level on the side the move would take out (ask for up, bid for down) as the ratio of its
size at A to its size at A - 300 ms while its price is unchanged. Output (write-once): lead_flow.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh lead_flow -- python3 predictfun/reviews/take_exit_venue_20261125/lead_flow.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import lead_venues as LV  # noqa: E402

TR, C, DATA = LV.TR, LV.C, LV.DATA
W = 300.
SHIFTS = {'ev': 0., 'pm': -10000., 'pp': 10000.}


def load_hour(day, h, carry):
    """Trades and best-level updates of one hour (transaction time), with the previous hour's last second carried in front."""
    ft, fb = DATA / f'mm_hf/raw/trade/BTCUSDT/{day}_{h:02d}.csv.gz', DATA / f'mm_hf/raw/bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz'
    tr = pd.read_csv(ft, header=None, usecols=[2, 5, 6], names=['t', 'q', 'bm'], dtype={2: np.int64, 5: np.float64, 6: np.int8}) if ft.exists() else pd.DataFrame(columns=['t', 'q', 'bm'])
    bt = pd.read_csv(fb, header=None, usecols=[2, 4, 5, 6, 7], names=['t', 'b', 'bq', 'a', 'aq'], dtype={2: np.int64, 4: np.float64, 5: np.float32, 6: np.float64, 7: np.float32}) if fb.exists() else pd.DataFrame(columns=['t', 'b', 'bq', 'a', 'aq'])
    if carry is not None:
        tr, bt = pd.concat([carry[0], tr]), pd.concat([carry[1], bt])
    tr, bt = tr.sort_values('t', kind='stable'), bt.sort_values('t', kind='stable')
    lo, _ = LV.day_ms(day); h1 = lo + (h + 1) * 3600000
    nxt = (tr[tr.t >= h1 - 1000], bt[bt.t >= h1 - 1000])
    t = tr.t.to_numpy(float); q = tr.q.to_numpy(float); buy = tr.bm.to_numpy() == 0           # buyer-initiated when the buyer is not the maker
    cb, cs = np.r_[0., np.cumsum(np.where(buy, q, 0.))], np.r_[0., np.cumsum(np.where(buy, 0., q))]
    return (t, cb, cs, bt.t.to_numpy(float), bt.b.to_numpy(float), bt.bq.to_numpy(float), bt.a.to_numpy(float), bt.aq.to_numpy(float)), nxt


def feats(D, a, d):
    t, cb, cs, bt, b, bq, ask, aq = D
    i0, i1 = np.searchsorted(t, a - W, 'right'), np.searchsorted(t, a, 'right')
    vb, vs = cb[i1] - cb[i0], cs[i1] - cs[i0]
    vdir, vopp = (vb, vs) if d == 1 else (vs, vb)
    j0, j1 = np.searchsorted(bt, a - W, 'right') - 1, np.searchsorted(bt, a, 'right') - 1
    dep = np.nan
    if j0 >= 0 and j1 >= 0:
        if d == 1 and ask[j0] == ask[j1] and aq[j0] > 0:
            dep = aq[j1] / aq[j0]
        elif d == -1 and b[j0] == b[j1] and bq[j0] > 0:
            dep = bq[j1] / bq[j0]
    return vdir, vopp, dep


def main():
    fn = H / 'lead_flow.json'
    C.require(not fn.exists(), 'lead_flow.json exists (write-once)')
    E = pd.read_parquet(LV.OUT / 'lead_events.parquet', columns=['day', 'A', 'd', 'cls'])
    cols = {f'{k}_{s}': np.full(len(E), np.nan) for k in ('vdir', 'vopp', 'dep') for s in SHIFTS}
    skipped = collections.Counter()
    for day in LV.DAYS:
        lo, hi = LV.day_ms(day)
        pairs = collections.defaultdict(list)                                  # hour -> (event, shift) whose window ends in it
        for i in np.flatnonzero(E.day.values == day):
            for s, sh in SHIFTS.items():
                a = E.A.values[i] + sh
                if a - W < lo + 1000. or a >= hi:                               # needs a day not read (the day before / after)
                    skipped['window_outside_day'] += 1; continue
                pairs[int((a - lo) // 3600000)].append((i, s, a))
        carry = None
        for h in range(24):
            D, carry = load_hour(day, h, carry)
            for i, s, a in pairs.get(h, []):
                v = feats(D, a, int(E.d.values[i]))
                cols[f'vdir_{s}'][i], cols[f'vopp_{s}'][i], cols[f'dep_{s}'][i] = v
            del D
        print(day, 'done', flush=True)
    for k, v in cols.items():
        E[k] = v
    out = dict(asof='2026-11-25', window_ms=W, events=dict(collections.Counter(E.cls)), skipped=dict(skipped), groups={})
    for name, mask, sfx in (('early', E.cls == 'early', ['ev']), ('late', E.cls == 'late', ['ev']), ('placebo', E.cls.notna(), ['pm', 'pp'])):
        vd = np.concatenate([E.loc[mask, f'vdir_{s}'].values for s in sfx]); vo = np.concatenate([E.loc[mask, f'vopp_{s}'].values for s in sfx])
        keep = np.isfinite(vd) & np.isfinite(vo)
        dp = np.concatenate([E.loc[mask, f'dep_{s}'].values for s in sfx])
        vd, vo, dp = vd[keep], vo[keep], dp[keep]
        tot = vd + vo; imb = np.where(tot > 0, (vd - vo) / np.where(tot > 0, tot, 1.), np.nan)
        out['groups'][name] = dict(
            n=int(len(vd)), any_aggressive_volume=float(np.mean(tot > 0)),
            vol_dir_btc_p50_p75_p90=[float(np.quantile(vd, q)) for q in (.5, .75, .9)], vol_opp_btc_p50_p75_p90=[float(np.quantile(vo, q)) for q in (.5, .75, .9)],
            imbalance_p25_p50_p75=[float(np.nanquantile(imb, q)) for q in (.25, .5, .75)],
            share_imbalance_ge_0_5=float(np.nanmean(imb >= .5)), share_dir_vol_ge_1btc=float(np.mean(vd >= 1.)),
            depletion_measured=float(np.mean(np.isfinite(dp))), depletion_ratio_p25_p50_p75=[float(np.nanquantile(dp, q)) for q in (.25, .5, .75)],
            share_best_level_halved=float(np.nanmean(dp <= .5)))
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
