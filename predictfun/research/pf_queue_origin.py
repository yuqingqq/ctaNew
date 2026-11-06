"""Where the queue at a new best price comes from (descriptive; user 2026-11-05: "check if they are already there right before our
post or sitting there for quite some time; if they are, we should also use a ladder to post there for some time").
Unit: a best-price change on one side of the Predict book (venue frames, own-date parts, 8 read days) in the first 60 s of a live
window. For the new best price p:
  kind      improve (p inside the previous spread) / from_deeper (p was shown at levels 2-5 in the previous frame) / from_hidden
  s_pre     size at p in the previous frame (while p was not yet the best)
  age_pre   how long p had been shown with size before it became the best (continuous run; capped at the 30 s look-back)
  s100      size at p in the last frame <= +100 ms (~ when our order would go live after seeing the change: 53 ms feed + 2 ms
            decision + 40 ms create)  = the queue ahead of a touch-join order
  ladder    queue ahead of an order that had rested at p since the start of p's run (joined behind the size shown then):
            min size at p over [run start, +100 ms] (FIFO, displayed size caps the queue ahead); ladder_1s = joined 1 s before
Status: p no longer the best at +100 ms (moved) -> excluded from the s100 / ladder rows, counted.
Usage: python3 -m predictfun.research.pf_queue_origin [--selftest]
"""
import sys, glob, json
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
assert all(d not in ('20261028', '20261029', '20261030') and d < '20261104' for d in D8)
LOOK = 30_000.0; JOIN = 100.0; LIVE = (0.0, 60_000.0)
LV = range(1, 6)


def side_events(t, P, S, ws_ms, ask):
    """t (n,), P / S (n, 5) one market, one side, sorted by t. Returns a list of event dicts."""
    assert np.all(np.diff(t) >= 0), 'frames not sorted'
    best = P[:, 0]; out = []
    ch = np.flatnonzero((best[1:] != best[:-1]) & (best[1:] >= 1)) + 1
    for i in ch:
        rel = t[i] - ws_ms
        if not (LIVE[0] <= rel < LIVE[1]):
            continue
        p, pb = best[i], best[i - 1]
        hit = (P[i - 1] == p) & (S[i - 1] > 0)
        s_pre = float(S[i - 1][hit][0]) if hit.any() else 0.0
        if pb < 1:
            kind = 'other'
        elif (p < pb) if ask else (p > pb):
            kind = 'improve'
        else:
            kind = 'from_deeper' if hit.any() else 'from_hidden'
        j0 = int(np.searchsorted(t, t[i] - LOOK, side='left'))
        shown = ((P[j0:i] == p) & (S[j0:i] > 0)).any(axis=1)
        off = np.flatnonzero(~shown)
        rs = j0 + (off[-1] + 1 if len(off) else 0)          # first frame of p's continuous shown run
        capped = len(off) == 0 and i > j0
        age = float(t[i] - t[rs]) if rs < i else 0.0
        k = int(np.searchsorted(t, t[i] + JOIN, side='right')) - 1
        moved = not np.all(best[i:k + 1] == p)
        r = dict(kind=kind, s_pre=s_pre, age_pre=age, capped=capped, moved=moved)
        if not moved:
            sz = np.where(P[rs:k + 1] == p, S[rs:k + 1], 0.0).sum(axis=1)   # size at p, run start .. +100 ms
            r['s100'] = float(sz[-1]); r['ladder'] = float(sz.min())
            j1 = int(np.searchsorted(t, t[i] - 1000.0, side='right')) - 1   # frame shown 1 s before the change
            r['ladder_1s'] = float(sz[j1 - rs:].min()) if j1 >= rs else np.nan
        out.append(r)
    return out


def selftest():
    # ask side: 52 rests at level 2 (100 -> 120) for 2.5 s, 51 clears at t = 500, 52 becomes the best and grows to 300 by 560
    t = np.array([-2000.0, -1000.0, 500.0, 560.0, 700.0])
    P = np.array([[51, 52, -1, -1, -1], [51, 52, -1, -1, -1], [52, -1, -1, -1, -1], [52, -1, -1, -1, -1], [52, -1, -1, -1, -1]], float)
    S = np.array([[200, 100, 0, 0, 0], [200, 120, 0, 0, 0], [120, 0, 0, 0, 0], [300, 0, 0, 0, 0], [280, 0, 0, 0, 0]], float)
    e = side_events(t, P, S, 0.0, True)
    assert len(e) == 1 and e[0]['kind'] == 'from_deeper' and e[0]['s_pre'] == 120 and e[0]['age_pre'] == 2500, e
    assert e[0]['s100'] == 300 and e[0]['ladder'] == 100 and e[0]['ladder_1s'] == 120, e
    # control: 52 never shown before -> from_hidden, nothing resting, ladder = the size of its first frame
    P2 = P.copy(); P2[:2, 1] = -1; S2 = S.copy(); S2[:2, 1] = 0
    e2 = side_events(t, P2, S2, 0.0, True)
    assert e2[0]['kind'] == 'from_hidden' and e2[0]['s_pre'] == 0 and e2[0]['age_pre'] == 0 and e2[0]['ladder'] == 120, e2
    # bid improve inside the spread
    e3 = side_events(np.array([0.0, 10.0]), np.array([[50, 49, -1, -1, -1], [51, 50, 49, -1, -1]], float),
                     np.array([[10, 10, 0, 0, 0], [5, 10, 10, 0, 0]], float), 0.0, False)
    assert e3[0]['kind'] == 'improve' and e3[0]['s_pre'] == 0, e3
    # refusal: unsorted frames
    try:
        side_events(t[::-1].copy(), P, S, 0.0, True); raise SystemExit('selftest: unsorted input accepted')
    except AssertionError:
        pass
    print('selftest ok')


def day(d):
    cols = ['mid', 't_ms'] + [f'{s}{i}_{k}' for s in 'ba' for i in LV for k in 'ps']
    B = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f'{io.OUT}/step1/{d}/books/part_{d}_*.parquet'))], ignore_index=True)
    ws = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid').ws * 1000.0
    B['w'] = B.mid.map(ws)
    B = B[B.w.notna() & (B.t_ms >= B.w - LOOK) & (B.t_ms <= B.w + LIVE[1] + 2000)].sort_values(['mid', 't_ms'], kind='stable')
    ev = []
    for mid, g in B.groupby('mid', sort=False):
        t = g.t_ms.values.astype(float); w = float(g.w.values[0])
        for ask, s in ((False, 'b'), (True, 'a')):
            P = g[[f'{s}{i}_p' for i in LV]].values.astype(float); S = g[[f'{s}{i}_s' for i in LV]].values.astype(float)
            for r in side_events(t, P, S, w, ask):
                r['day'] = d; r['side'] = 'ask' if ask else 'bid'; ev.append(r)
    return ev


def main():
    selftest()
    E = pd.DataFrame([r for d in D8 for r in day(d)])
    E.to_parquet(f'{io.OUT}/step4/queue_origin_events.parquet')
    rep = dict(days=D8, events=len(E), kinds=E.kind.value_counts().to_dict(), moved_by_100ms=float(E.moved.mean()),
               age_capped=int(E.capped.sum()), by_kind={})
    print(f"best-price changes in 0-60 s, 8 read days: {len(E)}; kinds {rep['kinds']}; p no longer best at +100 ms {E.moved.mean():.1%}")
    for k, g in E.groupby('kind'):
        h = g[~g.moved]
        r = dict(n=len(g), share=len(g) / len(E), moved=float(g.moved.mean()),
                 s_pre_median=float(g.s_pre.median()), age_pre_p25_p50_p75=[float(x) for x in g.age_pre.quantile([.25, .5, .75])],
                 s100_median=float(h.s100.median()), pre_share_of_s100=float(np.minimum(h.s_pre, h.s100).sum() / max(h.s100.sum(), 1e-9)),
                 ladder_median=float(h.ladder.median()), ladder_1s_median=float(h.ladder_1s.median()),
                 ladder_1s_available=float(h.ladder_1s.notna().mean()),
                 ladder_lt_half_s100=float((h.ladder < 0.5 * h.s100).mean()))
        rep['by_kind'][k] = r
        print(f"{k:12s} n {r['n']:7d} ({r['share']:.0%}) moved {r['moved']:.0%} | size before it became best: median {r['s_pre_median']:.0f}, "
              f"shown for p25/50/75 {r['age_pre_p25_p50_p75'][0]:.0f}/{r['age_pre_p25_p50_p75'][1]:.0f}/{r['age_pre_p25_p50_p75'][2]:.0f} ms | "
              f"queue at +100 ms median {r['s100_median']:.0f}, already resting before {r['pre_share_of_s100']:.0%} | ahead of a ladder order "
              f"since run start {r['ladder_median']:.0f}, since 1 s before {r['ladder_1s_median']:.0f} ({r['ladder_1s_available']:.0%} of levels "
              f"shown >= 1 s); ladder < half the touch-join queue {r['ladder_lt_half_s100']:.0%}")
    io.write_json(f'{io.OUT}/step4/report_queue_origin.json', rep)


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
