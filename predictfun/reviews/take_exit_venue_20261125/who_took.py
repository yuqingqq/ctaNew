"""Who removed the quotes our 172 ms takes missed? Diagnostic on consumed days (no strategy change, no selection).
For every take sent in the stage-1 172 ms BASE ledgers (408 complete markets), the level it aimed at (the known best on the opposite side
at the decision, = the take's limit) is followed through the venue frames from the decision's known frame to the take's arrival: when it
vanished from the book (venue time), how much of its shown size printed as public trades at or through that price before it vanished
(other takers), and the rest (removed without a trade: makers pulled). Trade times carry anchor uncertainty, so two windows are read:
tight = [known frame, vanish + 100 ms], wide = [known frame - 250 ms, vanish + 500 ms]. Output (write-once): who_took.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh who_took -- python3 predictfun/reviews/take_exit_venue_20261125/who_took.py
"""
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import venue_common as V  # noqa: E402

F, OD, MM, TR, C = V.F, V.OD, V.MM, V.TR, V.C
S1D = F.X.DATA / 'take_exit_venue_s1_d172_20261125'
WIN = dict(tight=(0., 100.), wide=(-250., 500.))
EPS = V.EPS


def level(row, side, price):
    """Shown size at price on book side (0 bid, 1 ask) in one frame row (P5 / S5 layout: prices 2+10s..6+10s, sizes 7+10s..11+10s)."""
    b = 2 + side * 10
    return float(sum(row[b + 5 + k] for k in range(5) if row[b + k] >= 0 and row[b + k] == price))


def follow(p, x):
    """One take: the aimed level's shown size at decision, when it vanished, and the public trades at or through it before that."""
    s, L, g, a = int(x['side']), float(x['limit']), float(x['submit']), float(x['act'])
    opp = 1 - s
    f = np.asarray(p['frames'], float); tr = np.asarray(p['trades'], float).reshape(-1, 4)
    jk = int(np.searchsorted(f[:, 1], g, side='right') - 1)          # the frame known at the decision (the limit came from it)
    C.require(jk >= 0, 'no known frame')
    s0 = level(f[jk], opp, L)
    t_gone, j = None, jk
    while j + 1 < len(f) and f[j + 1, 0] <= a:
        j += 1
        if level(f[j], opp, L) <= EPS:
            t_gone = float(f[j, 0]); break
    out = dict(reason=x['reason'], shown=s0, known_frame_venue_ms=float(f[jk, 0]) - g, gone_ms=None if t_gone is None else t_gone - g)
    for w, (lo, hi) in WIN.items():
        end = (t_gone if t_gone is not None else a) + hi
        m = (tr[:, 1] == opp) & ((tr[:, 2] >= L) if opp == 1 else (tr[:, 2] <= L)) & (tr[:, 0] >= f[jk, 0] + lo) & (tr[:, 0] <= end)
        out[f'traded_{w}'] = float(tr[m, 3].sum())
    return out


def main():
    fn = H / 'who_took.json'
    C.require(not fn.exists(), 'who_took.json exists (write-once)')
    sm = json.loads((S1D / 'summary.json').read_text())
    C.require(sm['delay_ms'] == 172., 'not the 172 ms run')
    want = collections.defaultdict(set)
    for r in sm['rows']:
        want[r['day']].add(int(r['mid']))
    src, recs = MM.KP.Tape(), []
    for day in OD.DAYS3:
        reg = MM.registry(day).set_index('mid'); off = src.offset(day)
        for h, g in reg.groupby('hour'):
            mids = [int(m) for m in g.index if int(m) in want[day]]
            if not mids:
                continue
            for m, st, p, info, yup in F.worlds_yup(day, mids, src, off, reg):
                C.require(st == 'complete', f'{m} {st}')
                led = json.loads(gzip.decompress((S1D / f'{m}_BASE.json.gz').read_bytes()))
                for x in led['ioc_results']:
                    recs.append(dict(follow(p, x), day=day, mid=m))
    res = dict(asof='2026-11-25', source='stage-1 172 ms BASE ledgers, 408 complete markets', takes=len(recs), by_reason={})
    for reason in sorted({r['reason'] for r in recs}):
        R = [r for r in recs if r['reason'] == reason]
        blk = dict(n=len(R), shown_shares=sum(r['shown'] for r in R),
                   vanished_before_arrival=sum(r['gone_ms'] is not None for r in R),
                   vanished_before_our_decision=sum(r['gone_ms'] is not None and r['gone_ms'] <= 0 for r in R))
        for w in WIN:
            fr = [min(1., r[f'traded_{w}'] / r['shown']) if r['shown'] > EPS else None for r in R]
            ok = [(f_, r) for f_, r in zip(fr, R) if f_ is not None and r['gone_ms'] is not None]
            sh = sum(r['shown'] for _, r in ok)
            blk[w] = dict(
                levels=len(ok),
                taken_by_trades=sum(f_ >= .8 for f_, _ in ok), mixed=sum(.2 <= f_ < .8 for f_, _ in ok), pulled_no_trade=sum(f_ < .2 for f_, _ in ok),
                share_of_shown_traded=sum(f_ * r['shown'] for f_, r in ok) / sh if sh > EPS else None)
        g = [r['gone_ms'] for r in R if r['gone_ms'] is not None]
        blk['vanish_ms_after_decision_p25_p50_p75'] = [float(np.quantile(g, q)) for q in (.25, .5, .75)] if g else None
        res['by_reason'][reason] = blk
    TR.dump(fn, res)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
