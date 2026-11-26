"""For our takes, how should we exit? (USER 2026-11-26: "Ok, then for us, how do we exit"). Readout of SCORE_REPLAY's takes (the public
score's top-1% moments and the random control, venue-matched harness, 22 ms path) under alternative exits; no new replay.
Per filled take (q shares at its fill price p, UP space, side sg = +1 when it bought UP): the replayed passive exit (the ledger at
settlement, per market); hold to settlement = sg x (100 x payout - p) - entry fee per share; take back at 1 s / 10 s = the take's mid
markout at that horizon (already net of the entry fee) minus the exit fee 2% x min(m, 100 - m) at the entry price, as an UPPER BOUND (no
spread paid, no 150 ms hold on the exit) and with half a cent of spread; the top taker's style = hold to settlement when its side's
entry price >= 60c, else take back at 10 s. Declared before the run; point estimates (G = 4 days); settlement outcomes are binary and few:
noisy.
Output (write-once): exit_options.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_options.py [<rehearsal dir>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import score_replay as SRy  # noqa: E402

C, TR = SRy.C, SRy.TR
HALF_SPREAD = .5
PB = ((0., 40., 'our side < 40c'), (40., 60., '40-60c'), (60., 100.01, '>= 60c'))


def main():
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'exit_options.json'
    C.require(not fn.exists(), 'exit_options.json exists (write-once)')
    D = [json.loads((SRy.OUT / f'score_replay_{d}.json').read_text()) for d in SRy.DAYS]
    pay = {(r['mid']): r['payout'] for dd in D for r in dd['rows']}
    T = pd.DataFrame([x for dd in D for x in dd['takes']])
    T['payout'] = T.mid.map(pay)
    C.require(T.payout.isin([0., .5, 1.]).all(), 'a take without its payout')
    T['sg'] = np.where(T.side == 0, 1., -1.)
    T['fee_sh'] = T.fee_c / T.q
    T['own'] = np.where(T.sg > 0, T.price, 100. - T.price)
    T['hold'] = T.sg * (100. * T.payout - T.price) - T.fee_sh
    xf = .02 * np.minimum(T.price, 100. - T.price)
    for h in ('m1s_c', 'm10s_c'):
        T[f'back_{h}_ub'] = T[h] - xf
        T[f'back_{h}_hs'] = T[h] - xf - HALF_SPREAD
    T['style_ub'] = np.where(T.own >= 60., T.hold, T['back_m10s_c_ub'])
    T['style_hs'] = np.where(T.own >= 60., T.hold, T['back_m10s_c_hs'])
    out = dict(asof='2026-11-26', half_spread_c=HALF_SPREAD, configs={})
    for cfg, g in T.groupby('config'):
        led = sum(r['arms'][cfg]['total_c'] for dd in D for r in dd['rows']) / 100.
        sh = float(g.q.sum())
        wm = lambda c: float((g[c] * g.q)[g[c].notna()].sum() / g.q[g[c].notna()].sum()) if g[c].notna().any() else None
        o = dict(takes=int(len(g)), shares=sh, passive_exit_ledger_usd=float(led), passive_exit_c_per_share=100. * led / sh,
                 hold_to_settlement_c=wm('hold'), take_back_1s_c_upper=wm('back_m1s_c_ub'), take_back_1s_c_half_spread=wm('back_m1s_c_hs'),
                 take_back_10s_c_upper=wm('back_m10s_c_ub'), take_back_10s_c_half_spread=wm('back_m10s_c_hs'), top_taker_style_c_upper=wm('style_ub'),
                 top_taker_style_c_half_spread=wm('style_hs'), settled_up_share=float((g.payout == 1.).mean()), by_our_side_price={})
        for a, b, lb in PB:
            m = (g.own >= a) & (g.own < b)
            gg = g[m]
            o['by_our_side_price'][lb] = dict(takes=int(m.sum()), share=float(gg.q.sum() / sh), won=float((gg.q * (gg.sg * (gg.payout - .5) > 0)).sum() / gg.q.sum()) if m.any() else None,
                                              hold_c=float((gg.hold * gg.q).sum() / gg.q.sum()) if m.any() else None,
                                              back_10s_c_upper=float((gg.back_m10s_c_ub * gg.q)[gg.back_m10s_c_ub.notna()].sum() / gg.q[gg.back_m10s_c_ub.notna()].sum()) if m.any() else None)
        o['by_day_hold_c'] = {d: float((x.hold * x.q).sum() / x.q.sum()) for d, x in g.groupby('day')}
        out['configs'][cfg] = o
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
