"""How the top taker exits, in more detail (USER 2026-11-26: "I just want to check how they exit"). Readout of exit_times' round trips
(OUT/exit_trips.parquet), positions it opened by taking. Declared before the run: taker exits split by the price change its way before fees
(gain > 0 / flat 0 / loss < 0) and by holding time (<= 5 s, 5-30 s, 30-60 s, 60-120 s, > 120 s): volume share, net per share, median hold;
the time left in the window at each exit (window end - exit) and at the entries it holds to settlement; settlement holds by the entry price of
its side (< 40 / 40-60 / > 60 c): volume share and win rate.
Output (write-once): exit_profile.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_profile.py [<rehearsal dir>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import exit_times as XT  # noqa: E402

WT, C, TR = XT.WT, XT.C, XT.TR
HB = ((0., 5000., '<= 5 s'), (5000., 30000., '5-30 s'), (30000., 60000., '30-60 s'), (60000., 120000., '60-120 s'), (120000., np.inf, '> 120 s'))
PB = ((0., 40., 'its side < 40c'), (40., 60., '40-60c'), (60., 100.01, '> 60c'))


def main():
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'exit_profile.json'
    C.require(not fn.exists(), 'exit_profile.json exists (write-once)')
    T = pd.read_parquet(WT.OUT / 'exit_trips.parquet')
    L = pd.concat([pd.read_parquet(WT.OUT / f'exit_legs_{d}.parquet', columns=['taker_hash', 'mid', 'role']) for d in WT.DAYS])
    mid_of = L[L.role == 'taker'].drop_duplicates('taker_hash').set_index('taker_hash').mid
    reg = pd.concat([WT.MM.registry(d) for d in WT.DAYS])
    end_of = dict(zip(reg.mid.astype(int), reg.ws.astype(float) * 1000. + 300000.))
    E = T[(T.entry_role == 'taker') & T.taker_hash.notna()].copy()
    E['end'] = E.taker_hash.map(mid_of).map(end_of)
    C.require(E.end.notna().mean() > .99, 'trips without their market window')
    E['g'] = E.side * (E.exit_price - E.entry_price)
    w = lambda g: float(g.shares.sum())
    X, S = E[E.ending == 'taker_exit'], E[E.ending == 'settlement']
    tot = w(X)
    def blk(g):
        return dict(share=w(g) / tot, net_c=float((g.net_c * g.shares).sum() / g.shares.sum()) if len(g) else None,
                    gain_c=float((g.g * g.shares).sum() / g.shares.sum()) if len(g) else None, hold_s_median=float(np.median(g.hold_ms)) / 1000. if len(g) else None, n=int(len(g)))
    out = dict(asof='2026-11-26', taker_exit_volume=tot, settlement_volume=w(S),
               taker_exits_by_result={k: blk(X[m]) for k, m in (('gain', X.g > 1e-9), ('flat', X.g.abs() <= 1e-9), ('loss', X.g < -1e-9))},
               taker_exits_by_hold={lb: blk(X[(X.hold_ms > a) & (X.hold_ms <= b)] if a > 0 else X[X.hold_ms <= b]) for a, b, lb in HB},
               time_left_at_exit_s={f'p{q}': float(np.quantile((X.end - X.exit_t) / 1000., q / 100.)) for q in (10, 25, 50, 75, 90)},
               exits_with_30s_or_less_left=float(X.shares[(X.end - X.exit_t) <= 30000.].sum() / tot),
               settlement_entries_time_left_s={f'p{q}': float(np.quantile((S.end - S.entry_t) / 1000., q / 100.)) for q in (10, 25, 50, 75, 90)},
               settlement_entries_with_30s_or_less_left=float(S.shares[(S.end - S.entry_t) <= 30000.].sum() / w(S)),
               settlement_by_own_price={})
    own = np.where(S.side > 0, S.entry_price, 100. - S.entry_price)
    won = S.side * (S.exit_price - 50.) > 0
    for a, b, lb in PB:
        m = (own >= a) & (own < b)
        out['settlement_by_own_price'][lb] = dict(share=float(S.shares[m].sum() / w(S)), won=float(S.shares[m & won].sum() / S.shares[m].sum()) if m.any() else None,
                                                  net_c=float((S.net_c[m] * S.shares[m]).sum() / S.shares[m].sum()) if m.any() else None)
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
