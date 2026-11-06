"""D1's A6 exposure lookup (r60) refit on Predict.fun, for the inventory-mechanism test of pm_ladder_v1 (pf_ladder_inv_build.py, RL<usd>).
D1 (P-2026-003, pmmmrust crates/strategy/src/research.rs): exposure = |net| x r60[bin(sensitivity(mid, t))], sensitivity = phi(z)^2 x
min(60, left) / left with z = Phi^-1(mid), left = 300 - t; a place that increases |net| is withheld when exposure >= the limit (1.86 $, the
90th percentile of the older strategy's exposure).
Here, on the 9 round-i days (consumed; in-sample by construction):
  lookup: every window of the day, t = 0, 1, ..., 239 s: Predict mid (b1 + a1) / 2 of the last frame at or before ws + t (venue time) and at
    ws + t + 60 s; both quotes present or the sample is dropped with a status. Bins: deciles of sensitivity (9 edges);
    move_per_share = mean |mid(t + 60) - mid(t)| in $ per share.
  limit: the 90th percentile of v1's own exposure after each fill (round-i J fills of the candidate, net in fill order per window, mid of
    the last frame at or before the fill), rounded to 2 dp.
Output: step4/inv_r60.json. Selftest of sensitivity / bin rule against D1's definition runs first.
Usage: python3 -m predictfun.research.pf_inv_r60
"""
import json, math, os, statistics, sys
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102', '20261103']
STEM = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'
ND = statistics.NormalDist()


def sensitivity(m, rel):
    """D1's sensitivity(mid, when), mid in (0, 1)."""
    if m <= 0.0 or m >= 1.0:
        return 0.0
    z = ND.inv_cdf(m); left = 300.0 - rel
    return math.exp(-z * z) / (2.0 * math.pi) * min(60.0, left) / left


def bin_of(edges, s):
    """D1: edges.partition_point(|e| e <= s) = number of edges <= s."""
    return int(np.searchsorted(edges, s, side='right'))


def selftest():
    assert abs(sensitivity(0.5, 0.0) - 1.0 / (2.0 * math.pi) * 60.0 / 300.0) < 1e-15
    assert sensitivity(0.5, 270.0) == 1.0 / (2.0 * math.pi)                 # left 30 < 60: the cap is left itself
    assert sensitivity(0.0, 10.0) == 0.0 and sensitivity(1.0, 10.0) == 0.0
    assert abs(sensitivity(0.2, 0.0) - sensitivity(0.8, 0.0)) < 1e-12 and sensitivity(0.2, 0.0) < sensitivity(0.5, 0.0)
    e = [0.1, 0.2]
    assert (bin_of(e, 0.05), bin_of(e, 0.1), bin_of(e, 0.15), bin_of(e, 0.2), bin_of(e, 0.3)) == (0, 1, 1, 2, 2)


def day_books(d):
    S1 = f'{io.OUT}/step1/{d}'
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws'])
    b = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'recv_ns', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
    b = b[b.mid.isin(reg.mid)].sort_values(['mid', 't_ms', 'recv_ns'], kind='stable')
    return reg.set_index('mid').ws, {m: (g.t_ms.values.astype(np.float64), g.b1_p.values, g.a1_p.values) for m, g in b.groupby('mid', sort=False)}


def mid_at(bk, t):
    """mid in cents of the last frame at or before t (venue ms), None if no frame or a side is missing."""
    tt, b1, a1 = bk; i = int(np.searchsorted(tt, t, side='right')) - 1
    if i < 0 or b1[i] < 0 or a1[i] < 0 or a1[i] <= b1[i]:
        return None
    return (b1[i] + a1[i]) / 2.0


def main():
    selftest()
    S, M, st = [], [], dict(samples=0, dropped_no_quote=0, windows_no_book=0)
    books = {}
    for d in DAYS:
        ws, bk = day_books(d); books[d] = (ws, bk)
        for m, w in ws.items():
            if m not in bk:
                st['windows_no_book'] += 1; continue
            w_ms = float(w) * 1000.0
            for rel in range(240):
                a = mid_at(bk[m], w_ms + rel * 1000.0); b = mid_at(bk[m], w_ms + (rel + 60) * 1000.0)
                if a is None or b is None:
                    st['dropped_no_quote'] += 1; continue
                S.append(sensitivity(a / 100.0, float(rel))); M.append(abs(b - a) / 100.0); st['samples'] += 1
    S = np.array(S); M = np.array(M)
    edges = np.unique(np.quantile(S, np.arange(1, 10) / 10.0)).tolist()
    bi = np.searchsorted(edges, S, side='right')
    mps = [float(M[bi == k].mean()) for k in range(len(edges) + 1)]
    lut = dict(edges=edges, move_per_share=mps)
    # v1's exposure after each fill (round-i J)
    E = []
    for d in DAYS:
        ws, bk = books[d]
        f = pd.read_parquet(f'{io.OUT}/step4/ladder_tokyoreal_custom_e2_preds_t60k_lJi/{d}/fills/{STEM}', columns=['mid', 't', 'side', 'q'])
        for m, g in f.sort_values(['mid', 't'], kind='stable').groupby('mid', sort=False):
            net = np.cumsum(np.where(g.side.values == 0, g.q.values, -g.q.values))
            for t, n in zip(g.t.values, net):
                mc = mid_at(bk[m], float(ws[m]) * 1000.0 + t * 1000.0)
                if mc is None:
                    continue
                E.append(abs(n) * mps[bin_of(edges, sensitivity(mc / 100.0, float(t)))])
    E = np.array(E); L = round(float(np.percentile(E, 90)), 2)
    out = dict(days=DAYS, status=st, edges=edges, move_per_share=mps, limit_usd=L,
               v1_fill_exposure=dict(n=int(len(E)), p50=round(float(np.median(E)), 3), p90=L, max=round(float(E.max()), 3)),
               shares_at_limit_by_bin=[round(L / x, 1) if x > 0 else None for x in mps])
    io.write_json(f'{io.OUT}/step4/inv_r60.json', out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
