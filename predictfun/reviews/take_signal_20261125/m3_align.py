"""TAKE-SIGNAL v1: Maker3 take alignment with the BTC triggers (contract.md 'Maker3 alignment'). One row per Maker3 take record
(day, market, sid), status-ok taker legs from maker3_book_conditions legs_m3. Agreeing = a qualifying event of the take's YES direction
with decision time in [stamp - 3.5 s, stamp]. Null: 500 seeded circular shifts of each market's take schedule within [ws, END].
Also (labelled exploratory, not in the contract): the same with the kept OFF->ON triggers only.
Usage: run_light.sh ts_m3 -- python3 predictfun/reviews/take_signal_20261125/m3_align.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
OUT = R / 'data/predict_fun/research/take_signal_20261125'
LEGS = R / 'data/predict_fun/research/maker3_book_conditions_20261125/legs_m3.parquet'
V = R / 'data/predict_fun/research/maker_trace_v2_20261123'
DAYS = ('20261024', '20261025', '20261026', '20261027')
WIN_LO, WIN_HI = 3500, 0
DRAWS, SEED = 500, 20261125


def agree_frac(tt, ss, ev_t, ev_d):
    """For take times tt (ms) with directions ss: fraction with a same-direction event in [t - WIN_LO, t - WIN_HI]."""
    hit = np.zeros(len(tt), bool)
    for s in (1, -1):
        e = np.sort(ev_t[ev_d == s])
        m = ss == s
        if not m.any() or not len(e):
            continue
        lo = np.searchsorted(e, tt[m] - WIN_LO, 'left')
        hi = np.searchsorted(e, tt[m] - WIN_HI, 'right')
        hit[np.where(m)[0]] = hi > lo
    return hit


def main():
    L = pd.read_parquet(LEGS)
    tk = L[(L.role == 'taker') & ~L.self_trade.astype(bool)]
    g = tk.groupby(['day', 'mid', 'sid'])
    rec = g.agg(t=('settle_ms', 'max'), s=('s', 'first'), s_n=('s', 'nunique'), q=('q', 'sum'), net=('net_res', 'sum'),
                v1=('v_1s', 'sum'), vend=('v_end', 'sum'), t_rel_s=('t_rel_s', 'max'),
                add=('inv', lambda x: x.isin(['ADD', 'OPEN', 'FLIP']).mean())).reset_index()
    assert (rec.s_n == 1).all(), 'a take record mixes YES directions'
    rec['kind'] = np.where(rec['add'] >= 0.5, 'adding', 'reducing')
    ev = pd.read_parquet(OUT / 'events_all.parquet')
    mk = pd.concat([pd.read_parquet(V / f'markets_{d}.parquet', columns=['mid', 'ws', 'we']).assign(day=d) for d in DAYS])
    rec = rec.merge(mk, on=['day', 'mid'], how='left')
    rng = np.random.default_rng(SEED)
    out = {}
    for label, E in (('all_qualifying', ev), ('kept_triggers_exploratory', ev[ev.kept])):
        obs = np.zeros(len(rec), bool)
        null = np.zeros((DRAWS, len(rec)), bool)
        for (day, mid), x in rec.groupby(['day', 'mid']):
            e = E[(E.day == day) & (E.mid == mid)]
            idx = x.index.to_numpy()
            tt, ss = x.t.to_numpy(np.int64), x.s.to_numpy(np.int64)
            et, ed = e.t.to_numpy(np.int64), e.dir.to_numpy(np.int64)
            obs[idx] = agree_frac(tt, ss, et, ed)
            ws, end = int(x.ws.iloc[0]) * 1000, int(x.we.iloc[0]) * 1000 - 2000
            span = end - ws
            for k in range(DRAWS):
                off = rng.integers(0, span)
                null[k, idx] = agree_frac(ws + (tt - ws + off) % span, ss, et, ed)
        res = {}
        for kind in ('adding', 'reducing', 'all'):
            m = (rec.kind == kind).to_numpy() if kind != 'all' else np.ones(len(rec), bool)
            o = float(obs[m].mean())
            nd = null[:, m].mean(axis=1)
            sub = rec[m]
            a, na = sub[obs[m]], sub[~obs[m]]
            res[kind] = dict(takes=int(m.sum()), observed_agree=o, null_mean=float(nd.mean()), null_p95=float(np.quantile(nd, .95)),
                             frac_null_ge_observed=float((nd >= o).mean()),
                             agree_net_c=float(100 * a.net.sum() / a.q.sum()) if len(a) else None,
                             nonagree_net_c=float(100 * na.net.sum() / na.q.sum()) if len(na) else None,
                             agree_v1_c=float(100 * a.v1.sum() / a.q.sum()) if len(a) else None,
                             nonagree_v1_c=float(100 * na.v1.sum() / na.q.sum()) if len(na) else None,
                             agree_vend_c=float(100 * a.vend.sum() / a.q.sum()) if len(a) else None,
                             nonagree_vend_c=float(100 * na.vend.sum() / na.q.sum()) if len(na) else None)
        early = (rec.t_rel_s < 60).to_numpy() & (rec.kind == 'adding').to_numpy()
        res['adding_first_60s'] = dict(takes=int(early.sum()), observed_agree=float(obs[early].mean()),
                                       null_mean=float(null[:, early].mean(axis=1).mean()),
                                       frac_null_ge_observed=float((null[:, early].mean(axis=1) >= obs[early].mean()).mean()))
        out[label] = res
    out['take_records'] = int(len(rec))
    (H / 'm3_align.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
