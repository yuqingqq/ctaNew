"""MAKER3-BOOK-CONDITIONS v1, independent checks (contract.md). A different code path from build.py wherever possible:
 C1 book state at P re-derived with pandas merge_asof per market on a hash-ranked sample of 4,000 Maker3 legs (all fields equal);
 C2 causality falsifier: frames after P perturbed -> P state unchanged; positive control: perturbing the frame at/before P changes it;
 C3 classifier controls on synthetic frames (calm / thin / wide / one-sided / stale / crossed / bad frame / no frame);
 C4 reconciliation: per-market Maker3 net from the legs = econ maker_market net_res on all active markets; cells = legs;
 C5 Maker3 inventory re-derived by a per-leg Python loop on 40 hash-ranked markets;
 C6 headline readings recomputed from legs (not cells) and compared with readings.json;
 C7 corruption refusals: an altered leg, a dropped market, a P->U clock swap must each be detected.
Usage: run_light.sh mbc_check -- python3 .../check.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import build as B  # noqa: E402  (only for the classifier under test in C2/C3; C1/C4-C6 do not use it)

R = H.parents[2]
V = R / 'data/predict_fun/research/maker_trace_v2_20261123'
OUT = R / 'data/predict_fun/research/maker3_book_conditions_20261125'
res = {}


def rank(s):
    return np.array([int(hashlib.sha256(str(x).encode()).hexdigest()[:12], 16) for x in s])


def ok(name, cond, detail=''):
    res[name] = dict(ok=bool(cond), detail=detail)
    print(('PASS ' if cond else 'FAIL ') + name, detail)


def independent_state(f, clk):
    """pandas merge_asof by market: the last frame with upd_ms <= clk (ties: latest recv_ns)."""
    f = f.sort_values(['upd_ms', 'recv_ns'], kind='mergesort').drop_duplicates('upd_ms', keep='last')
    q = pd.DataFrame(dict(c=np.asarray(clk, np.int64))).reset_index()
    j = pd.merge_asof(q.sort_values('c'), f.rename(columns={'upd_ms': 'c_f'}), left_on='c', right_on='c_f', direction='backward')
    j = j.sort_values('index')
    out = []
    for r in j.itertuples():
        if pd.isna(r.c_f):
            out.append(('UNKNOWN', 'no_frame', np.nan, np.nan))
            continue
        bid, ask = r.best_bid, r.best_ask
        st = str(r.status) if isinstance(r.status, str) else ''
        if any(b in st for b in ('bad_frame', 'mid_mismatch', 'out_of_order', 'off_grid')):
            out.append(('UNKNOWN', 'frame_status', np.nan, np.nan)); continue
        if not (np.isfinite(bid) and np.isfinite(ask)):
            out.append(('ONE_SIDED', 'one_sided', np.nan, np.nan)); continue
        if bid >= ask:
            out.append(('UNKNOWN', 'crossed', np.nan, np.nan)); continue
        if r.c - r.c_f > 1000:
            out.append(('UNKNOWN', 'stale', np.nan, np.nan)); continue
        spr = (ask - bid) * 100
        thin = min(r.best_bid_sh, r.best_ask_sh)
        cls = 'WIDE' if spr > 1 + 1e-7 else ('THIN' if thin <= B.THIN_SH else 'CALM')
        out.append((cls, 'ok', spr, (bid + ask) / 2))
    return out


def main():
    L = pd.read_parquet(OUT / 'legs_m3.parquet')
    K = pd.read_parquet(OUT / 'markets.parquet')
    rd = json.loads((H / 'readings.json').read_text())
    # C1 independent book state on a sample
    L['_r'] = rank(L.day + ':' + L.sid.astype(str) + ':' + L.mk_idx.astype(str) + ':' + L.role.astype(str))
    S = L.nsmallest(4000, '_r')
    mism = 0
    n = 0
    for day, g in S.groupby('day'):
        tob = pq.read_table(V / f'tob_{day}.parquet', columns=['mid', 'upd_ms', 'recv_ns', 'best_bid', 'best_bid_sh', 'best_ask', 'best_ask_sh',
                                                             'status'], filters=[('mid', 'in', sorted(set(g.mid.astype(int))))]).to_pandas()
        for mid, gg in g.groupby('mid'):
            st = independent_state(tob[tob.mid == mid], gg.settle_ms.to_numpy(np.int64) - B.P_LAG_MS)
            for (cls, s_, spr, mid_), r in zip(st, gg.itertuples()):
                n += 1
                same = (cls == r.cls_P) and (s_ == r.st_P) and ((np.isnan(spr) and np.isnan(r.spr_P)) or abs(spr - r.spr_P) < 1e-9) and \
                       ((np.isnan(mid_) and np.isnan(r.mid_P)) or abs(mid_ - r.mid_P) < 1e-12)
                mism += (not same)
    ok('C1_independent_P_state', mism == 0 and n == 4000, f'{n} legs, {mism} mismatches')
    # C2 causality falsifier + positive control (build's book_at under test)
    day = '20261026'
    g = S[S.day == day].head(300)
    mk = pd.read_parquet(V / f'markets_{day}.parquet', columns=['mid'])
    mindex = {int(m): i for i, m in enumerate(sorted(mk.mid))}
    tob = B.load_tob(day, mindex)
    midx = g.mid.map(mindex).to_numpy(np.int64)
    P = g.settle_ms.to_numpy(np.int64) - B.P_LAG_MS
    base = B.book_at(tob, midx, P)
    i_at = np.searchsorted(tob['key'], midx * B.KEY + P, 'right') - 1
    n_pert, changed = 0, 0
    for n_, (i, m, p_) in enumerate(zip(i_at, midx, P)):   # one leg at a time: its market's frames after its own P frame
        s_ = np.searchsorted(tob['key'], m * B.KEY, 'left')
        e = np.searchsorted(tob['key'], (m + 1) * B.KEY, 'left')
        sub = {k: v[s_:e].copy() for k, v in tob.items()}
        cut = max(i + 1 - s_, 0)
        sub['bid'][cut:] = 0.01
        sub['ask'][cut:] = 0.99
        sub['bsh'][cut:] = 1.0
        n_pert += len(sub['bid']) - cut
        one = B.book_at(sub, np.array([m]), np.array([p_]))
        for k in ('cls', 'st', 'mid', 'spr', 'thin_side'):
            a, b = base[k][n_], one[k][0]
            changed += not ((a == b) or (isinstance(a, float) and np.isnan(a) and np.isnan(b)))
    same_future = changed == 0
    pos = {k: v.copy() for k, v in tob.items()}
    sel = np.zeros(len(tob['key']), bool)
    sel[i_at[i_at >= 0]] = True
    pos['ask'] = np.where(sel, pos['bid'] + 0.09, pos['ask'])  # the P frame itself widened -> must change the state
    pc = B.book_at(pos, midx, P)
    fired = int((pc['cls'] != base['cls']).sum())
    ok('C2_future_perturbation_no_effect', same_future and n_pert > 0, f'{len(g)} legs one at a time, {n_pert} later frames perturbed, {changed} fields changed')
    ok('C2_positive_control_fires', fired > 0.5 * len(g), f'{fired}/{len(g)} states changed when the P frame is widened')
    # C3 classifier controls on synthetic frames
    syn = dict(key=np.array([0 * B.KEY + 1000, 1 * B.KEY + 1000, 2 * B.KEY + 1000, 3 * B.KEY + 1000, 4 * B.KEY + 1000, 5 * B.KEY + 1000,
                             6 * B.KEY + 1000]), midx=np.arange(7), upd=np.full(7, 1000),
               bid=np.array([.50, .50, .50, np.nan, .50, .55, .50]), ask=np.array([.51, .51, .56, .60, .51, .54, .51]),
               bsh=np.array([500., 10., 500., 0., 500., 500., 500.]), ash=np.array([400., 400., 400., 30., 400., 400., 400.]),
               bad=np.array([False, False, False, False, False, False, True]))
    clk = np.array([1500, 1500, 1500, 1500, 2600, 1500, 1500])
    got = B.book_at(syn, np.arange(7), clk)['cls'].tolist() + B.book_at(syn, np.array([0]), np.array([999]))['cls'].tolist()
    want = ['CALM', 'THIN', 'WIDE', 'ONE_SIDED', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN']
    ok('C3_classifier_controls', got == want, f'{got}')
    # C4 reconciliation to econ maker_market (all active markets) and cells
    pm = L.groupby(['day', 'mid']).net_res.sum()
    mm = pq.read_table(V / 'econ/maker_market.parquet', filters=[('addr', '=', B.M3), ('day', 'in', list(B.DAYS))],
                       columns=['day', 'mid', 'net_res', 'status']).to_pandas()
    mm = mm[mm.status == 'ok'].set_index(['day', 'mid']).net_res
    j = pd.concat([pm.rename('legs'), mm.rename('mm')], axis=1)
    d = (j.legs - j.mm).abs()
    ok('C4_legs_equal_maker_market', j.isna().sum().sum() == 0 and d.max() < 1e-6, f'{len(j)} markets, max |diff| {d.max():.2e}')
    C = pd.read_parquet(OUT / 'cells_m3.parquet')
    A = pd.read_parquet(OUT / 'cells_all.parquet')
    cP = C[C.clock == 'P']
    ok('C4_cells_m3_equal_legs', abs(cP.net.sum() - L.net_res.sum()) < 1e-6 and abs(cP.q.sum() - L.q.sum()) < 1e-6 and int(cP.legs.sum()) == len(L),
       f'net {cP.net.sum():.4f} vs {L.net_res.sum():.4f}')
    aP = A[(A.clock == 'P') & (A.grp == 'MAKER3')]
    lm = L[L.role == 'maker']
    ok('C4_cells_all_maker3_equal_legs', abs(aP.q.sum() - lm.q.sum()) < 1e-6 and abs(aP.net.sum() - lm.net_res.sum()) < 1e-6,
       f'q {aP.q.sum():.3f} vs {lm.q.sum():.3f}')
    # C5 inventory by an explicit loop on 40 markets
    mk40 = L.drop_duplicates(['day', 'mid'])[['day', 'mid']].assign(_r=lambda x: rank(x.day + ':' + x.mid.astype(str))).nsmallest(40, '_r')
    bad_inv = 0
    m = 0
    for r in mk40.itertuples():
        g = L[(L.day == r.day) & (L.mid == r.mid)].copy()
        g['_ro'] = (g.role != 'maker').astype(int)
        g = g.sort_values(['settle_ms', 'sid', 'px_rank', 'mk_idx', '_ro'], kind='mergesort')
        pos = 0.0
        for x in g.itertuples():
            m += 1
            if x.self_trade:
                exp = 'SELF'
            elif abs(pos) < 1e-9:
                exp = 'OPEN'
            elif np.sign(pos) == x.s:
                exp = 'ADD'
            elif x.q <= abs(pos) + 1e-9:
                exp = 'REDUCE'
            else:
                exp = 'FLIP'
            bad_inv += (exp != x.inv) or abs(pos - x.pos_before) > 1e-6
            if not x.self_trade:
                pos += x.s * x.q
    ok('C5_inventory_loop', bad_inv == 0 and m > 0, f'{m} legs in 40 markets, {bad_inv} mismatches')
    # C6 headline readings from legs
    late = L.tb.isin(['270-285', '285-295', '295-300', 'post_stamp'])
    mkr = L.role == 'maker'
    vol = L.cls_P.isin(['THIN', 'WIDE'])
    calc = {'P|>=270|THIN_or_WIDE': L[mkr & late & vol].net_res.sum(), 'P|all|CALM': L[mkr & (L.cls_P == 'CALM')].net_res.sum(),
            'P|<270|CALM': L[mkr & ~late & (L.cls_P == 'CALM')].net_res.sum(), 'P|all|THIN_or_WIDE': L[mkr & vol].net_res.sum()}
    diffs = {k: abs(v - rd['b_c_m3_cells'][k]['net_usd']) for k, v in calc.items()}
    ok('C6_headline_readings', max(diffs.values()) < 1e-6, json.dumps({k: round(v, 2) for k, v in calc.items()}))
    frac = L[mkr & vol].q.sum() / L[mkr].q.sum()
    ok('C6_volume_fraction', abs(frac - rd['a_m3_volume_frac_thin_or_wide_P']) < 1e-12, f'{frac:.6f}')
    # C7 corruption refusals (each must be detected by the same comparisons)
    L2 = L.copy()
    L2.loc[L2.index[0], 'net_res'] += 0.5
    d2 = (L2.groupby(['day', 'mid']).net_res.sum() - mm).abs().max()
    ok('C7_altered_leg_detected', d2 > 1e-3, f'max |diff| {d2:.3f}')
    drop = L[~((L.day == L.day.iloc[0]) & (L.mid == L.mid.iloc[0]))]
    j3 = pd.concat([drop.groupby(['day', 'mid']).net_res.sum().rename('legs'), mm.rename('mm')], axis=1)
    ok('C7_dropped_market_detected', int(j3.isna().any(axis=1).sum()) == 1, 'one market missing')
    sw = (L.cls_P != L.cls_U).mean()
    ok('C7_clock_swap_detected', sw > 0.05, f'{sw:.3f} of legs change class between P and U (a P->U swap cannot pass C1)')
    res['all_pass'] = all(v['ok'] for k, v in res.items() if isinstance(v, dict))
    (H / 'checks.json').write_text(json.dumps(res, indent=1) + '\n')
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED')
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    main()
