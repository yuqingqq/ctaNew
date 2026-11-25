"""MAKER3-BOOK-CONDITIONS v1, stage 2: book state at every maker leg (all addresses) and every Maker3 leg, Oct 24-27; contract.md.
Inputs (read-only, already-built day files): maker_trace_v2 markets_<day>, tob_<day>, econ/fills_valued, econ/maker_market,
econ/identities; stage-1 grid.parquet. Outputs (write-once): data/.../maker3_book_conditions_20261125/{legs_m3.parquet, cells_all.parquet,
cells_m3.parquet, markets.parquet} and build.json in this packet.
Usage: run_light.sh mbc_build -- python3 predictfun/reviews/maker3_book_conditions_20261125/build.py
"""
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
H = Path(__file__).resolve().parent
R = H.parents[2]
V = R / 'data/predict_fun/research/maker_trace_v2_20261123'
OUT = R / 'data/predict_fun/research/maker3_book_conditions_20261125'
DAYS = ('20261024', '20261025', '20261026', '20261027')
M3 = '0x57FbC891E89184DA071AE9D97E745531cd7F718a'
P_LAG_MS = 2500
THIN_SH = 43.7                       # stage-1 q25 of the thinner-side touch (grid ok seconds), fixed in contract.md
BAD = ('bad_frame', 'mid_mismatch', 'out_of_order', 'off_grid')
TB_EDGES = [-1e12, 0, 60, 120, 180, 240, 270, 285, 295, 300, 1e12]
TB_LABELS = ['pre', '000-060', '060-120', '120-180', '180-240', '240-270', '270-285', '285-295', '295-300', 'post_stamp']
KEY = 10 ** 13
FV_COLS = ['day', 'mid', 'sid', 'mk_idx', 'px_rank', 'role', 'addr', 'y_side', 's', 'q', 'p', 'self_trade', 'settle_ms', 't_ms', 't_status',
           't_rel_s', 'V', 'st_res', 'gross_res', 'fee_res', 'net_res', 'v_1s', 'st_1s', 'v_10s', 'st_10s', 'v_60s', 'st_60s', 'v_end', 'st_end',
           'status']
VALS = ('v_1s', 'v_10s', 'v_60s', 'v_end')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def guard(tag):
    mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    assert mib < 2200, f'VmHWM self-guard at {tag}: {mib:.0f} MiB'
    return round(mib, 1)


def book_at(tob, midx, clock):
    """State of the last tob frame with upd_ms <= clock in market midx (vectorized; econ MID rule)."""
    q = midx.astype(np.int64) * KEY + clock.astype(np.int64)
    i = np.searchsorted(tob['key'], q, 'right') - 1
    found = (i >= 0)
    j = np.where(found, i, 0)
    found &= tob['midx'][j] == midx
    bid, ask = tob['bid'][j], tob['ask'][j]
    age = clock - tob['upd'][j]
    st = np.full(len(q), 'ok', dtype=object)
    one = ~(np.isfinite(bid) & np.isfinite(ask))
    st[age > 1000] = 'stale'
    st[~one & (bid >= ask)] = 'crossed'
    st[one] = 'one_sided'
    st[tob['bad'][j]] = 'frame_status'
    st[~found] = 'no_frame'
    ok2 = st == 'ok'
    mid = np.where(ok2, (bid + ask) / 2, np.nan)
    spr = np.where(ok2, (ask - bid) * 100, np.nan)
    thin_side = np.where(ok2, np.minimum(tob['bsh'][j], tob['ash'][j]), np.nan)
    present = np.where(st == 'one_sided', np.where(np.isfinite(bid), bid, ask) * 100, np.nan)
    cls = np.full(len(q), 'UNKNOWN', dtype=object)
    cls[ok2 & (spr <= 1 + 1e-7) & (thin_side > THIN_SH)] = 'CALM'
    cls[ok2 & (spr <= 1 + 1e-7) & (thin_side <= THIN_SH)] = 'THIN'
    cls[ok2 & (spr > 1 + 1e-7)] = 'WIDE'
    cls[st == 'one_sided'] = 'ONE_SIDED'
    band = np.full(len(q), 'na', dtype=object)
    band[ok2 & (spr <= 1 + 1e-7)] = '<=1'
    band[ok2 & (spr > 1 + 1e-7) & (spr <= 2 + 1e-7)] = '1-2'
    band[ok2 & (spr > 2 + 1e-7) & (spr <= 4 + 1e-7)] = '2-4'
    band[ok2 & (spr > 4 + 1e-7)] = '>4'
    px = np.where(ok2, mid * 100, present)
    money = np.where(np.isfinite(px), np.where((px <= 10) | (px >= 90), 'EXTREME', 'NEAR'), 'na')
    return dict(st=st, cls=cls, band=band, money=money, mid=mid, spr=spr, thin_side=thin_side, age=np.where(found, age, -1))


def load_tob(day, mindex):
    t = pq.read_table(V / f'tob_{day}.parquet', columns=['mid', 'upd_ms', 'recv_ns', 'best_bid', 'best_bid_sh', 'best_ask', 'best_ask_sh',
                                                         'status'], use_threads=False).to_pandas()
    t['midx'] = t.mid.map(mindex)
    assert t.midx.notna().all(), 'tob market outside the registry'
    t = t.sort_values(['midx', 'upd_ms', 'recv_ns'], kind='mergesort')
    bad = t.status.fillna('').astype(str).str.contains('|'.join(BAD), regex=True).to_numpy(bool)
    return dict(key=t.midx.to_numpy(np.int64) * KEY + t.upd_ms.to_numpy(np.int64), midx=t.midx.to_numpy(np.int64), upd=t.upd_ms.to_numpy(np.int64),
                bid=t.best_bid.to_numpy(float), ask=t.best_ask.to_numpy(float), bsh=t.best_bid_sh.to_numpy(float),
                ash=t.best_ask_sh.to_numpy(float), bad=bad)


def sums(df, keys):
    d = df.assign(q_edge=np.where(np.isfinite(df.edge_P), df.q, 0.0), edge_P=df.edge_P.fillna(0.0), out_P=df.out_P.fillna(0.0),
                  **{f'q_{v}': np.where(np.isfinite(df[v]), df.q, 0.0) for v in VALS}, **{v: df[v].fillna(0.0) for v in VALS})
    agg = dict(legs=('q', 'size'), q=('q', 'sum'), gross=('gross_res', 'sum'), fee=('fee_res', 'sum'), net=('net_res', 'sum'),
               edge=('edge_P', 'sum'), out=('out_P', 'sum'), q_edge=('q_edge', 'sum'),
               **{v: (v, 'sum') for v in VALS}, **{f'q_{v}': (f'q_{v}', 'sum') for v in VALS})
    return d.groupby(keys, observed=True, dropna=False).agg(**agg).reset_index()


def inventory(m3):
    """Position before each Maker3 leg in the econ FIFO order; self-trades do not move it (econ lots)."""
    m3 = m3.assign(_ro=(m3.role != 'maker').astype(int)).sort_values(['day', 'mid', 'settle_ms', 'sid', 'px_rank', 'mk_idx', '_ro'],
                                                                     kind='mergesort')
    d = np.where(m3.self_trade.to_numpy(bool), 0.0, m3.s.to_numpy(float) * m3.q.to_numpy(float))
    cum = pd.Series(d, index=m3.index).groupby([m3.day, m3.mid]).cumsum()
    before = cum.to_numpy() - d
    s = m3.s.to_numpy(float)
    q = m3.q.to_numpy(float)
    cls = np.where(np.abs(before) < 1e-9, 'OPEN', np.where(np.sign(before) == s, 'ADD', np.where(q <= np.abs(before) + 1e-9, 'REDUCE', 'FLIP')))
    cls = np.where(m3.self_trade.to_numpy(bool), 'SELF', cls)
    return m3.assign(pos_before=before, inv=cls).drop(columns='_ro')


def main():
    t0 = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    OUT.mkdir(parents=True, exist_ok=True)
    outs = {n: OUT / n for n in ('legs_m3.parquet', 'cells_all.parquet', 'cells_m3.parquet', 'markets.parquet')}
    for p in outs.values():
        assert not p.exists(), f'write-once: {p.name} exists'
    idn = json.loads((V / 'econ/identities.json').read_text())['ex_ante']
    top10 = [r['addr'] for r in idn['top10_shares']]
    assert len(top10) == 10 and top10[2] == M3 and idn['day'] == '20261023'
    other = set(top10) - {M3}
    excl, rec = {}, dict(days={})
    all_cells, m3_legs = [], []
    for day in DAYS:
        mk = pd.read_parquet(V / f'markets_{day}.parquet', columns=['mid', 'ws', 'we', 'status'])
        mindex = {int(m): i for i, m in enumerate(sorted(mk.mid))}
        tob = load_tob(day, mindex)
        f = pq.read_table(V / 'econ/fills_valued.parquet', columns=FV_COLS, filters=[('day', '=', day)], use_threads=False,
                          read_dictionary=['addr', 'role', 'y_side', 't_status', 'st_res', 'st_1s', 'st_10s', 'st_60s', 'st_end', 'status']).to_pandas()
        n0 = len(f)
        cens = set(mk.mid[mk.status.astype(str).str.contains('boundary_censored')].astype(int))
        bad_status = (f.status != 'ok')
        bad_cens = f.mid.isin(cens) & ~bad_status
        bad_res = ~f.st_res.isin(['ok', 'tie']) & ~bad_status & ~bad_cens
        for k, b in (('status_not_ok', bad_status), ('boundary_censored', bad_cens), ('res_undefined', bad_res)):
            excl[f'{day}:{k}'] = int(b.sum())
        f = f[~(bad_status | bad_cens | bad_res)].copy()
        f['grp'] = np.where(f.addr == M3, 'MAKER3', np.where(f.addr.isin(other), 'OTHER_TOP10', 'FIELD'))
        keep = (f.role == 'maker') | (f.grp == 'MAKER3')
        f = f[keep].copy()
        midx = f.mid.map(mindex).to_numpy(np.int64)
        settle = f.settle_ms.to_numpy(np.int64)
        # P-clock pre-trade check on lifecycle rows: P must not exceed the lifecycle lower end lo (= t_ms on lo rows)
        lo_rows = (f.t_status == 'lo').to_numpy()
        rec['days'][day] = dict(rows_read=n0, rows_kept=len(f), lo_rows=int(lo_rows.sum()),
                                lo_rows_P_after_lo=int(((settle - P_LAG_MS) > f.t_ms.to_numpy(np.int64))[lo_rows].sum()),
                                lo_win_max_ms=int((settle - f.t_ms.to_numpy(np.int64))[lo_rows].max()) if lo_rows.any() else None)
        for clk, c in (('P', settle - P_LAG_MS), ('U', settle)):
            b = book_at(tob, midx, c)
            for k in ('cls', 'band', 'money', 'st'):
                f[f'{k}_{clk}'] = b[k]
            f[f'mid_{clk}'] = b['mid']
            f[f'spr_{clk}'] = b['spr']
            f[f'thin_{clk}'] = b['thin_side']
        f['tb'] = pd.cut(f.t_rel_s, TB_EDGES, right=False, labels=TB_LABELS).astype(str)
        guard(f'day {day} loaded')
        for clk in ('P', 'U'):
            g = f.assign(clock=clk, cls=f[f'cls_{clk}'], band=f[f'band_{clk}'], money=f[f'money_{clk}'],
                         edge_P=f.s * f.q * (f[f'mid_{clk}'] - f.p), out_P=f.s * f.q * (f.V - f[f'mid_{clk}']))
            all_cells.append(sums(g[g.role == 'maker'], ['grp', 'day', 'clock', 'tb', 'cls', 'band', 'money']))
        m3_legs.append(f[f.grp == 'MAKER3'].drop(columns=['grp']))
        del f, tob
        guard(f'day {day} done')
    legs = inventory(pd.concat(m3_legs, ignore_index=True))
    legs['edge_P'] = legs.s * legs.q * (legs.mid_P - legs.p)
    legs['out_P'] = legs.s * legs.q * (legs.V - legs.mid_P)
    cells_all = pd.concat(all_cells, ignore_index=True)
    m3c = []
    for clk in ('P', 'U'):
        g = legs.assign(clock=clk, cls=legs[f'cls_{clk}'], band=legs[f'band_{clk}'], money=legs[f'money_{clk}'],
                        edge_P=legs.s * legs.q * (legs[f'mid_{clk}'] - legs.p), out_P=legs.s * legs.q * (legs.V - legs[f'mid_{clk}']))
        m3c.append(sums(g, ['role', 'day', 'mid', 'clock', 'tb', 'cls', 'band', 'money', 'inv']))
    cells_m3 = pd.concat(m3c, ignore_index=True)
    # market level: late-window book profile from the stage-1 grid, joined to Maker3's whole-market economics
    grid = pd.read_parquet(OUT / 'grid.parquet')
    grid = grid[grid.st != 'boundary_censored'].copy()
    spr = grid.ask_c - grid.bid_c
    thin = np.minimum(grid.bid_sh, grid.ask_sh)
    ok2 = grid.st == 'ok'
    grid['cls'] = np.select([ok2 & (spr <= 1 + 1e-7) & (thin > THIN_SH), ok2 & (spr <= 1 + 1e-7) & (thin <= THIN_SH), ok2 & (spr > 1 + 1e-7),
                             grid.st == 'one_sided'], ['CALM', 'THIN', 'WIDE', 'ONE_SIDED'], 'UNKNOWN')
    late = grid[grid.k >= 240]
    prof = late.groupby(['day', 'mid']).cls.value_counts(normalize=True).unstack(fill_value=0.0).reindex(
        columns=['CALM', 'THIN', 'WIDE', 'ONE_SIDED', 'UNKNOWN'], fill_value=0.0).reset_index()
    g270 = grid[grid.k == 270][['day', 'mid', 'cls', 'bid_c', 'ask_c']]
    px = np.where(g270.cls.isin(['CALM', 'THIN', 'WIDE']), (g270.bid_c + g270.ask_c) / 2, np.where(np.isfinite(g270.bid_c), g270.bid_c, g270.ask_c))
    g270 = g270.assign(decided270=(g270.cls == 'ONE_SIDED') | (px <= 10) | (px >= 90))[['day', 'mid', 'decided270']]
    prof = prof.merge(g270, on=['day', 'mid'], how='left')
    vol = prof.THIN + prof.WIDE
    prof['late_class'] = np.select([prof.CALM >= 0.5, vol >= 0.5, prof.ONE_SIDED >= 0.5], ['CALM_LATE', 'VOLATILE_LATE', 'PINNED_LATE'], 'MIXED')
    mm = pq.read_table(V / 'econ/maker_market.parquet', filters=[('addr', '=', M3), ('day', 'in', list(DAYS))],
                       columns=['day', 'mid', 'status', 'res_status', 'n_mk', 'sh_mk', 'n_tk', 'sh_tk', 'gross_res_mk', 'gross_res_tk', 'fee_res',
                                'net_res', 'rt_pnl', 'resid_res', 'resid_drift', 'resid_resol', 'net_pos'], use_threads=False).to_pandas()
    markets = prof.merge(mm, on=['day', 'mid'], how='left', indicator=True)
    pos = legs[legs.inv != 'SELF'].assign(d=lambda x: x.s * x.q).sort_values(['day', 'mid', 'settle_ms'])
    for cut in (240, 270, 285):
        pc_ = pos[pos.t_rel_s < cut].groupby(['day', 'mid']).d.sum().rename(f'pos_{cut}')
        markets = markets.merge(pc_, on=['day', 'mid'], how='left')
    for name, df in (('legs_m3.parquet', legs), ('cells_all.parquet', cells_all), ('cells_m3.parquet', cells_m3), ('markets.parquet', markets)):
        df.to_parquet(outs[name], index=False)
    rec.update(exclusions=excl, maker3_legs=len(legs), maker3_roles=legs.role.value_counts().to_dict(), inv=legs.inv.value_counts().to_dict(),
               cells_all=len(cells_all), cells_m3=len(cells_m3), markets=len(markets), markets_m3_active=int((markets._merge == 'both').sum()),
               p_lag_ms=P_LAG_MS, thin_sh=THIN_SH, outputs={k: sha(p) for k, p in outs.items()},
               inputs={str(p.relative_to(R)): sha(p) for p in [V / 'econ/fills_valued.parquet', V / 'econ/maker_market.parquet',
                                                                V / 'econ/identities.json', OUT / 'grid.parquet']
                       + [V / f'{k}_{d}.parquet' for d in DAYS for k in ('markets', 'tob')]},
               cpu_s=round(time.process_time() - t0, 2), vmhwm_mib=guard('end'))
    (H / 'build.json').write_text(json.dumps(rec, indent=1, default=str) + '\n')
    print(json.dumps({k: rec[k] for k in ('maker3_legs', 'maker3_roles', 'inv', 'markets', 'markets_m3_active', 'cpu_s', 'vmhwm_mib')}, default=str))
    print(json.dumps(rec['days']))


if __name__ == '__main__':
    main()
