"""Entry-local data correction, adapted from verified stop-model fair_fix.

Own copy freezes the shared correction without importing a changing stop-model
training module. Adds corrected basis_bps alongside fair/gap. No stored step5
artifact is overwritten. Admissible book prehistory is 30 s, dates explicit.
"""
import json
import os
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import pf_kp_s4 as K

DAYS = ('20261025','20261026','20261027','20261031','20261101','20261102','20261103')
MUT_FAIR_DUP = 0
NAN = float('nan')

def relay_basis(rel_t, rel_v, t, min_n=5):
    """60 s median of relay / Binance ratios known in [t - 60 s, t] (step 5's window), >= min_n prints, else NaN."""
    a = np.searchsorted(rel_t, t - 60000.0); b = np.searchsorted(rel_t, t, side='right')
    return np.array([np.median(rel_v[x:y]) if y - x >= min_n else np.nan for x, y in zip(a, b)])


def fair_fix(day, W0):
    """Step 5's fair price with the relay basis rebuilt so that every relay print counts once (2026-11-15, s5check2: step 5's per-hour
    Binance passes include the prints of each hour's 10 s carry in two passes, biasing the 60 s median basis for steps within 60 s after an
    hour boundary - the first minute of every HH:00 window; up to 0.7 bps, a few cents of fair price). Same clocks as step 5: relay known at
    receipt - (72 - decision) ms, Binance at event + 1.5 + decision ms; same median window and minimum; bn_mid / sigma / time left / start
    price from step 5 and the registry (verified against raw data by s5check2). Returns ({market: (tau, p_bn, tau_rem, sigma, gap)}, stats)."""
    import pf_io as io
    from scipy.special import ndtr
    ck = json.load(open(f'{K.IO}/step1/{day}/clock_rl_d2.json')); dd = float(ck['d_dec']); b_in = float(ck['bn_in']) + dd; net = 72.0 - dd
    reg = pd.read_parquet(f'{K.IO}/step1/{day}/registry.parquet', columns=['mid', 'ws', 'start_price']); SP = dict(zip(reg.mid.astype(int), reg.start_price.astype(float)))
    WSd = dict(zip(reg.mid.astype(int), reg.ws.astype(float)))
    S = pd.read_parquet(f'{K.IO}/step5/{day}/rows_rl_d2_pk.parquet', columns=['mid', 'tau', 'bn_mid', 'tau_rem', 'sigma', 'p_bn', 'gap', 'basis_bps'])
    S = S[S.mid.isin(W0)]; S = S[S.tau.values < S.mid.map(WSd).values * 1000.0 + 61000.0].sort_values(['mid', 'tau'], kind='stable')
    d0 = pd.Timestamp(f'{day[:4]}-{day[4:6]}-{day[6:]}', tz='UTC'); hours = []
    for i in range(-1, 24):
        h = (d0 + pd.Timedelta(hours=i)).strftime('%Y%m%d_%H')
        if K.part_allowed(f'part_{h}.parquet', day):
            hours.append(h)
    REF = np.array(sorted((r[0] / 1e6 - net, float(r[2])) for h in hours for r in io.ref_rows(h) if not r[3]))
    rv = np.full(len(REF), np.nan); prev = None
    for h in hours:                                        # stream Binance hours; each print gets the mid known at its time, once
        p_ = f'{io.BN_DIR}/{h}.csv.gz'
        if not os.path.exists(p_):
            continue
        df = pd.read_csv(p_, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'u', 'bid', 'ask'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna(); df = df[(df.bid > 0) & (df.ask > df.bid)].drop_duplicates('u', keep='first')
        k = df.E.values.astype(np.float64) + b_in; o = np.argsort(k, kind='stable'); k = k[o]; m = ((df.bid.values + df.ask.values) / 2.0)[o]; del df
        if prev is not None:
            k = np.r_[prev[0], k]; m = np.r_[prev[1], m]
        sel = np.flatnonzero(np.isnan(rv) & (REF[:, 0] >= k[0]) & (REF[:, 0] <= k[-1]))
        if MUT_FAIR_DUP:
            sel = np.flatnonzero((REF[:, 0] >= k[0]) & (REF[:, 0] <= k[-1]))
        ii = np.searchsorted(k, REF[sel, 0], side='right') - 1; ok = ii >= 0
        rv[sel[ok]] = REF[sel[ok], 1] / m[ii[ok]]
        keep = k >= k[-1] - 10000.0; prev = (k[keep], m[keep])
    okr = np.isfinite(rv); rel_t = REF[okr, 0]; rel_v = rv[okr]
    bas = relay_basis(rel_t, rel_v, S.tau.values.astype(np.float64))
    sp = S.mid.map(SP).values.astype(np.float64); old = 1.0 + S.basis_bps.values.astype(np.float64) / 1e4
    den = S.sigma.values.astype(np.float64) * np.sqrt(np.maximum(S.tau_rem.values.astype(np.float64), 1e-3)); mid_ = S.bn_mid.values.astype(np.float64)
    # the correction is a DELTA on step 5's own p_bn (both evaluations from the same stored inputs), so rows whose basis is unchanged keep
    # step 5's value bit for bit (a full recomputation from float32-stored inputs moved every row by up to 0.0085 c - float32 noise)
    same = np.isfinite(bas) & np.isfinite(old) & (np.abs(bas - old) <= 1e-9)
    delta = 100.0 * (ndtr(np.log(mid_ * bas / sp) / den) - ndtr(np.log(mid_ * old / sp) / den))
    p5 = S.p_bn.values.astype(np.float64)
    pbn = np.where(same, p5, np.where(np.isfinite(old), p5 + delta, 100.0 * ndtr(np.log(mid_ * bas / sp) / den)))
    gapc = S.gap.values.astype(np.float64) + (pbn - p5)    # step 5's gap = p_bn - the known Predict mid
    dif = np.abs(pbn - S.p_bn.values) > 1e-4; hh = (S.mid.map(WSd).values % 3600.0) == 0.0
    st = dict(rows=int(len(S)), changed=float(np.nanmean(dif)), changed_hh00=float(np.nanmean(dif[hh])) if hh.any() else NAN,
              changed_other=float(np.nanmean(dif[~hh])) if (~hh).any() else NAN, max_abs_c=float(np.nanmax(np.abs(pbn - S.p_bn.values))) if len(S) else NAN,
              relay_prints=int(len(rel_t)))
    out = {}
    for mm_, g in pd.DataFrame({'mid': S.mid.values, 'tau': S.tau.values, 'p': pbn, 'tr': S.tau_rem.values, 'sg': S.sigma.values, 'gp': gapc, 'basis': np.where(same, S.basis_bps.values, (bas-1.)*1e4)}).groupby('mid', sort=False):
        out[int(mm_)] = (g.tau.values.astype(float).tolist(), g.p.values, g.tr.values, g.sg.values, g.gp.values, g.basis.values)
    return out, st


def load_day(day):
    if day not in DAYS:
        raise ValueError('undeclared day')
    read = pd.read_parquet
    def guarded(path, *args, **kwargs):
        p = Path(path)
        if p.parent.name == 'books' and not K.part_allowed(str(p), day):
            raise ValueError('inadmissible book before read')
        return read(path, *args, **kwargs)
    with mock.patch.object(K, 'SKIP_FOREIGN_PARTS', True), mock.patch.object(K, 'PREROLL_MS', 30000.), mock.patch.object(pd, 'read_parquet', guarded):
        data = K.load_day(day)
    sealed = []
    if day == '20261031':
        bound = K._ms('2026-10-31') + 1800000.
        sealed = [m for m,t in data['W0'].items() if t < bound]
        data['W0'] = {m:t for m,t in data['W0'].items() if m not in sealed}
    fixed, stats = fair_fix(day, data['W0'])
    for m in data['W0']:
        if m not in data['ROWS']:
            continue
        x=data['ROWS'][m]; corr=fixed[m]
        if not np.array_equal(x.tau.to_numpy(float), np.asarray(corr[0])):
            raise ValueError('fair correction timestamp mismatch')
        x.loc[:,'gap']=corr[4]; x.loc[:,'basis_bps']=corr[5]
    stats.update(preroll_ms=30000., sealed_lookback_windows=sealed)
    return data, stats
