"""The frozen chase line's decision delay: research's replay decides every tick at t + FEED + DECIDE (3.5 ms after the tick's Binance
E); a live port can decide only once the tick's bookTicker frame has arrived, which in bursts is later (USER 2026-11-28: "Yes do the
request commit then do the local work"; the local work = re-price the frozen line for what live cannot have, stage 2b's research
finding). P-2026-005, public data, no orders. Declared before the run. A re-pricing of the frozen candidate on the forward days it was
tested on (11-20 .. 11-26, read by this line on 11-27), not a new test: nothing is chosen, there is no PASS rule; the selections, limits
and models are the frozen run's, on the primary selection's EARLY rows.
Tokyo's stage-1 hour (its receipts, as chase_tokyo_2b converted them; research's frozen rows on it, research_rows.parquet), readings:
  T1 information: per decision rule, the trades with E in (t - 300, t] (both sides) and the depth20 frames with E in (t - 1 s, t] not
     yet received at the decision instant, and the share of the its-way 100 ms volume. Rules: G = t + FEED + DECIDE (stage 2b's; its
     8.1% is the positive control and must reproduce chase_tokyo_2b_diagnose.json), and D_w = max(t + FEED + DECIDE, f + w), f = Tokyo's
     receipt of the tick ms's last bookTicker frame, w in W. The rule priced is D_2 (w = DECIDE): research's own model with the measured
     frame receipt in place of FEED. Its inputs are research's (E <= t) only if T1 finds nothing late under it; the exits stage refuses
     otherwise. Known-bad: the trades received 3 ms later must make D_2 find late trades.
  T2 the delay X = D_2 - (t + FEED + DECIDE) = max(0, lambda - FEED), lambda = f - t, at the decision points and the actions.
  Covariate (declared; chosen on Tokyo's hour by Spearman with lambda among the ELIGIBLE: the trade counts over 5 / 20 / 50 ms, which
  research's trade tape gives as Tokyo's does (the exits stage reads it); recorded, not eligible: bookTicker frame counts, book-update-id
  counts (u's running maximum; they would need research's full bookTicker tape read again for each day, ~1 CPU-min) and the move's
  ticks): n_tr_20 = trades (de-duplicated by id, p, q > 0) with E in (t - 20, t]. Strata: its quintiles at Tokyo's decision points; a
  stratum's X sample = its decision points' X. (Revised before any P&L: the first declaration made every count eligible and the stage
  refused when a book-update count correlated more; that count is still reported.)
Research's days: each action takes its stratum's X sample. Its P&L at an entry delay x: chase_forward_tokyo.exits_day_tokyo with the big-
move configs' entry delays Le_k + x, x in GRID (Le enters the entry match and the exit floor as t does), the EARLY scope re-applied at
t + x, and the burst after the send (chase_move_size.move_stats) at t + x for R2 / R3 / R5; linear in x between grid points, flat beyond
the last (counted). Readings (chase_forward_bigmove's R1..R5), each: (a) x = 0 (must equal chase_forward_bigmove.json's), (b) every
uniform x of GRID, (c) MATCHED: each action's expectation over its stratum's X sample (the estimate), (d) MATCHED_X2: every X doubled
(research's days busier than Tokyo's Saturday hour); for R1 and R5, CAL_DRAWS bootstrap draws of Tokyo's hour by 1-minute blocks: the
spread from the one-hour calibration.
Limits declared: one Tokyo hour (Saturday 10-11 UTC, 6,370 decision points, 28 actions); research's actions above Tokyo's covariate
range take the top stratum (counted); the Predict inputs and the limit stay at t + 3.5 ms's known frame (a later decision may know a
newer one); the action set is the frozen run's (T1: the inputs are unchanged under D_2).
Parity: the x = 0 columns equal chase_forward_bigmove's stored exits on the same rows (the exits stage refuses to write otherwise); the
read's x = 0 readings equal chase_forward_bigmove.json's.
Stages: selftest | tokyo | exits <day> | read.  Outputs (write-once): chase_decision_delay_tokyo.json, OUT/chase_forward/exits_delay/
<day>.parquet + <day>.json, chase_decision_delay.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_decision_delay.py <stage> ...
"""
import datetime as dt
import hashlib
import json
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_forward as CF  # noqa: E402
import chase_forward_bigmove as CFB  # noqa: E402
import chase_forward_latency as CFL  # noqa: E402
import chase_forward_tokyo as CFT  # noqa: E402
import chase_move_size as CMS  # noqa: E402

EF, C, TR, CT, TM, DC = CF.EF, CF.C, CF.TR, CF.CT, CF.TM, CF.DCm
SRC = CF.WT.OUT / 'chase_tokyo_2b'
OUT_D = CF.FOUT / 'exits_delay'
TOKYO, RESULT, DIAG = H / 'chase_decision_delay_tokyo.json', H / 'chase_decision_delay.json', H / 'chase_tokyo_2b_diagnose.json'
SEL = CFB.SEL
W = (0.34, 0.5, 1.0, 1.66, 2.0)
PRICED = 'D_2'
GRID = (0., 2., 8., 30., 120.)
COV, NSTRATA, ELIGIBLE = 'n_tr_20', 5, ('n_tr_5', 'n_tr_20', 'n_tr_50')
CAL_DRAWS, SEED, BLOCK_MS = 200, 20261130, 60000.
READINGS = ('R1_BIG', 'R2_BIG_MOVE', 'R3_BIG_BURST', 'R4_BOTH', 'R5_BOTH_BURST')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ instruments
def late(t, d, D, tr, dp):
    """Per decision (t, d) and instant D (Tokyo's clock, ms): trades with E in (t - 300, t] and depth20 frames with E in (t - 1000, t]
    received after D; the its-way volume with E in (t - 100, t] received after D, and its share of that volume."""
    E_, R_, q_, buy = tr.E.to_numpy(float), tr.r.to_numpy(float) / 1e6, tr.q.to_numpy(float), tr.m.to_numpy() == DC.BUY_AGGR
    dE, dR = dp.E.to_numpy(float), dp.r.to_numpy(float) / 1e6
    n_tr = rows_tr = n_dp = rows_dp = rows_v = 0
    v_late = v_all = 0.
    for ti, di, Di in zip(np.asarray(t, float), np.asarray(d, float), np.asarray(D, float)):
        a, b = np.searchsorted(E_, ti - 300., 'right'), np.searchsorted(E_, ti, 'right')
        k = int(np.sum(R_[a:b] > Di))
        n_tr += k
        rows_tr += k > 0
        a1 = np.searchsorted(E_, ti - 100., 'right')
        its = buy[a1:b] if di > 0 else ~buy[a1:b]
        lv = float(q_[a1:b][its & (R_[a1:b] > Di)].sum())
        v_late += lv
        v_all += float(q_[a1:b][its].sum())
        rows_v += lv > 0
        a, b = np.searchsorted(dE, ti - 1000., 'right'), np.searchsorted(dE, ti, 'right')
        k = int(np.sum(dR[a:b] > Di))
        n_dp += k
        rows_dp += k > 0
    return dict(trades_late=int(n_tr), rows_trades_late=int(rows_tr), its100_volume_late_btc=v_late,
                its100_volume_late_share=v_late / v_all if v_all > 0 else None, rows_its100_late=int(rows_v),
                depth20_late=int(n_dp), rows_depth20_late=int(rows_dp))


def count_in(E_, t, w):
    """Events of the sorted times E_ in (t - w, t]."""
    return (np.searchsorted(E_, t, 'right') - np.searchsorted(E_, t - w, 'right')).astype(float)


def hat_weights(x, grid=GRID):
    """Mean linear-interpolation weights of the values x onto the grid, flat beyond its ends -> (weights, values beyond the last)."""
    g, x = np.asarray(grid, float), np.asarray(x, float)
    C.require(len(x) > 0, 'an empty sample')
    xc = np.clip(x, g[0], g[-1])
    j = np.clip(np.searchsorted(g, xc, 'right') - 1, 0, len(g) - 2)
    a = (xc - g[j]) / (g[j + 1] - g[j])
    w = np.zeros(len(g))
    np.add.at(w, j, 1. - a)
    np.add.at(w, j + 1, a)
    return w / len(x), int(np.sum(x > g[-1]))


def strata_of(v, edges):
    return np.searchsorted(np.asarray(edges, float), np.asarray(v, float), 'right')


def stratum_weights(X, sb, n=NSTRATA, fallback=None):
    """Per stratum the hat weights of its X sample (a stratum without sample points takes fallback's row, counted)."""
    rows, clamped, empty = [], 0, 0
    for b in range(n):
        xs = X[sb == b]
        if not len(xs):
            C.require(fallback is not None, f'stratum {b} has no sample points')
            rows.append(fallback[b])
            empty += 1
            continue
        w, c = hat_weights(xs)
        rows.append(w)
        clamped += c
    return np.stack(rows), clamped, empty


# ------------------------------------------------------------------------------------------------------------------ tokyo
def tokyo_frames(raw):
    rd = lambda k, cols, names: pd.concat([pd.read_csv(f, header=None, usecols=cols, names=names)  # noqa: E731
                                           for f in sorted((raw / k / 'BTCUSDT').glob('*.csv.gz'))], ignore_index=True)
    bt = rd('bookTicker', [0, 1, 3], ['r', 'E', 'u'])
    tr = rd('trade', [0, 1, 3, 4, 5, 6], ['r', 'E', 'id', 'p', 'q', 'm'])
    dp = rd('depth20', [0, 1], ['r', 'E'])
    tr = tr.sort_values('r', kind='stable').drop_duplicates('id')                 # a trade id's first receipt
    tr = tr[(tr.p > 0) & (tr.q > 0)].sort_values('E', kind='stable').reset_index(drop=True)
    return bt, tr, dp.sort_values('E', kind='stable').reset_index(drop=True)


QN = (('p10', .1), ('p25', .25), ('p50', .5), ('p75', .75), ('p90', .9), ('p95', .95), ('p99', .99))


def quant(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), mean=float(x.mean()), **{k: float(np.quantile(x, q)) for k, q in QN}, max=float(x.max()), share_positive=float(np.mean(x > 0)))


def stage_tokyo():
    C.require(not TOKYO.exists(), f'{TOKYO.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    rows_fp, raw = SRC / 'research_rows.parquet', SRC / 'data' / 'raw'
    X = pd.read_parquet(rows_fp, columns=['t', 'd', 'act', 'early', 'mv_ticks_50', 'mv_since_ms'])
    bt, tr, dp = tokyo_frames(raw)
    t, d = X.t.to_numpy(float), X.d.to_numpy(float)
    f = X.t.map(bt.groupby('E').r.max()).to_numpy(float) / 1e6
    C.require(np.isfinite(f).all(), "a decision point without its tick ms's bookTicker frame in Tokyo's recording")
    lam = f - t
    g0 = t + TM.FEED + TM.DECIDE
    rules = dict(G=g0, **{f'D_{w:g}': np.maximum(g0, f + w) for w in W})
    C.require(PRICED in rules and float(PRICED.split('_')[1]) == TM.DECIDE, 'the priced rule is D_DECIDE')
    T1 = {k: late(t, d, v, tr, dp) for k, v in rules.items()}
    dg = json.loads(DIAG.read_text())['its_100']['research_volume_received_after_its_decision_time']
    pc = dict(rows=T1['G']['rows_its100_late'], rows_2b=dg['rows'], share=T1['G']['its100_volume_late_share'], share_2b=dg['share_of_research_volume'])
    C.require(pc['rows'] == pc['rows_2b'] and abs(pc['share'] - pc['share_2b']) < 1e-6, f'the positive control did not reproduce stage 2b: {pc}')
    kb = late(t, d, rules[PRICED], tr.assign(r=tr.r + 3_000_000), dp)
    C.require(kb['trades_late'] > 0, f'the known-bad tape (trades 3 ms later) found nothing late under {PRICED}: the instrument cannot fire')
    clean = T1[PRICED]['trades_late'] == 0 and T1[PRICED]['depth20_late'] == 0
    Xd = np.maximum(lam - TM.FEED, 0.)
    A = X.act.to_numpy(bool) & X.early.to_numpy(bool)
    trE = tr.E.to_numpy(float)
    bts = bt.sort_values(['E', 'u'], kind='stable')
    bE, bu = bts.E.to_numpy(float), np.maximum.accumulate(bts.u.to_numpy(float))      # the largest update id received with E <= a time
    cov = {}
    for w in (5., 20., 50.):
        cov[f'n_tr_{int(w)}'] = count_in(trE, t, w)
        cov[f'n_bt_{int(w)}'] = count_in(bE, t, w)
        j1, j0 = np.searchsorted(bE, t, 'right') - 1, np.searchsorted(bE, t - w, 'right') - 1
        cov[f'du_{int(w)}'] = np.where(j0 >= 0, bu[np.maximum(j1, 0)] - bu[np.maximum(j0, 0)], np.nan)
    cov['mv_ticks_50'], cov['mv_since_ms'] = X.mv_ticks_50.to_numpy(float), X.mv_since_ms.to_numpy(float)
    rho = {}
    for k, v in cov.items():
        ok = np.isfinite(v)
        rho[k] = float(spearmanr(v[ok], lam[ok])[0])
    C.require(max(ELIGIBLE, key=rho.get) == COV, f'the declared covariate {COV} is not the best eligible by Spearman: {rho}')
    v = cov[COV]
    edges = [float(e) for e in np.quantile(v, [i / NSTRATA for i in range(1, NSTRATA)])]
    sb = strata_of(v, edges)
    blk = np.floor((t - t.min()) / BLOCK_MS).astype(int)
    by = {str(b): dict(points=int(np.sum(sb == b)), actions=int(np.sum(A & (sb == b))), lam=quant(lam[sb == b]), X=quant(Xd[sb == b]))
          for b in range(NSTRATA)}
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), inputs=dict(research_rows_sha256=sha(rows_fp), tokyo_recording='pmmmrust data/tokyo-chase-stage1-20261128 (0fec30b)',
               raw_files={k: len(list((raw / k / 'BTCUSDT').glob('*.csv.gz'))) for k in ('bookTicker', 'trade', 'depth20')},
               frames=dict(bookTicker=int(len(bt)), trades=int(len(tr)), depth20=int(len(dp)))),
               decision_points=int(len(X)), actions=int(A.sum()), feed_ms=TM.FEED, decide_ms=TM.DECIDE,
               T1=T1, T1_positive_control=pc, T1_known_bad_trades_3ms_later=kb, priced_rule=PRICED, priced_rule_clean=bool(clean),
               lam=dict(points=quant(lam), actions=quant(lam[A])), X=dict(points=quant(Xd), actions=quant(Xd[A])),
               covariates=dict(declared=COV, eligible=list(ELIGIBLE), spearman_with_lam=rho, best_overall=max(rho, key=rho.get)),
               strata=dict(edges=edges, by_stratum=by, blocks=int(blk.max() + 1), covariate_points=quant(v)),
               samples=dict(X=[round(float(x), 6) for x in Xd], stratum=[int(b) for b in sb], block=[int(b) for b in blk]))
    TR.dump(TOKYO, out)
    print('T1', {k: (r['trades_late'], r['depth20_late'], round(r['its100_volume_late_share'], 5)) for k, r in T1.items()}, 'clean', clean,
          'X points', {k: round(v_, 2) for k, v_ in quant(Xd).items() if k in ('mean', 'p50', 'p90', 'p99')}, 'rho', {k: round(v_, 3) for k, v_ in rho.items()},
          'edges', edges, flush=True)


# ------------------------------------------------------------------------------------------------------------------ exits
def configs_delay(ms):
    """Entries D{k}X{i}: the big-move entry delay k plus GRID[i]; exits D{k}X{i}_S{j}: its settlement floor j, TAKE_C1000 (as the big-move's)."""
    nk = len(ms['Le'])
    ent = [dict(name=f'D{k}X{i}', Le=float(ms['Le'][k] + x)) for i, x in enumerate(GRID) for k in range(nk)]
    ext = [dict(name=f'D{k}X{i}_S{j}', entry=f'D{k}X{i}', Sa=s, Sb=0., Lx=ms['Lx'], pols=('TAKE_C1000',))
           for i in range(len(GRID)) for k in range(nk) for j, s in enumerate(ms['S'])]
    names = [x['name'] for x in ext]
    C.require(len(set(names)) == len(names) and len({e['name'] for e in ent}) == len(ent), 'config names')
    C.require(all(e['Le'] == ms['Le'][k] for k, e in enumerate(ent[:nk])), 'the x = 0 entries are not the big-move delays exactly')
    return ent, ext


def parity0(new, old, nk, nS):
    """The x = 0 columns of new against the big-move columns of old (same rows), exactly (NaN = NaN). -> mismatched column names."""
    pairs = [('cut_minus_act_ms', 'cut_minus_act_ms'), ('t', 't'), ('d', 'd'), ('mid', 'mid'), ('lim', 'lim')]
    for k in range(nk):
        for s in (3, 10):
            pairs.append((f'D{k}X0_AB_s{s}_fill', f'D{k}_AB_s{s}_fill'))
            pairs += [(f'D{k}X0_S{j}_AB_s{s}_TAKE_C1000_pnl', f'D{k}_S{j}_AB_s{s}_TAKE_C1000_pnl') for j in range(nS)]
    pairs += [('mv_up100_X0', 'mv_up100'), ('mv_n_post200_X0', 'mv_n_post200')]
    if len(new) != len(old):
        return ['rows differ']
    return [a for a, b in pairs if a not in new.columns or b not in old.columns or not CFL.same_col(new[a].reset_index(drop=True), old[b].reset_index(drop=True))]


def stage_exits(day):
    C.require(day in CF.FWD, f'{day} is not a declared forward day')
    fp, fj = OUT_D / f'{day}.parquet', OUT_D / f'{day}.json'
    C.require(not fp.exists() and not fj.exists(), f'{fp.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    tk = json.loads(TOKYO.read_text())
    C.require(tk['priced_rule'] == PRICED and tk['priced_rule_clean'], f"Tokyo's hour found late inputs under {PRICED}: the inputs are not research's; not priced")
    CF.frozen()
    ms = CFB.measured()
    nk, nS = len(ms['Le']), len(ms['S'])
    ent, ext = configs_delay(ms)
    CF.forward_mode((day,))
    S = pd.read_parquet(CF.FOUT / 'forward_selections.parquet')
    S = S[S[SEL].to_numpy(bool) & (S.day == day).to_numpy()].reset_index(drop=True)
    old = pd.read_parquet(CF.FOUT / 'exits_bigmove' / f'{day}.parquet')
    C.require(np.array_equal(old.t.to_numpy(float), S.t.to_numpy(float)) and np.array_equal(old.d.to_numpy(float), S.d.to_numpy(float)),
              "the stored big-move exits are not the selection's rows")
    early = old.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS
    Se, oe = S[early].reset_index(drop=True), old[early].reset_index(drop=True)
    X, st = CFT.exits_day_tokyo(day, Se, ent, ext)
    C.require(np.array_equal(X.t.to_numpy(float), Se.t.to_numpy(float)) and np.array_equal(X.d.to_numpy(float), Se.d.to_numpy(float)), 'row order')
    t, dd = Se.t.to_numpy(float), Se.d.to_numpy(float)
    Eb, Mb = CT.mid_series(day)
    for i, x in enumerate(GRID):
        mv = CMS.move_stats(Eb, Mb, t + x, dd)
        X[f'mv_up100_X{i}'], X[f'mv_n_post200_X{i}'] = mv['up100'], mv['n_post200']
    bad = parity0(X, oe, nk, nS)
    C.require(not bad, f'x = 0 parity with the stored big-move exits failed on {day}: {bad[:8]}')
    tr = DC.read_trades(day).drop_duplicates('id')
    tr = tr[(tr.p > 0) & (tr.q > 0)]
    X[COV] = count_in(np.sort(tr.E.to_numpy(float)), t, float(COV.split('_')[-1]))
    del tr
    keep = list(Se.columns) + ['cut_minus_act_ms', COV] + [f'mv_{k}_X{i}' for i in range(len(GRID)) for k in ('up100', 'n_post200')]
    keep += [f'D{k}X{i}_AB_s3_fill' for i in range(len(GRID)) for k in range(nk)]
    keep += [f'D{k}X{i}_S{j}_AB_s3_TAKE_C1000_pnl' for i in range(len(GRID)) for k in range(nk) for j in range(nS)]
    X = X[list(dict.fromkeys(keep))]
    cv = X[COV].to_numpy(float)
    edges = tk['strata']['edges']
    st.update(rows_selected=int(len(S)), rows_early=int(len(Se)), parity=dict(columns=4 + 2 * nk * (1 + nS) + 2, mismatched=bad),
              covariate=dict(name=COV, actions=quant(cv), by_stratum=np.bincount(strata_of(cv, edges), minlength=NSTRATA).tolist(),
                             above_tokyo_p99=int(np.sum(cv > tk['strata']['covariate_points']['p99'])),
                             above_tokyo_max=int(np.sum(cv > tk['strata']['covariate_points']['max']))),
              grid=list(GRID), grid_drops=dict(CF.GRID_DROPS))
    OUT_D.mkdir(parents=True, exist_ok=True)
    X.to_parquet(fp, index=False)
    TR.dump(fj, st)
    print(day, json.dumps(TR.clean({k: v for k, v in st.items() if k in ('actions', 'rows_early', 'parity', 'covariate', 'unloaded')})), flush=True)


# ------------------------------------------------------------------------------------------------------------------ read
def stage_read():
    C.require(not RESULT.exists(), f'{RESULT.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    rc = CF.frozen()
    p, sel_p, s_p = CFL.primary(rc)
    C.require(sel_p == SEL, f"the receipt's primary selection is {sel_p}, the rows here are {SEL}")
    s = int(s_p)
    tk = json.loads(TOKYO.read_text())
    ms = CFB.measured()
    days = tuple(d for d in CF.FWD if (OUT_D / f'{d}.parquet').exists())
    C.require(days == CF.FWD, f'exits_delay has {len(days)} of {len(CF.FWD)} days')
    X = pd.concat([pd.read_parquet(OUT_D / f'{d}.parquet') for d in days], ignore_index=True)
    keys, dayk = [], []
    for d in days:
        reg = CF.MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]
        dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    CF.SX.DAYS = days
    per_day = len(keys) / len(days)
    wk = (X.day.astype(str) + ':' + X.mid.astype(np.int64).astype(str)).to_numpy()
    nk, nS, ng = len(ms['A']), len(ms['S']), len(GRID)
    big = [k for k, r in enumerate(ms['rows']) if r['probe'] == 'big']
    allk = list(range(nk))
    cov = {k_: np.array([r[k_] for r in ms['rows']], float) for k_ in ('up100', 'n_post200')}
    cma = X.cut_minus_act_ms.to_numpy(float)
    m0 = cma >= EF.EARLY_MS
    C.require(m0.all(), 'a row outside the EARLY scope at x = 0')
    every = lambda ks: np.tile(np.isin(np.arange(nk), ks), (len(X), 1))  # noqa: E731
    V = {r: np.zeros((ng, len(X))) for r in READINGS}
    Fv = {r: np.zeros((ng, len(X))) for r in READINGS}
    scope = []
    for gi, x in enumerate(GRID):
        P = np.stack([np.stack([X[f'D{k}X{gi}_S{j}_AB_s{s}_TAKE_C1000_pnl'].to_numpy(float) for j in range(nS)]) for k in range(nk)]).mean(axis=1)
        F = np.stack([(np.nan_to_num(X[f'D{k}X{gi}_AB_s{s}_fill'].to_numpy(float)) > 0).astype(float) for k in range(nk)])
        mx = (cma - x) >= EF.EARLY_MS
        scope.append(int(mx.sum()))
        up, npost = X[f'mv_up100_X{gi}'].to_numpy(float), X[f'mv_n_post200_X{gi}'].to_numpy(float)
        sets = dict(R1_BIG=every(big), R2_BIG_MOVE=CFB.strata_sets(up, cov['up100'][big], big, nk, up[mx])[0],
                    R3_BIG_BURST=CFB.strata_sets(npost, cov['n_post200'][big], big, nk, npost[mx])[0], R4_BOTH=every(allk),
                    R5_BOTH_BURST=CFB.strata_sets(npost, cov['n_post200'], allk, nk, npost[mx])[0])
        for r, st_ in sets.items():
            C.require(st_ is not None, f'{r} is not computable at x = {x:g}')
            pn, ff = CFB.set_mean(P, F, st_)
            V[r][gi] = np.where(mx, np.nan_to_num(pn), 0.)
            Fv[r][gi] = np.where(mx, ff, 0.)
    stats = lambda v, f: CFT.stats_of(v, m0, f, wk, keys, dayk, per_day)  # noqa: E731
    B = json.loads(CFB.RESULT.read_text())
    edges = tk['strata']['edges']
    bA = strata_of(X[COV].to_numpy(float), edges)
    Xs, sb, blk = (np.array(tk['samples'][k]) for k in ('X', 'stratum', 'block'))
    W1, cl1, _ = stratum_weights(Xs, sb)
    W2, cl2, _ = stratum_weights(2. * Xs, sb)
    readings, par = {}, {}
    for r in READINGS:
        uni = {f'x{x:g}': stats(V[r][gi], Fv[r][gi]) for gi, x in enumerate(GRID)}
        b0 = B['readings'][r]
        u0 = json.loads(json.dumps(TR.clean(uni['x0'])))
        par[r] = [k for k in ('total_usd', 'days_positive', 'sharpe_day_equiv', 'filled', 'daily_usd', 'actions') if u0[k] != b0[k]]
        mt = {}
        for nm, Wb in (('MATCHED', W1), ('MATCHED_X2', W2)):
            wa = Wb[bA]                                                                     # actions x grid
            mt[nm] = stats(np.sum(wa.T * V[r], axis=0), np.sum(wa.T * Fv[r], axis=0))
            mt[nm]['delta_vs_x0_usd'] = mt[nm]['total_usd'] - uni['x0']['total_usd']
        readings[r] = dict(uniform=uni, **mt)
    C.require(not any(par.values()), f"the x = 0 readings differ from chase_forward_bigmove.json's: {par}")
    rng = np.random.default_rng(SEED)
    nb = int(blk.max() + 1)
    draws = {r: [] for r in ('R1_BIG', 'R5_BOTH_BURST')}
    empty_draws = 0
    for _ in range(CAL_DRAWS):
        pick = rng.integers(0, nb, nb)
        idx = np.concatenate([np.flatnonzero(blk == b) for b in pick])
        Wd, _, e = stratum_weights(Xs[idx], sb[idx], fallback=W1)
        empty_draws += e > 0
        for r in draws:
            draws[r].append(float(np.sum(np.nan_to_num(np.sum(Wd[bA].T * V[r], axis=0)[m0])) / 100.))
    q = lambda v: dict(p5=float(np.quantile(v, .05)), median=float(np.median(v)), p95=float(np.quantile(v, .95)), share_positive=float(np.mean(np.array(v) > 0)))  # noqa: E731
    ex = {f'stratum_{b}': dict(actions=int(np.sum(bA == b)), mean_X_ms=float(np.mean(Xs[sb == b]))) for b in range(NSTRATA)}
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), receipt=CF.receipt_path().name, primary=p, days=list(days), grid_ms=list(GRID),
               tokyo=dict(priced_rule=tk['priced_rule'], priced_rule_clean=tk['priced_rule_clean'], T1={k: tk['T1'][k] for k in ('G', PRICED)},
                          X_points=tk['X']['points'], X_actions=tk['X']['actions'], covariate=COV, edges=edges),
               actions=int(m0.sum()), early_scope_by_x={f'x{x:g}': n for x, n in zip(GRID, scope)}, actions_by_stratum=ex,
               expected_X_ms=float(np.mean([ex[f'stratum_{b}']['mean_X_ms'] for b in bA])), clamped_sample_points=dict(MATCHED=cl1, MATCHED_X2=cl2),
               readings=readings, calibration_draws=dict(n=CAL_DRAWS, seed=SEED, block_ms=BLOCK_MS, empty_stratum_draws=int(empty_draws),
                                                         **{r: dict(total_usd=q(v)) for r, v in draws.items()}),
               parity_x0={r: dict(mismatched=v) for r, v in par.items()},
               bigmove_x0={r: {k: B['readings'][r][k] for k in ('total_usd', 'days_positive', 'sharpe_day_equiv')} for r in READINGS})
    TR.dump(RESULT, out)
    print(json.dumps(TR.clean({r: dict(x0=round(v['uniform']['x0']['total_usd'], 1), matched=round(v['MATCHED']['total_usd'], 1), x2=round(v['MATCHED_X2']['total_usd'], 1),
                                       days=v['MATCHED']['days_positive'], sharpe=v['MATCHED']['sharpe_day_equiv'],
                                       uniform={k: round(u['total_usd'], 1) for k, u in v['uniform'].items()}) for r, v in readings.items()})),
          'draws', {r: q(v) for r, v in draws.items()}, flush=True)


# ------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    # late(): a receipt exactly at D is not late, after D is; the window edges; the its-way share; depth frames
    tr = pd.DataFrame(dict(E=[700., 950., 999., 1000.], r=[1004e6, 1010e6, 1003.5e6, 1003.6e6], q=[5., 2., 1., 1.], m=[0, 0, 0, 1]))
    dp = pd.DataFrame(dict(E=[990., 995.], r=[1003.4e6, 1004e6]))
    L = late([1000.], [1.], [1003.5], tr, dp)
    C.require(L['trades_late'] == 2 and L['depth20_late'] == 1 and abs(L['its100_volume_late_share'] - 2. / 3.) < 1e-12 and L['rows_its100_late'] == 1, f'late {L}')
    L = late([1000.], [-1.], [1003.5], tr, dp)
    C.require(L['its100_volume_late_share'] == 1. and L['trades_late'] == 2, f'late, d = -1: {L}')
    L = late([1000.], [1.], [1003.5], tr.assign(r=tr.r + 3e6), dp)
    C.require(L['trades_late'] == 3, f'late, receipts 3 ms later: {L}')
    C.require(late([1000.], [1.], [1100.], tr, dp)['trades_late'] == 0, 'nothing is late at a late enough decision')
    # hat weights: on grid points, between them, beyond the last (clamped, counted)
    w, c = hat_weights([0., 2., 120.])
    C.require(np.allclose(w, [1 / 3, 1 / 3, 0, 0, 1 / 3]) and c == 0, f'hat on points {w}')
    w, c = hat_weights([1., 5., 75., 500.])
    C.require(np.allclose(w, np.array([.5, .5 + .5, .5, .5, 1. + .5]) / 4.) and c == 1, f'hat between {w} {c}')
    C.require(abs(np.dot(hat_weights([5.])[0], [0., 2., 8., 30., 120.]) - 5.) < 1e-12, 'linear interpolation reproduces x')
    # strata: edge values go up, empty strata fall back (counted), or are refused without a fallback
    C.require(strata_of([0, 3, 4, 9], [3., 6.]).tolist() == [0, 1, 1, 2], 'strata')
    Wd, _, e = stratum_weights(np.array([1., 2.]), np.array([0, 0]), fallback=np.tile(hat_weights([0.])[0], (NSTRATA, 1)))
    C.require(e == NSTRATA - 1 and np.allclose(Wd[0], hat_weights([1., 2.])[0]), f'stratum fallback {e}')
    try:
        stratum_weights(np.array([1.]), np.array([0]))
        raise AssertionError('an empty stratum without a fallback was accepted')
    except ValueError as e_:
        assert 'no sample points' in str(e_), e_
    # configs: 30 delays x GRID entries, x 35 floors exits, the x = 0 entries exactly the big-move delays
    ms = CFB.measured()
    ent, ext = configs_delay(ms)
    C.require(len(ent) == len(GRID) * 30 and len(ext) == len(GRID) * 30 * 35, f'config counts {len(ent)} {len(ext)}')
    C.require(all(abs(e['Le'] - (ms['Le'][k % 30] + GRID[k // 30])) < 1e-12 for k, e in enumerate(ent)), 'entry delays')
    # parity0 flags a changed value, a missing column and a row count
    a = pd.DataFrame({'cut_minus_act_ms': [1., 2.], 't': [1., 2.], 'd': [1., -1.], 'mid': [5, 6], 'lim': [40., 50.], 'mv_up100_X0': [1., 2.], 'mv_n_post200_X0': [3., 4.]})
    b = a.rename(columns={'mv_up100_X0': 'mv_up100', 'mv_n_post200_X0': 'mv_n_post200'})
    for s_ in (3, 10):
        a[f'D0X0_AB_s{s_}_fill'] = [1., 0.]
        b[f'D0_AB_s{s_}_fill'] = [1., 0.]
        a[f'D0X0_S0_AB_s{s_}_TAKE_C1000_pnl'] = [np.nan, 2.]
        b[f'D0_S0_AB_s{s_}_TAKE_C1000_pnl'] = [np.nan, 2.]
    C.require(parity0(a, b, 1, 1) == [], f'parity0 on equal frames: {parity0(a, b, 1, 1)}')
    b2 = b.copy()
    b2.loc[1, 'D0_S0_AB_s3_TAKE_C1000_pnl'] = 2.0000001
    C.require(parity0(a, b2, 1, 1) == ['D0X0_S0_AB_s3_TAKE_C1000_pnl'], 'parity0 missed a changed value')
    C.require('D0X0_AB_s10_fill' in parity0(a.drop(columns='D0X0_AB_s10_fill'), b, 1, 1), 'parity0 missed a missing column')
    C.require(parity0(a.iloc[:1], b, 1, 1) == ['rows differ'], 'parity0 missed a row count')
    # stage guards: an undeclared day, write-once outputs
    try:
        stage_exits('20261101')
        raise AssertionError('stage_exits accepted a consumed day')
    except ValueError as e_:
        assert 'not a declared forward day' in str(e_), e_
    global RESULT, TOKYO
    keep = RESULT, TOKYO
    with tempfile.NamedTemporaryFile(suffix='.json') as tf:
        RESULT = TOKYO = Path(tf.name)
        try:
            for fn in (stage_read, stage_tokyo):
                try:
                    fn()
                    raise AssertionError(f'{fn.__name__} overwrote its result')
                except ValueError as e_:
                    assert 'write-once' in str(e_), e_
        finally:
            RESULT, TOKYO = keep
    print('chase_decision_delay selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    dict(selftest=selftest, tokyo=stage_tokyo, read=stage_read, exits=lambda: stage_exits(a_[0]))[cmd]()
