"""The frozen chase forward candidate under Tokyo's measured taker execution, with the acceptance delays of BUYs sent into large Binance
moves (USER 2026-11-28: "Check the Tokyo live commit"; the request said research would re-run the frozen line with this distribution,
weighted to the line's own move sizes). P-2026-005, public data, no orders. Declared before the run. A re-pricing of the frozen candidate
on the forward days it was tested on (11-20 .. 11-26, read by this line on 11-27), not a new test: nothing is chosen, there is no PASS
rule, and the selections, limits and models are the frozen run's (forward_selections.parquet), on the primary selection's rows only.
The execution, from the two committed checks of Tokyo's receipts (tokyo_taker_check.json 1c1d85553; tokyo_bigmove_check.json):
  - entry: Predict accepts the BUY A ms after sending, one of 30 measured after-move delays: D0..D19 the big-move probe's (BUYs sent on
    a Binance move of at least $3.5 in 100 ms, in pair order), D20..D29 the first probe's (BUYs sent at a move's first tick); the match
    is HOLD_LIVE = 151 ms after acceptance: Le = A - E.NET + (HOLD_LIVE - E.HOLD), added to the entry match as chase_forward's L is;
  - exit: TAKE_C1000 takes at the first 1 s Binance calm after the settlement notice, S ms after the match, one of the 35 measured (26 +
    9); the exit IOC is accepted at the two probes' pooled SELL median: Lx = that - E.NET + (HOLD_LIVE - E.HOLD) in EF.take_exit_L.
Configs: L0 / L10 / L25 (chase_forward.exits_day's own; the parity configs), D{k}_S{j} (entry delay k, exit floor S_j; TAKE_C1000).
Per action, after the day's parity check, from the day's Binance mid (chase_move_size.move_stats, the line's own definitions, the mid
collapsed per E ms): mv_up100, the move in the action's direction over the 100 ms before; mv_n_post200, the mid changes in the 200 ms after.
Readings (declared), on the primary cell (MOVE_top10 + 3c, TAKE_C1000), EARLY scope as the forward result; each action's P&L is its mean
over a set of delays x the 35 S:
  R1 BIG        - the 20 big-move delays, every action alike (the request's measurement, unweighted);
  R2 BIG_MOVE   - matched on the move before (the request's declared weighting): two strata at the line's median mv_up100, or at the
                  probe set's own median if a stratum would hold fewer than MIN_STRATUM probe BUYs (then not computable if it still
                  would); an action takes its stratum's delays;
  R3 BIG_BURST  - matched on the burst after (mv_n_post200), the same rule: acceptance tracked it in both probes (Spearman 0.52; 0.54);
  R4 BOTH       - all 30 delays, every action alike;
  R5 BOTH_BURST - all 30, matched on the burst after, the same rule.
  An action whose covariate is NaN takes the whole set (counted). Per reading: total, days > 0 of 7, day-equivalent Sharpe, max drawdown,
  expected filled; by delay (each of the 30 over the 35 S); DRAWS = 200 seeded draws for R1 and R5 (each action one delay of its set and
  one S): the spread from execution luck alone.
Limits declared: two probes, 9 and 11 minutes on one Saturday; the big-move probe's moves were $3.5-6.5 (the line's median is $7.0), and
its sends were followed by less Binance activity than the line's entries; matching assumes acceptance depends on the burst after the send
as the probes show (n 20 and 30), not on anything else that differs between the probes and the line; Predict's book keeps the replay's
53.1 ms feed; the exit IOC keeps the replay's rule.
Parity: the L0 / L10 / L25 columns and the identity columns equal the forward run's stored exits exactly on the primary selection's rows
(the day stage refuses to write otherwise), and the read's L cells of the primary selection equal chase_forward_result.json's.
Stages: selftest | exits <day> | read.  Outputs (write-once): OUT/chase_forward/exits_bigmove/<day>.parquet + <day>.json,
chase_forward_bigmove.json (CFB_OUT=<dir>: a rehearsal writes there; its read skips the cell parity with fewer than seven days).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_forward_bigmove.py <stage> ...
"""
import datetime as dt
import json
import os
import re
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_forward as CF  # noqa: E402
import chase_forward_latency as CFL  # noqa: E402
import chase_forward_tokyo as CFT  # noqa: E402
import chase_move_size as CMS  # noqa: E402

E, EF, M2, C, TR, CT = CF.E, CF.EF, CF.M2, CF.C, CF.TR, CF.CT
FIRST, BIG = H / 'tokyo_taker_check.json', H / 'tokyo_bigmove_check.json'
REHEARSAL = bool(os.environ.get('CFB_OUT'))
OUT_B = Path(os.environ['CFB_OUT']) if REHEARSAL else CF.FOUT / 'exits_bigmove'
RESULT = OUT_B / 'read.json' if REHEARSAL else H / 'chase_forward_bigmove.json'
SEL = 'MOVE_top10'
DRAWS, SEED, MIN_STRATUM = 200, 20261129, 5
MV = ('mv_up100', 'mv_n_post200')
CFT.NEWCOL = re.compile(r'D\d+_')     # the measured configs' columns here (chase_forward_tokyo.parity3 reads it at call time)


# ------------------------------------------------------------------------------------------------------------------ execution
def measured(first=None, big=None):
    """The two committed checks -> dict(A, cov, S, hold_live, sell_accept, Le, Lx): A[k] with its probe and covariates, k = 0..29."""
    F = first if first is not None else json.loads(FIRST.read_text())
    B = big if big is not None else json.loads(BIG.read_text())
    for nm, R in (('first', F), ('big-move', B)):
        C.require(R['parity']['mismatched'] == 0 and R['parity']['values_compared'] > 0, f'the {nm} check did not reproduce Tokyo\'s report')
    C.require(B['trigger']['trigger_recomputed_equal'] and B['trigger']['tokyo_counts_reproduced'], 'the big-move check did not reproduce the triggers')
    big_rows = sorted(B['buys'], key=lambda r: r['n'])
    first_rows = [r for r in B['pooled_with_first_probe']['rows'] if r['probe'] == 'first']
    fa = sorted(float(o['accept_wallet_after_send_ms']) for o in F['orders'] if o['tag'] == 'run1' and o['side'] == 'BUY' and o['kind'] == 'Move')
    C.require(sorted(float(r['accept_ms']) for r in first_rows) == fa and len(fa) == 10, "the first probe's delays and their move statistics differ")
    C.require(len(big_rows) == 20 and len({r['n'] for r in big_rows}) == 20, 'expected the 20 big-move BUYs')
    rows = [dict(probe='big', n=r['n'], A=float(r['accept_ms']), up100=r['up100'], n_post200=r['n_post200']) for r in big_rows] + \
           [dict(probe='first', n=None, A=float(r['accept_ms']), up100=r['up100'], n_post200=r['n_post200']) for r in first_rows]
    S = [float(o['match_to_wallet_success_ms']) for R in (F, B) for o in R['orders'] if o.get('match_to_wallet_success_ms') is not None]
    C.require(len(S) == F['settlement']['match_to_success_notice_all_fills']['n'] + B['hold']['fills'], 'settlement counts')
    run1 = [o for R in (F, B) for o in R['orders'] if o['tag'] == 'run1']
    hold = float(np.median([o['accept_wallet_to_match_ms'] for o in run1 if o.get('accept_wallet_to_match_ms') is not None]))
    xa = float(np.median([o['accept_wallet_after_send_ms'] for o in run1 if o['side'] == 'SELL' and o.get('accept_wallet_after_send_ms') is not None]))
    C.require(hold == 151. and all(r['A'] > 0 for r in rows) and min(S) > M2.LEARN, f'implausible measured values (hold {hold})')
    dh = hold - E.HOLD
    return dict(rows=rows, A=[r['A'] for r in rows], S=S, hold_live=hold, sell_accept=xa, Le=[r['A'] - E.NET + dh for r in rows], Lx=xa - E.NET + dh)


def configs(ms):
    ent = [dict(name=f'L{int(L)}', Le=float(L)) for L in CFT.PAR_LATS] + [dict(name=f'D{k}', Le=le) for k, le in enumerate(ms['Le'])]
    ext = [dict(name=f'L{int(L)}', entry=f'L{int(L)}', Sa=M2.LEARN, Sb=float(L), Lx=float(L), pols=CF.POL) for L in CFT.PAR_LATS]
    ext += [dict(name=f'D{k}_S{j}', entry=f'D{k}', Sa=s, Sb=0., Lx=ms['Lx'], pols=('TAKE_C1000',)) for k in range(len(ms['Le'])) for j, s in enumerate(ms['S'])]
    names = [x['name'] for x in ext]
    C.require(len(set(names)) == len(names) and {x['entry'] for x in ext} <= {e['name'] for e in ent}, 'config names')
    return ent, ext


# ------------------------------------------------------------------------------------------------------------------ day
_MID = {}
_mid_series = CT.mid_series


def mid_once(day):
    """chase_takes.mid_series, loaded once per day: the covariates and exits_day_tokyo's Binance read the same arrays (M3.Binance copies)."""
    if day not in _MID:
        _MID.clear()
        _MID[day] = _mid_series(day)
    return _MID[day]


CT.mid_series = mid_once


def stage_exits(day):
    C.require(day in CF.FWD, f'{day} is not a declared forward day')
    fp, fj = OUT_B / f'{day}.parquet', OUT_B / f'{day}.json'
    C.require(not fp.exists() and not fj.exists(), f'{fp.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    CF.frozen()
    ms = measured()
    ent, ext = configs(ms)
    CF.forward_mode((day,))
    S = pd.read_parquet(CF.FOUT / 'forward_selections.parquet')
    S = S[S[SEL].to_numpy(bool)]
    Sd = S[S.day == day].reset_index(drop=True)
    Eb, Mb = CT.mid_series(day)
    mv = CMS.move_stats(Eb, Mb, Sd.t.to_numpy(float), Sd.d.to_numpy(float))
    X, st = CFT.exits_day_tokyo(day, S, ent, ext)
    C.require(np.array_equal(X.t.to_numpy(float), Sd.t.to_numpy(float)) and np.array_equal(X.d.to_numpy(float), Sd.d.to_numpy(float)), 'row order')
    old = pd.read_parquet(CF.FOUT / 'exits' / f'{day}.parquet')
    old = old[old[SEL].to_numpy(bool)].reset_index(drop=True)
    par = CFT.parity3(X, old)
    del old
    C.require(par['columns'] > 0 and not par['mismatched'], f'L0 / L10 / L25 parity failed on {day}: {par["mismatched"][:8]}')
    X['mv_up100'], X['mv_n_post200'] = mv['up100'], mv['n_post200']
    st.update(parity=par, grid_drops=dict(CF.GRID_DROPS), measured=ms, rehearsal=REHEARSAL, rows=SEL)
    OUT_B.mkdir(parents=True, exist_ok=True)
    X.to_parquet(fp, index=False)
    TR.dump(fj, st)
    print(day, json.dumps(TR.clean(dict(st, measured=None))), flush=True)


# ------------------------------------------------------------------------------------------------------------------ read
def split_for(x_line, x_probe, n_min=MIN_STRATUM):
    """The strata split: the line's median, or the probe set's median if a stratum would hold fewer than n_min probe BUYs; None if
    that fails too. -> (split, source)."""
    xp = np.asarray(x_probe, float)
    for src, v in (('line_median', float(np.nanmedian(x_line))), ('probe_median', float(np.median(xp)))):
        hi = int(np.sum(xp >= v))
        if min(hi, len(xp) - hi) >= n_min:
            return v, src
    return None, 'not_computable'


def set_mean(P, F, sets):
    """Per action i, the mean of P[k, i] (NaN where every k is NaN) and of F[k, i] over its delay set sets[i] (a boolean row over k)."""
    w = sets.astype(float).T                               # k x actions
    Pn = np.where(np.isnan(P), 0., P)
    cnt = np.sum(w * ~np.isnan(P), axis=0)
    pn = np.where(cnt > 0, np.sum(w * Pn, axis=0) / np.maximum(cnt, 1), np.nan)
    ff = np.sum(w * F, axis=0) / np.maximum(np.sum(w, axis=0), 1)
    return pn, ff


def strata_sets(x, x_probe, ks, nk, x_line):
    """Boolean sets (actions x nk): the delays ks (indices into the nk) split into two strata at split_for; an action takes its stratum's,
    or every k of ks if its covariate is NaN. -> (sets, info)."""
    ks = np.asarray(ks)
    v, src = split_for(x_line, x_probe)
    sets = np.zeros((len(x), nk), bool)
    if v is None:
        return None, dict(split=None, source=src)
    hi_k = ks[np.asarray(x_probe, float) >= v]
    lo_k = ks[np.asarray(x_probe, float) < v]
    nan = np.isnan(x)
    sets[np.ix_(~nan & (x >= v), hi_k)] = True
    sets[np.ix_(~nan & (x < v), lo_k)] = True
    sets[np.ix_(nan, ks)] = True
    return sets, dict(split=v, source=src, probe_high=int(len(hi_k)), probe_low=int(len(lo_k)), line_high_share=float(np.mean(x[~nan] >= v)),
                      nan_actions=int(nan.sum()))


def stage_read():
    C.require(not RESULT.exists(), f'{RESULT.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    rc = CF.frozen()
    p, sel_p, s_p = CFL.primary(rc)
    C.require(sel_p == SEL, f'the receipt\'s primary selection is {sel_p}, the rows here are {SEL}')
    ms = measured()
    days = tuple(d for d in CF.FWD if (OUT_B / f'{d}.parquet').exists())
    C.require(days == CF.FWD or (REHEARSAL and days), f'exits_bigmove has {len(days)} of {len(CF.FWD)} days')
    X = pd.concat([pd.read_parquet(OUT_B / f'{d}.parquet') for d in days], ignore_index=True)
    keys, dayk = [], []
    for d in days:
        reg = CF.MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]; dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    CF.SX.DAYS = days
    per_day = len(keys) / len(days)
    cells0 = CF.cells_of(X, keys, dayk, per_day, days, EF.EARLY_MS, CFT.PAR_LATS)
    R0 = json.loads((CF.H / 'chase_forward_result.json').read_text())
    par = {}
    for L in CFT.PAR_LATS:
        tag = f'{SEL}|L{int(L)}_'
        a = json.loads(json.dumps(TR.clean({k: v for k, v in cells0.items() if k.startswith(tag)}), allow_nan=False))
        b = {k: v for k, v in R0['cells'].items() if k.startswith(tag)}
        par[f'L{int(L)}'] = (len(a), sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b) if a[k] != b[k]))
    if days == CF.FWD:
        C.require(all(n > 0 and not bd for n, bd in par.values()), f'parity cells differ from chase_forward_result.json: {par}')
    early = X.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS
    m = X[SEL].to_numpy(bool) & early
    wk = (X.day.astype(str) + ':' + X.mid.astype(np.int64).astype(str)).to_numpy()
    nk, nS = len(ms['A']), len(ms['S'])
    s = int(s_p)
    PS = np.stack([np.stack([X[f'D{k}_S{j}_AB_s{s}_TAKE_C1000_pnl'].to_numpy(float) for j in range(nS)]) for k in range(nk)])   # k x S x rows
    P = PS.mean(axis=1)                                     # over S, as chase_forward_tokyo.mix (NaN only where the action is unloaded)
    F = np.stack([(np.nan_to_num(X[f'D{k}_AB_s{s}_fill'].to_numpy(float)) > 0).astype(float) for k in range(nk)])
    up, npost = X.mv_up100.to_numpy(float), X.mv_n_post200.to_numpy(float)
    line_mv = dict(up100=CMS.quant(up[m]), n_post200=CMS.quant(npost[m]))
    cms = json.loads(CMS.OUT.read_text())['line'][SEL]
    if days == CF.FWD:
        C.require(all(abs(line_mv[k]['p50'] - cms[k]['p50']) < 1e-9 and line_mv[k]['n'] == cms[k]['n'] for k in line_mv),
                  f"the actions' move statistics differ from chase_move_size.json: {line_mv} vs {cms['up100']['p50']}, {cms['n_post200']['p50']}")
    big = [k for k, r in enumerate(ms['rows']) if r['probe'] == 'big']
    allk = list(range(nk))
    cov = {k_: np.array([r[k_] for r in ms['rows']], float) for k_ in ('up100', 'n_post200')}
    every = lambda ks: np.tile(np.isin(np.arange(nk), ks), (len(X), 1))  # noqa: E731
    sets, info = {}, {}
    sets['R1_BIG'], info['R1_BIG'] = every(big), dict(delays=len(big))
    sets['R2_BIG_MOVE'], info['R2_BIG_MOVE'] = strata_sets(up, cov['up100'][big], big, nk, up[m])
    sets['R3_BIG_BURST'], info['R3_BIG_BURST'] = strata_sets(npost, cov['n_post200'][big], big, nk, npost[m])
    sets['R4_BOTH'], info['R4_BOTH'] = every(allk), dict(delays=nk)
    sets['R5_BOTH_BURST'], info['R5_BOTH_BURST'] = strata_sets(npost, cov['n_post200'], allk, nk, npost[m])
    readings = {}
    for nm, st_ in sets.items():
        if st_ is None:
            readings[nm] = dict(computable=False, **info[nm])
            continue
        pn, ff = set_mean(P, F, st_)
        readings[nm] = dict(computable=True, **info[nm], **CFT.stats_of(pn, m, ff, wk, keys, dayk, per_day),
                            mean_accept_ms=float(np.mean((st_[m] * np.array(ms['A'])).sum(1) / st_[m].sum(1))))
        readings[nm]['total_positive'] = bool(readings[nm]['total_usd'] > 0)
    by_delay = [dict(k=k, probe=ms['rows'][k]['probe'], n=ms['rows'][k]['n'], accept_ms=ms['A'][k], up100=ms['rows'][k]['up100'],
                     n_post200=ms['rows'][k]['n_post200'], **{x: CFT.stats_of(P[k], m, F[k], wk, keys, dayk, per_day)[x]
                                                              for x in ('total_usd', 'days_positive', 'sharpe_day_equiv', 'filled')}) for k in range(nk)]
    rng = np.random.default_rng(SEED)
    idx = np.flatnonzero(m)
    draws = {}
    for nm in ('R1_BIG', 'R5_BOTH_BURST'):
        if sets[nm] is None:
            continue
        out, cs = [], sets[nm][idx]
        cnt, cum = cs.sum(1), np.cumsum(cs, axis=1)
        for _ in range(DRAWS):
            pos = np.argmax(cum > np.floor(rng.random(len(idx)) * cnt)[:, None], axis=1)   # one delay of each action's set, uniform
            pn = np.full(len(X), np.nan)
            pn[idx] = PS[pos, rng.integers(0, nS, len(idx)), idx]
            st_ = CFT.stats_of(pn, m, np.zeros(len(X)), wk, keys, dayk, per_day)
            out.append((st_['total_usd'], st_['days_positive'], np.nan if st_['sharpe_day_equiv'] is None else st_['sharpe_day_equiv']))
        dr = np.array(out, float)
        q = lambda v: dict(p5=float(np.nanquantile(v, .05)), median=float(np.nanmedian(v)), p95=float(np.nanquantile(v, .95)), nan=int(np.isnan(v).sum()))  # noqa: E731
        draws[nm] = dict(n=DRAWS, seed=SEED, total_usd=q(dr[:, 0]), days_positive=q(dr[:, 1]), sharpe_day_equiv=q(dr[:, 2]), share_total_positive=float(np.mean(dr[:, 0] > 0)))
    replay = R0['cells'][f'{sel_p}|L0_AB_s{s}|TAKE_C1000']
    R1 = json.loads(CFT.RESULT.read_text())
    prev = dict(both=R1['decomposition']['both'], primary_by_accept=R1['primary_by_accept'])
    stt = {d: json.loads((OUT_B / f'{d}.json').read_text()) for d in days}
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), receipt=CF.receipt_path().name, primary=p, days=list(days), rehearsal=REHEARSAL,
               measured={k: ms[k] for k in ('rows', 'S', 'hold_live', 'sell_accept', 'Le', 'Lx')}, line_moves=line_mv, readings=readings, by_delay=by_delay,
               draws=draws, replay_L0=replay, first_probe_result=prev, parity_cells={k: dict(cells=n, mismatched=b) for k, (n, b) in par.items()},
               parity_checked=days == CF.FWD, statuses={d: dict(actions=v['actions'], unloaded=v['unloaded'], capped_loops=v['capped_loops'],
                                                                grid_drops=v['grid_drops'], parity=v['parity']) for d, v in stt.items()})
    TR.dump(RESULT, out)
    print('primary', p, json.dumps(TR.clean({k: {x: v.get(x) for x in ('total_usd', 'days_positive', 'sharpe_day_equiv', 'filled', 'split', 'source')}
                                             for k, v in readings.items()})), 'parity', {k: (n, len(b)) for k, (n, b) in par.items()}, flush=True)


# ------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    F, B = json.loads(FIRST.read_text()), json.loads(BIG.read_text())
    ms = measured(F, B)
    C.require(len(ms['A']) == 30 and len(ms['S']) == 35 and ms['hold_live'] == 151., f"measured {len(ms['A'])} {len(ms['S'])} {ms['hold_live']}")
    C.require(all(abs(le - (a - 21.)) < 1e-12 for a, le in zip(ms['A'], ms['Le'])) and abs(ms['Lx'] - (ms['sell_accept'] - 21.)) < 1e-12, 'Le, Lx')
    C.require([r['probe'] for r in ms['rows']] == ['big'] * 20 + ['first'] * 10 and [r['n'] for r in ms['rows'][:20]] == list(range(20)), 'delay order')
    for bad_, why in ((lambda b: b['parity'].update(mismatched=1), 'did not reproduce Tokyo'), (lambda b: b['trigger'].update(trigger_recomputed_equal=False), 'triggers'),
                      (lambda b: b['pooled_with_first_probe']['rows'].pop(0), 'differ'), (lambda b: b['buys'].pop(), 'expected the 20')):
        b2 = json.loads(json.dumps(B)); bad_(b2)
        try:
            measured(F, b2)
            raise AssertionError(f'measured accepted a bad check ({why})')
        except ValueError as e_:
            assert why in str(e_), e_
    ent, ext = configs(ms)
    C.require(len(ent) == 3 + 30 and len(ext) == 3 + 30 * 35 and all(x['pols'] == ('TAKE_C1000',) for x in ext[3:]), f'config counts {len(ent)} {len(ext)}')
    C.require(CFT.NEWCOL.match('D12_S3_AB_s3_TAKE_C1000_pnl') and not CFT.NEWCOL.match('mv_up100') and not CFT.NEWCOL.match('A0_AB_s3_fill'), 'NEWCOL')
    # the strata rule: the line's median when both strata hold MIN_STRATUM probe BUYs, else the probe's, else none
    C.require(split_for([1, 2, 3, 6, 7], [0, 0, 1, 2, 2.5, 3, 6, 7, 8, 9]) == (3., 'line_median'), 'split at the line median')
    C.require(split_for([1, 2, 3, 6, 7], [0, 0, 1, 2, 3, 6, 7, 8, 9, 10]) == (4.5, 'probe_median'), 'four probe BUYs under the line median: the probe median')
    C.require(split_for([7, 8, 9], [3.5, 3.6, 3.6, 4, 4, 4.5, 4.5, 5, 6, 6.5]) == (4.25, 'probe_median'), 'no probe BUY above the line median: the probe median')
    C.require(split_for([7], [1, 1, 1, 1, 1, 1, 9]) == (None, 'not_computable'), 'ties: neither split holds 5 a side')
    x = np.array([0., 10., np.nan])
    sets, inf = strata_sets(x, np.array([0., 0., 0., 0., 0., 20., 20., 20., 20., 20.]), list(range(10)), 12, np.array([0., 6., 10.]))
    C.require(inf['split'] == 6. and sets[0, :5].all() and not sets[0, 5:].any() and sets[1, 5:10].all() and not sets[1, :5].any()
              and sets[2, :10].all() and not sets[:, 10:].any() and inf['nan_actions'] == 1, f'strata sets {inf}')
    P = np.array([[1., np.nan, np.nan], [3., 4., np.nan]])
    Fm = np.array([[1., 0., 0.], [0., 1., 0.]])
    pn, ff = set_mean(P, Fm, np.array([[True, True], [True, False], [True, True]]))
    C.require(pn[0] == 2. and np.isnan(pn[1]) and np.isnan(pn[2]) and ff[0] == .5 and ff[1] == 0. and ff[2] == .0, f'set_mean {pn} {ff}')
    pn, _ = set_mean(P, Fm, np.array([[False, True], [False, True], [True, False]]))
    C.require(pn[0] == 3. and pn[1] == 4. and np.isnan(pn[2]), f'set_mean picks its set {pn}')
    try:
        stage_exits('20261101')
        raise AssertionError('stage_exits accepted a consumed day')
    except ValueError as e_:
        assert 'not a declared forward day' in str(e_), e_
    global RESULT
    keep = RESULT
    with tempfile.NamedTemporaryFile(suffix='.json') as tf:
        RESULT = Path(tf.name)
        try:
            stage_read()
            raise AssertionError('stage_read overwrote its result')
        except ValueError as e_:
            assert 'write-once' in str(e_), e_
        finally:
            RESULT = keep
    print('chase_forward_bigmove selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    dict(selftest=selftest, read=stage_read, exits=lambda: stage_exits(a_[0]))[cmd]()
