"""The frozen chase forward candidate under Tokyo's measured taker execution (USER 2026-11-28: "Tokyo answers, check the commit";
"You mean you will do the replay right?"). P-2026-005, public data, no orders. Declared before the run. A re-pricing of the frozen
candidate on the forward days it was tested on (11-20 .. 11-26, read by this line on 11-27), not a new test: nothing is chosen, there
is no PASS rule, and the selections, limits and models are the frozen run's (forward_selections.parquet).
The execution, from tokyo_taker_check.json (ctaNew 1c1d85553: Tokyo's receipt pmmmrust f3b3db4 recomputed from its raw records):
  - entry: Predict accepted a BUY sent right after a Binance mid change A ms after sending (10 measured values, median 85.9; the replay
    uses E.NET = 22), and matched it HOLD_LIVE = 151 ms after acceptance (the replay 150). An entry at A matches A - 22 + 1 ms later
    than the replay's: Le = A - E.NET + (HOLD_LIVE - E.HOLD), added to the entry match exactly as chase_forward's L is.
  - exit: bought shares can be sold only after settlement. Every exit is decided no earlier than the settlement notice, S ms after the
    match (26 measured values, median 1,858.6; the replay decides from the fill notice, M2.LEARN = 22 ms after the match).
    TAKE_C1000 takes at the first 1 s Binance calm from then; TB_SETTLE takes at the notice itself.
  - the exit IOC: accepted a median 24.7 ms after sending (quiet SELLs) and matched 151 ms after: Lx = 24.7 - 22 + 1 in EF.take_exit_L.
    The maker exit (EF.FIXED) starts at the settlement-floored calm with Lx on its messages.
Configs (per action; the fill depends on the entry config only):
  L0 / L10 / L25 - chase_forward.exits_day's own (Le = Lx = L, exit floor LEARN + L, every POL policy): the parity configs.
  A{i}_N  - entry at A_i, the replay's exit (floor LEARN, Lx 0): TAKE_C1000, HOLD           (the entry's share of the change)
  AR_S{j} - entry at 22 ms (the replay's, live hold), exit floor S_j, Lx: TAKE_C1000, TB_SETTLE (the exit's share)
  A{i}_S{j} - both measured: TAKE_C1000, TB_SETTLE                                         (the measured execution)
  A{i}_SM - both, at the median S: the maker exit EF.FIXED
Readings (declared; computed in read), per selection x limit, EARLY scope as the forward result:
  - the expected P&L under the measured execution: every action's P&L averaged over the 10 x 26 (A_i, S_j) configs, as one series
    (MIX_BOTH); the primary cell (RECEIPT_v5's, TAKE_C1000): total > 0, days > 0 of 7, day-equivalent Sharpe, max drawdown, filled;
  - its decomposition: the stored replay (L0) | entry only (MIX_ENTRY, mean over A_i_N) | exit only (MIX_EXIT, mean over AR_S_j) | both;
  - the primary by A_i (mean over S_j); TB_SETTLE; the maker exit at the median S (MIX_MAKER); the control (random move onsets);
  - DRAWS = 200 seeded draws giving each action one (A_i, S_j) at random: the spread of total, days > 0 and Sharpe from execution
    luck alone (actions' delays independent of each other and of the move: a limit, the probe saw slower acceptance after larger bursts).
Limits declared: one 11-minute probe on a Saturday; the probe's move trigger was any Binance mid change (the line's top-10% moves may
load the venue more); Predict's book reached Tokyo 67.8 ms after its stamp in the probe against the replay's 53.1 (unchanged here: the
selections and limits are the frozen run's); the exit IOC keeps the replay's rule (any price on the snapshot before the match).
Parity: the L0 / L10 / L25 columns and the identity columns equal the forward run's stored exits exactly on every day (the day stage
refuses to write otherwise), and the read's L0 / L10 / L25 cells equal chase_forward_result.json's.
Stages: selftest | exits <day> | read.  Outputs (write-once): OUT/chase_forward/exits_tokyo/<day>.parquet + <day>.json,
chase_forward_tokyo.json (CFT_OUT=<dir>: a rehearsal writes there; its read skips the cell parity with fewer than seven days).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_forward_tokyo.py <stage> ...
"""
import collections
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

E, EF, EM, M2, M3, SX = CF.E, CF.EF, CF.EM, CF.M2, CF.M3, CF.SX
C, TR, TM, CT, MM, TK, VR, F = CF.C, CF.TR, CF.TM, CF.CT, CF.MM, CF.TK, CF.VR, CF.F
CHECK = H / 'tokyo_taker_check.json'
PAR_LATS = CF.LATS
REHEARSAL = bool(os.environ.get('CFT_OUT'))
OUT_T = Path(os.environ['CFT_OUT']) if REHEARSAL else CF.FOUT / 'exits_tokyo'
RESULT = OUT_T / 'read.json' if REHEARSAL else H / 'chase_forward_tokyo.json'
DRAWS, SEED = 200, 20261128
CMP = ('HOLD', 'TB1S', 'FAV_ONLY')
PARCOL = re.compile(r'L(0|10|25)_')
NEWCOL = re.compile(r'(A\d+_|AR_)')


# ------------------------------------------------------------------------------------------------------------------ execution
def measured(chk=None):
    """The measured execution from the committed check -> dict(A, S, Le, Lx, hold_live, sources)."""
    R = chk if chk is not None else json.loads(CHECK.read_text())
    C.require(R['parity']['mismatched'] == 0 and R['parity']['values_compared'] > 0, 'the Tokyo check did not reproduce Tokyo\'s report')
    A = sorted(float(o['accept_wallet_after_send_ms']) for o in R['orders'] if o['tag'] == 'run1' and o['side'] == 'BUY' and o['kind'] == 'Move')
    S = sorted(float(o['match_to_wallet_success_ms']) for o in R['orders'] if o.get('match_to_wallet_success_ms') is not None)
    P = R['pooled']
    hold = float(P['run_all']['accept_wallet_to_match_ms']['median'])
    xa = float(P['SELL']['accept_wallet_after_send_ms']['median'])
    C.require(len(A) == P['BUY_after_move']['orders'] and len(S) == R['settlement']['match_to_success_notice_all_fills']['n'], 'measured counts')
    C.require(abs(np.median(A) - P['BUY_after_move']['accept_wallet_after_send_ms']['median']) < 1e-9, 'after-move acceptance median')
    C.require(abs(np.median(S) - R['settlement']['match_to_success_notice_all_fills']['median']) < 1e-9, 'settlement median')
    C.require(all(a > 0 for a in A) and min(S) > M2.LEARN and hold >= E.HOLD, 'implausible measured values')
    dh = hold - E.HOLD
    return dict(A=A, S=S, hold_live=hold, sell_accept=xa, Le=[a - E.NET + dh for a in A], Lx=xa - E.NET + dh, Le_replay=dh,
                S_median=float(np.median(S)))


def configs(ms):
    """Entry configs (name, Le) and exit configs (name, entry, Sa, Sb, Lx, pols): the exit decision floor is the entry match + Sa + Sb."""
    ent = [dict(name=f'L{int(L)}', Le=float(L)) for L in PAR_LATS]
    ext = [dict(name=f'L{int(L)}', entry=f'L{int(L)}', Sa=M2.LEARN, Sb=float(L), Lx=float(L), pols=CF.POL) for L in PAR_LATS]
    ent += [dict(name=f'A{i}', Le=le) for i, le in enumerate(ms['Le'])] + [dict(name='AR', Le=ms['Le_replay'])]
    for i in range(len(ms['A'])):
        ext.append(dict(name=f'A{i}_N', entry=f'A{i}', Sa=M2.LEARN, Sb=0., Lx=0., pols=('TAKE_C1000', 'HOLD')))
        ext += [dict(name=f'A{i}_S{j}', entry=f'A{i}', Sa=s, Sb=0., Lx=ms['Lx'], pols=('TAKE_C1000', 'TB_SETTLE')) for j, s in enumerate(ms['S'])]
        ext.append(dict(name=f'A{i}_SM', entry=f'A{i}', Sa=ms['S_median'], Sb=0., Lx=ms['Lx'], pols=(EF.FIXED,)))
    ext += [dict(name=f'AR_S{j}', entry='AR', Sa=s, Sb=0., Lx=ms['Lx'], pols=('TAKE_C1000', 'TB_SETTLE')) for j, s in enumerate(ms['S'])]
    names = [x['name'] for x in ext]
    C.require(len(set(names)) == len(names) and {x['entry'] for x in ext} <= {e['name'] for e in ent}, 'config names')
    return ent, ext


def floor_start(t, Le, Sa, Sb):
    """The absolute decision floor of an exit, in chase_forward.exits_day's own order of additions (bit-equal in the parity configs)."""
    return t + TM.FEED + TM.DECIDE + E.NET + Le + E.HOLD + Sa + Sb


def settle_x(act, Sa, Sb, cut):
    """TB_SETTLE: take when the settlement notice arrives (path clock), unless that is past the cut (then held, as TAKE_C1000's calm)."""
    x = act + Sa + Sb
    return x if x <= cut else None


# ------------------------------------------------------------------------------------------------------------------ day
def exits_day_tokyo(day, S, ent, ext):
    """chase_forward.exits_day, its loop copied line for line, with per-config entry delays, exit floors and exit latencies (see the
    docstring); the L0 / L10 / L25 configs are its own and must reproduce its stored columns."""
    S = S[S.day == day].reset_index(drop=True)
    BN = M3.Binance(*CT.mid_series(day))
    LG = EF.legs_who(day)
    LGr, lst = EM.legs_of(day)
    C.require(set(LG) == set(LGr) and all(all(np.array_equal(LG[m][k], LGr[m][k]) for k in range(4)) for m in LG), 'legs_who differs from EM.legs_of')
    del LGr
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    by_mid = S.groupby('mid').indices
    o = collections.defaultdict(lambda: np.full(len(S), np.nan))
    n_cap = collections.Counter()
    tt = S.t.to_numpy(float)
    Le = {e['name']: e['Le'] for e in ent}
    CALM = {x['name']: M2.calm_times(BN.kn, floor_start(tt, Le[x['entry']], x['Sa'], x['Sb']), EF.GAP) for x in ext}
    by_ent = {e['name']: [x for x in ext if x['entry'] == e['name']] for e in ent}
    src = MM.KP.Tape(); off = src.offset(day)
    loader = collections.Counter()
    for h, gr in reg.groupby('hour'):
        mids = [int(m) for m in gr.index if int(m) in by_mid]
        if not mids:
            continue
        for m, W, lst_ in VR.worlds_all(day, mids, src, off, reg):
            if lst_ != 'loaded':
                loader[lst_] += len(by_mid[m])
                for i in by_mid[m]:
                    o['status_unloaded'][i] = 1.
                continue
            p_, st, info = VR.vm_path(W, float(reg.loc[m, 'ws']) * 1000., 50.)
            if p_ is None:
                loader[f'path_{st}'] += len(by_mid[m])
                for i in by_mid[m]:
                    o['status_unloaded'][i] = 1.
                continue
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p_['frames'], float)
            f0, fk = f[:, 0], f[:, 1]
            okb = (f[:, 2] >= 0) & (f[:, 12] <= 100) & (f[:, 2] < f[:, 12])
            midy = np.where(okb, (f[:, 2] + f[:, 12]) / 2., np.nan)
            cut = float(rg.loc[m, 'we']) * 1000. - t0 - E.LAST_MS
            lg = LG.get(int(m))
            for i in by_mid[m]:
                t, d, lim, pay = float(S.t.iat[i]), float(S.d.iat[i]), float(S.lim.iat[i]), float(S.payout.iat[i])
                act0 = t + TM.FEED + TM.DECIDE - t0 + E.NET + E.HOLD
                o['cut_minus_act_ms'][i] = cut - act0
                side = 1 if d > 0 else 0
                for en in ent:
                    act = act0 + en['Le']
                    jv = int(np.searchsorted(f0, act, 'right') - 1)
                    pa, sa = TK.side_levels(f[jv], side)
                    pb, sb_ = TK.side_levels(f[jv + 1], side) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
                    bk = TK.minbook(pa[None], sa[None], pb[None], sb_[None])
                    k0 = int(np.searchsorted(fk, act, 'right'))
                    kt, sm = fk[k0:], (midy[k0:] if d > 0 else 100. - midy[k0:])
                    for s in (3., 10.):
                        pe = f"{en['name']}_AB_s{int(s)}"
                        fl, co, fe_ = (x[0] for x in TK.vwalk(*bk, np.array([d]), np.array([lim + d * s])))
                        o[f'{pe}_fill'][i] = fl
                        o[f'{pe}_entry_fee_c'][i] = fe_
                        if fl <= 0:
                            for x in by_ent[en['name']]:
                                for pol in x['pols']:
                                    o[f"{x['name']}_AB_s{int(s)}_{pol}_pnl"][i] = 0.
                                    o[f"{x['name']}_AB_s{int(s)}_{pol}_exit_fee_c"][i] = 0.
                            continue
                        e = co / fl if d > 0 else 100. - co / fl
                        for x in by_ent[en['name']]:
                            pre, L = f"{x['name']}_AB_s{int(s)}", x['Lx']

                            def put(pol, mq, mc, tq, tc, tf):
                                o[f'{pre}_{pol}_pnl'][i] = d * (mc + tc + (fl - mq - tq) * 100. * pay - co) - fe_ - tf
                                o[f'{pre}_{pol}_exit_fee_c'][i] = tf
                            calm = float(CALM[x['name']][i]) - t0
                            for pol in x['pols']:
                                if pol in CMP:
                                    xx = E.decide(pol, E.trigger(kt, sm, e, cut), E.mid_at(kt, sm, act + E.T_MED), E.mid_at(kt, sm, act + E.T_UQ), act, cut)
                                    put(pol, 0., 0., *EF.take_exit_L(f, f0, end, d, fl, xx, L))
                                elif pol == 'TAKE_C1000':
                                    put(pol, 0., 0., *EF.take_exit_L(f, f0, end, d, fl, calm if calm <= cut else None, L))
                                elif pol == 'TB_SETTLE':
                                    put(pol, 0., 0., *EF.take_exit_L(f, f0, end, d, fl, settle_x(act, x['Sa'], x['Sb'], cut), L))
                                elif pol == EF.FIXED:
                                    res = EF.maker_loop_L(EF.MODE, EF.GAP, EF.PRICE, calm, fl, d, f, f0, fk, lg, t0, cut, BN, L, EF.T_DEAD)
                                    n_cap[x['name']] += int(res['capped'])
                                    inc = res['incs']
                                    mq = float(sum(z['q'] for z in inc)); mc = float(sum(z['q'] * z['px'] for z in inc))
                                    xm = M2.tail_x('TAKE', kt, sm, res['D']) if (res['D'] > res['t_first'] and fl - mq > 1e-9) else None
                                    put(EF.FIXED, mq, mc, *EF.take_exit_L(f, f0, end, d, fl - mq, xm, L))
                                    o[f'{pre}_maker_sh'][i] = mq
                                else:
                                    raise ValueError(f'unknown policy {pol}')
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    out = pd.concat([S, pd.DataFrame(dict(o), index=S.index)], axis=1)
    return out, dict(day=day, actions=int(len(S)), legs=lst, capped_loops={k: int(v) for k, v in n_cap.items()}, unloaded=dict(loader))


def parity3(new, old):
    """new's identity columns and L0 / L10 / L25 columns must equal old's exactly (NaN = NaN), same column sets and rows; new's other
    columns are only the measured configs'."""
    idn = sorted(c for c in new.columns if not PARCOL.match(c) and not NEWCOL.match(c))
    ido = sorted(c for c in old.columns if not PARCOL.match(c))
    pn = sorted(c for c in new.columns if PARCOL.match(c)); po = sorted(c for c in old.columns if PARCOL.match(c))
    bad = []
    if idn != ido:
        bad.append('identity columns differ')
    if pn != po:
        bad.append('parity columns differ')
    if len(new) != len(old):
        bad.append('rows differ')
    stray = [c for c in new.columns if not PARCOL.match(c) and NEWCOL.match(c) is None and c not in old.columns]
    if stray:
        bad.append(f'unexpected columns {stray[:3]}')
    cols = [c for c in idn + pn if c in old.columns]
    if not bad:
        bad += [c for c in cols if not CFL.same_col(new[c], old[c])]
    return dict(columns=len(cols), rows=int(len(new)), mismatched=bad)


def stage_exits(day):
    C.require(day in CF.FWD, f'{day} is not a declared forward day')
    fp, fj = OUT_T / f'{day}.parquet', OUT_T / f'{day}.json'
    C.require(not fp.exists() and not fj.exists(), f'{fp.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    CF.frozen()
    ms = measured()
    ent, ext = configs(ms)
    CF.forward_mode((day,))
    S = pd.read_parquet(CF.FOUT / 'forward_selections.parquet')
    X, st = exits_day_tokyo(day, S, ent, ext)
    old = pd.read_parquet(CF.FOUT / 'exits' / f'{day}.parquet')
    par = parity3(X, old)
    del old
    C.require(par['columns'] > 0 and not par['mismatched'], f'L0 / L10 / L25 parity failed on {day}: {par["mismatched"][:8]}')
    st.update(parity=par, grid_drops=dict(CF.GRID_DROPS), measured=ms, rehearsal=REHEARSAL)
    OUT_T.mkdir(parents=True, exist_ok=True)
    X.to_parquet(fp, index=False)
    TR.dump(fj, st)
    print(day, json.dumps(TR.clean(dict(st, measured=None))), flush=True)


# ------------------------------------------------------------------------------------------------------------------ read
def mix(X, names, pol, s):
    """Each action's P&L averaged over the exit configs `names` (one series)."""
    return np.mean(np.stack([X[f'{n}_AB_s{int(s)}_{pol}_pnl'].to_numpy(float) for n in names]), axis=0)


def stats_of(pnl_c, m, fill_frac, wk, keys, dayk, per_day):
    st = SX.stats(SX.window_series(np.nan_to_num(pnl_c[m]) / 100., wk[m], keys), dayk, per_day)
    return dict(total_usd=st['total_usd'], sharpe_day_equiv=st['sharpe_day_equiv'], daily_usd=st['daily_usd'],
                days_positive=int(sum(v > 0 for v in st['daily_usd'])), max_drawdown_usd=st['max_drawdown_usd'],
                actions=int(m.sum()), filled=float(np.sum(fill_frac[m])))


def stage_read():
    C.require(not RESULT.exists(), f'{RESULT.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    rc = CF.frozen()
    p, sel_p, s_p = CFL.primary(rc)
    ms = measured()
    ent, ext = configs(ms)
    days = tuple(d for d in CF.FWD if (OUT_T / f'{d}.parquet').exists())
    C.require(days == CF.FWD or (REHEARSAL and days), f'exits_tokyo has {len(days)} of {len(CF.FWD)} days')
    X = pd.concat([pd.read_parquet(OUT_T / f'{d}.parquet') for d in days], ignore_index=True)
    keys, dayk = [], []
    for d in days:
        reg = MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]; dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    SX.DAYS = days
    per_day = len(keys) / len(days)
    # parity cells: the L0 / L10 / L25 configs through chase_forward.cells_of against the stored result
    cells0 = CF.cells_of(X, keys, dayk, per_day, days, EF.EARLY_MS, PAR_LATS)
    R0 = json.loads((CF.H / 'chase_forward_result.json').read_text())
    C.require(R0['primary'] == p, 'the result and the receipt name different primaries')
    par = {f'L{int(L)}': CFL.cell_parity(cells0, R0['cells'], L) for L in PAR_LATS}
    if days == CF.FWD:
        C.require(all(n > 0 and not b for n, b in par.values()), f'parity cells differ from chase_forward_result.json: {par}')
    early = X.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS
    wk = (X.day.astype(str) + ':' + X.mid.astype(np.int64).astype(str)).to_numpy()
    nA, nS = len(ms['A']), len(ms['S'])
    fills = {e['name']: {s: (np.nan_to_num(X[f"{e['name']}_AB_s{int(s)}_fill"].to_numpy(float)) > 0).astype(float) for s in (3., 10.)} for e in ent}
    fA = {s: np.mean([fills[f'A{i}'][s] for i in range(nA)], axis=0) for s in (3., 10.)}
    series = {}
    for s in (3., 10.):
        series[('MIX_BOTH', 'TAKE_C1000', s)] = (mix(X, [f'A{i}_S{j}' for i in range(nA) for j in range(nS)], 'TAKE_C1000', s), fA[s])
        series[('MIX_BOTH', 'TB_SETTLE', s)] = (mix(X, [f'A{i}_S{j}' for i in range(nA) for j in range(nS)], 'TB_SETTLE', s), fA[s])
        series[('MIX_ENTRY', 'TAKE_C1000', s)] = (mix(X, [f'A{i}_N' for i in range(nA)], 'TAKE_C1000', s), fA[s])
        series[('MIX_ENTRY', 'HOLD', s)] = (mix(X, [f'A{i}_N' for i in range(nA)], 'HOLD', s), fA[s])
        series[('MIX_EXIT', 'TAKE_C1000', s)] = (mix(X, [f'AR_S{j}' for j in range(nS)], 'TAKE_C1000', s), fills['AR'][s])
        series[('MIX_EXIT', 'TB_SETTLE', s)] = (mix(X, [f'AR_S{j}' for j in range(nS)], 'TB_SETTLE', s), fills['AR'][s])
        series[('MIX_MAKER', EF.FIXED, s)] = (mix(X, [f'A{i}_SM' for i in range(nA)], EF.FIXED, s), fA[s])
        for i in range(nA):
            series[(f'A{i}', 'TAKE_C1000', s)] = (mix(X, [f'A{i}_S{j}' for j in range(nS)], 'TAKE_C1000', s), fills[f'A{i}'][s])
    cells = {}
    for sel in CF.SELS:
        m = X[sel].to_numpy(bool) & early
        for (cfg, pol, s), (pn, ff) in series.items():
            cells[f'{sel}|{cfg}_AB_s{int(s)}|{pol}'] = stats_of(pn, m, ff, wk, keys, dayk, per_day)
    # execution-luck draws for the primary cell: each action one (A_i, S_j), uniform
    m = X[sel_p].to_numpy(bool) & early
    rng = np.random.default_rng(SEED)
    M = np.stack([X[f'A{i}_S{j}_AB_s{int(s_p)}_TAKE_C1000_pnl'].to_numpy(float)[m] for i in range(nA) for j in range(nS)])
    draws = []
    for _ in range(DRAWS):
        k = rng.integers(0, M.shape[0], size=M.shape[1])
        pn = np.full(len(X), np.nan); pn[m] = M[k, np.arange(M.shape[1])]
        st_ = stats_of(pn, m, np.zeros(len(X)), wk, keys, dayk, per_day)
        draws.append((st_['total_usd'], st_['days_positive'], np.nan if st_['sharpe_day_equiv'] is None else st_['sharpe_day_equiv']))
    dr = np.array(draws, float)
    q = lambda v: dict(p5=float(np.nanquantile(v, .05)), median=float(np.nanmedian(v)), p95=float(np.nanquantile(v, .95)), nan=int(np.isnan(v).sum()))  # noqa: E731
    key = lambda cfg, pol='TAKE_C1000': f'{sel_p}|{cfg}_AB_s{int(s_p)}|{pol}'  # noqa: E731
    pm = cells[key('MIX_BOTH')]
    readings = dict(total_positive=bool(pm['total_usd'] > 0), days_positive=pm['days_positive'], days=len(days), sharpe_day_equiv=pm['sharpe_day_equiv'],
                    max_drawdown_usd=pm['max_drawdown_usd'], actions=pm['actions'], filled_expected=pm['filled'])
    decomp = dict(replay_L0=R0['cells'][f'{sel_p}|L0_AB_s{int(s_p)}|TAKE_C1000'], entry_only=cells[key('MIX_ENTRY')], exit_only=cells[key('MIX_EXIT')],
                  both=pm)
    by_A = [dict(accept_ms=ms['A'][i], **{k: cells[key(f'A{i}')][k] for k in ('total_usd', 'days_positive', 'sharpe_day_equiv', 'filled')}) for i in range(nA)]
    st = {d: json.loads((OUT_T / f'{d}.json').read_text()) for d in days}
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), receipt=CF.receipt_path().name, primary=p, days=list(days), rehearsal=REHEARSAL,
               measured={k: ms[k] for k in ('A', 'S', 'hold_live', 'sell_accept', 'Le', 'Lx', 'Le_replay', 'S_median')},
               replay_base=dict(order_ms=E.NET, hold_ms=E.HOLD, fill_notice_ms=M2.LEARN, binance_feed_ms=TM.FEED, decide_ms=TM.DECIDE),
               readings=readings, decomposition=decomp, primary_by_accept=by_A,
               draws=dict(n=DRAWS, seed=SEED, total_usd=q(dr[:, 0]), days_positive=q(dr[:, 1]), sharpe_day_equiv=q(dr[:, 2]),
                          share_total_positive=float(np.mean(dr[:, 0] > 0))),
               parity_cells={k: dict(cells=n, mismatched=b) for k, (n, b) in par.items()}, parity_checked=days == CF.FWD,
               statuses={d: dict(actions=v['actions'], unloaded=v['unloaded'], capped_loops=v['capped_loops'], grid_drops=v['grid_drops'],
                                 parity=v['parity']) for d, v in st.items()},
               cells=cells)
    TR.dump(RESULT, out)
    print('primary', p, json.dumps(TR.clean(readings)), 'parity', {k: (n, len(b)) for k, (n, b) in par.items()}, flush=True)


# ------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    chk = json.loads(CHECK.read_text())
    ms = measured(chk)
    C.require(len(ms['A']) == 10 and len(ms['S']) == 26 and ms['hold_live'] == 151. and abs(ms['Lx'] - (ms['sell_accept'] - 22. + 1.)) < 1e-12, f'measured {ms}')
    C.require(all(abs(le - (a - 21.)) < 1e-12 for a, le in zip(ms['A'], ms['Le'])), 'Le = A - NET + 1')
    bad = json.loads(json.dumps(chk)); bad['parity']['mismatched'] = 1
    try:
        measured(bad)
        raise AssertionError('measured accepted a check that did not reproduce')
    except ValueError as e_:
        assert 'did not reproduce' in str(e_), e_
    ent, ext = configs(ms)
    nx = {x['name']: x for x in ext}
    C.require(len(ent) == 3 + 10 + 1 and len(ext) == 3 + 10 * (1 + 26 + 1) + 26, f'config counts {len(ent)} {len(ext)}')
    C.require(all(nx[f'L{int(L)}']['pols'] == CF.POL and nx[f'L{int(L)}']['Lx'] == L for L in PAR_LATS), 'parity configs')
    # the parity configs' floor equals chase_forward.exits_day's expression bit for bit
    t = np.array([1795252800123.25, 1795253000456.5, 1795253999999.75])
    for L in PAR_LATS:
        ref = t + TM.FEED + TM.DECIDE + E.NET + L + E.HOLD + M2.LEARN + L
        C.require(np.array_equal(floor_start(t, float(L), M2.LEARN, float(L)), ref), f'floor order at L{L}')
    C.require(settle_x(100., 1858.6, 0., 5000.) == 1958.6 and settle_x(4000., 1858.6, 0., 5000.) is None, 'TB_SETTLE floor and cut')
    # parity3: positive control, then known-bad inputs it must flag
    new = pd.DataFrame(dict(day=['a', 'b'], t=[1., 2.], MOVE_top10=[True, False], L0_AB_s3_fill=[1., np.nan], L25_AB_s3_fill=[2., 3.],
                            A0_AB_s3_fill=[0., 1.], AR_S0_AB_s3_TAKE_C1000_pnl=[0., 1.]))
    old = new.drop(columns=['A0_AB_s3_fill', 'AR_S0_AB_s3_TAKE_C1000_pnl'])
    assert parity3(new, old)['mismatched'] == [] and parity3(new, old)['columns'] == 5, parity3(new, old)
    o2 = old.copy(); o2.loc[1, 'L25_AB_s3_fill'] = 0.
    assert parity3(new, o2)['mismatched'] == ['L25_AB_s3_fill'], parity3(new, o2)
    o3 = old.copy(); o3.loc[0, 't'] = 1.5
    assert parity3(new, o3)['mismatched'] == ['t'], parity3(new, o3)
    assert 'parity columns differ' in parity3(new.drop(columns=['L0_AB_s3_fill']), old)['mismatched']
    assert 'identity columns differ' in parity3(new.drop(columns=['MOVE_top10']), old)['mismatched']
    assert any('unexpected' in b for b in parity3(new.assign(junk=[0, 0]), old)['mismatched'])
    # mix: the mean over configs, NaN kept where every config is NaN
    Xm = pd.DataFrame({'A0_S0_AB_s3_TAKE_C1000_pnl': [1., 4., np.nan], 'A0_S1_AB_s3_TAKE_C1000_pnl': [3., 0., np.nan]})
    r = mix(Xm, ['A0_S0', 'A0_S1'], 'TAKE_C1000', 3.)
    assert r[0] == 2. and r[1] == 2. and np.isnan(r[2]), r
    # refusals, with their reasons
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
    print('chase_forward_tokyo selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    dict(selftest=selftest, read=stage_read, exits=lambda: stage_exits(a_[0]))[cmd]()
