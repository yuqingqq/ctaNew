"""Stage 3 of the chase live pilot: research's gate on Tokyo's 2-hour live shadow (pmmmrust docs/receipts/20261128_tokyo_chase_live_pilot_request,
stage 3; research's answer e1b80f2 section 5). USER 2026-11-28: "Yes we can prepare scripts now". P-2026-005, public data, no orders. Written and
declared before the shadow's data exists, and rehearsed on the stage-1 hour (tag rehearsal_stage1, the replay-fixes rows standing in for the
live rows), where it must reproduce chase_tokyo_2b's rows and chase_tokyo_replay_fixes' counts.
Stages (outputs write-once: OUT/chase_tokyo_shadow/<tag>/...; results chase_tokyo_shadow_<tag>[_reprice|_rehearsal].json here, counts and
delays only; the MOVE margins stay in OUT):
  convert <rec dir> <tag>   Tokyo's recording of one continuous run (binance-*, predict-*, markets-*.jsonl.gz, events.jsonl) -> research's
                            layouts as stage 2b's convert, for every UTC day it touches; book rows filed under their market's day.
  run <tag> <day>           research's frozen pipeline on one day of it (stage 2b's mode; one process per day: the frozen points and labels
                            lower their CPU limits, which a second day could not raise) -> research_rows_<day>.parquet.
  check <rec dir> <tag> <replay decisions.csv.gz> <live decisions.csv.gz>
     G1  the port at D against research's rows, row for row (chase_tokyo_replay_fixes' check 1 and its controls): no research row without
         Tokyo's; every row's inputs within stage 2b's tolerance, or differing only in trade-dependent groups (trades, flows) on a row with a
         trade of E in (t - 60 s, t] received after D (research's inputs have it, a port deciding by D cannot); the prediction within 1e-9 on
         the rows whose inputs agree; the D line's actions = research's in-scope actions, or differing only at such explained rows;
     G2  research's frozen model reproduces the at-frame prediction and MOVE of every replay and live row exactly, and the line re-derived in
         Pilot::decide's order gives every replay and live decision;
     G3  the live rows equal the replay's: the same rows (recv_ns, key), the 66 inputs, lim, md, prediction, MOVE and decision exactly;
     G4  at the frame against D, no input group differs without its cause (chase_tokyo_replay_fixes 2f), and the late trades recount exactly.
     Reported: the at-frame line's moves against research's in-scope actions with each difference attributed; the churn pooled with the
     stage-1 hour (chase_tokyo_replay_fixes.json; not when the run is that hour); the live decision latency, lag_us and X = dec - (t + 3.5 ms)
     at the last frame of research's matched points, overall and by burst stratum (chase_decision_delay's n_tr_20 edges), and lag_us's
     Spearman with n_tr_20; the run's events. Research rows within WARM_MS after a UTC day start (other than the run's own start) are
     counted apart from G1: research's readers read one day at a time, the port runs through midnight.
  reprice <tag>             the frozen line's 7-day readings (11-20..26, consumed by this line; chase_decision_atframe's stored exits, no new
                            exits) with the live decision delays: per burst stratum the hat weights of X on chase_decision_atframe's grid.
                            Control first: chase_decision_atframe's own AT_FRAME sample must reproduce its stored readings. 200 block-bootstrap
                            draws (1-minute blocks of the run) for R1 and R5; the churn stress on the live price.
  rehearsal <tag>           the tag's rows and check against chase_tokyo_2b's rows and chase_tokyo_replay_fixes' counts (the stage-1 hour).
The gate (research's, declared here): a go needs G1-G4, and R1 and R5 positive on the live delays. Otherwise the mismatches are the answer.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_tokyo_shadow.py <stage> ...
"""
import collections
import datetime as dt
import gzip
import json
import pickle
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_decision_atframe as CDF  # noqa: E402
import chase_decision_delay as CDD  # noqa: E402
import chase_forward as CF  # noqa: E402
import chase_forward_bigmove as CFB  # noqa: E402
import chase_forward_latency as CFL  # noqa: E402
import chase_forward_tokyo as CFT  # noqa: E402
import chase_tokyo_2b as CTB  # noqa: E402
import chase_tokyo_replay_fixes as CTRF  # noqa: E402

C, TR, CT, TM, EF = CF.C, CF.TR, CF.CT, CF.TM, CF.EF
OUT = CF.WT.OUT / 'chase_tokyo_shadow'
DAY_MS, WARM_MS, BLOCK_MS, AFTER_D_WIN = 86400000., CTB.WARM_MS, 60000., 60000.
CAL_DRAWS, SEED = 200, 20261202
TRADE_GROUPS = ('trades', 'flows')
REHEARSAL_SRC = 'tokyo-chase-stage1-20261128'


def res_fp(tag, kind=''):
    return H / f"chase_tokyo_shadow_{tag}{'_' + kind if kind else ''}.json"


def utc_day(ms):
    return dt.datetime.fromtimestamp(float(ms) / 1e3, dt.timezone.utc).strftime('%Y%m%d')


def after_midnight(t, start_ms):
    """Rows within WARM_MS after a UTC day start, other than the run's own warm-up."""
    t = np.asarray(t, float)
    return ((t - np.floor(t / DAY_MS) * DAY_MS) < WARM_MS) & (t >= start_ms + WARM_MS)


def model():
    R = json.loads((H / 'chase_forward_refit.json').read_text())
    with open(CF.FOUT / 'refit' / 'models.pkl', 'rb') as fh:
        return R['inputs'], float(R['thresholds']['MOVE_top10']), pickle.load(fh)['move']


def events_of(rec):
    ev = [json.loads(ln) for ln in open(Path(rec) / 'events.jsonl') if ln.strip()]
    st = [e for e in ev if e.get('kind') == 'start']
    C.require(len(st) == 1, f"expected one start event in the run's events.jsonl, found {len(st)}")
    return st[0]['recv_ns'] / 1e6, dict(collections.Counter(str(e.get('kind')) for e in ev))


def read_decisions(fp, with_d=True):
    with gzip.open(fp, 'rt') as fh:
        head = fh.readline().strip().split(',')
    cols = head if with_d else [c for c in head if not c.startswith('D_')]
    T = pd.read_csv(fp, usecols=cols, float_precision='round_trip')
    T = T.assign(t=T.t.astype(float), d=T.d.astype(float), key=T.key.astype(str), decision=T.decision.astype(str))
    if with_d:
        T['D_decision'] = T.D_decision.astype(str)
    T = T.reset_index(drop=True)
    return T.assign(row=np.arange(len(T)))


# ------------------------------------------------------------------------------------------------------------------ convert / run
def file_books(B, R):
    """Book rows -> {day: rows of the markets whose window starts that UTC day}."""
    day_of = {int(m): utc_day(ws * 1e3) for m, ws in zip(R.mid, R.ws)}
    dd = B.mid.map(day_of)
    return {d: B[(dd == d).to_numpy()] for d in sorted(set(dd.dropna()))}


def stage_convert(rec, tag):
    rec = Path(rec)
    C.require(rec.is_dir() and any(rec.glob('binance-*.jsonl.gz')), f'{rec} holds no recording')
    root = OUT / tag / 'data'
    C.require(not root.exists(), f'{root} exists (write-once)')
    start_ms, kinds = events_of(rec)
    lines = CTB.bn_lines(CTB.gz_frames(rec.glob('binance-*.jsonl.gz')))
    counts = {}
    for kind, by in lines.items():
        for hk, ls in by.items():
            p = root / 'raw' / kind / 'BTCUSDT' / f'{hk}.csv.gz'
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, 'wt') as fh:
                fh.write('\n'.join(ls) + '\n')
            counts[f'{kind}/{hk}'] = len(ls)
    days = sorted({hk[:8] for by in lines.values() for hk in by})
    R = CTB.registry_of(CTB.gz_frames(rec.glob('markets-*.jsonl.gz')))
    B = CTB.book_rows(CTB.gz_frames(rec.glob('predict-*.jsonl.gz')), R)
    byday = file_books(B, R)
    done = []
    for day in days:
        Rd = R[R.ws.map(lambda s: utc_day(s * 1e3)) == day]
        if not len(Rd) or day not in byday:
            continue
        s1 = root / 'step1' / day
        (s1 / 'books').mkdir(parents=True)
        Rd.to_parquet(s1 / 'registry.parquet', index=False)
        Bd = byday[day]
        for hk, g in Bd.groupby(Bd.recv_ns.map(CTB.hour_key)):
            g.to_parquet(s1 / 'books' / f'part_{hk}.parquet', index=False)
        TR.dump(s1 / 'clock_rl_d2.json', dict(day=day, shift_ms=0.0, d_dec=2.0, note="Tokyo's receipt is the known clock (stage 3)"))
        done.append(day)
    C.require(done, 'no day of the run has markets and books')
    wo = root / 'wt_out'
    wo.mkdir(parents=True)
    for day in done:
        pd.DataFrame({k: pd.Series(dtype=v) for k, v in CTB.LEGS_COLS.items()}).to_parquet(wo / f'winners_legs_{day}.parquet', index=False)
    pd.DataFrame({k: pd.Series(dtype=v) for k, v in CTB.FIRE_COLS.items()}).to_parquet(wo / 'fire_times_orders.parquet', index=False)
    TR.dump(root / 'convert.json', dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), rec=str(rec), start_ms=start_ms, events=kinds, days=days,
                                        days_with_markets=done, lines=counts, markets=int(len(R)), book_rows=int(len(B)),
                                        book_rows_by_day={d: int(len(byday[d])) for d in done}))
    print(json.dumps(dict(days=days, days_with_markets=done, lines=sum(counts.values()), markets=len(R), book_rows=len(B))), flush=True)


def stage_run(tag, day):
    root = OUT / tag / 'data'
    C.require((root / 'convert.json').exists(), 'run convert first')
    cv = json.loads((root / 'convert.json').read_text())
    C.require(day in cv['days_with_markets'], f'{day} is not a converted day of {tag}')
    fp = OUT / tag / f'research_rows_{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    CF.frozen()                                                              # the frozen points / labels set their own CPU limits
    ck = json.load(open(root / 'step1' / day / 'clock_rl_d2.json'))
    C.require(ck['shift_ms'] == 0.0, "the Tokyo-data clock must be Tokyo's receipt (shift 0)")
    CTB.apply_mode(day, root / 'raw', root / 'step1', root / 'wt_out', ck)
    od = OUT / tag / 'rows'
    CT.points(day, od)
    CT.labels(day, od)
    X, _, _ = CTB.score(day, od)
    X.to_parquet(fp, index=False)
    print(tag, day, len(X), 'rows', int(X.act.sum()), 'actions', int((X.act & X.early).sum()), 'in scope', flush=True)


def research_rows(tag):
    cv = json.loads((OUT / tag / 'data' / 'convert.json').read_text())
    fps = [OUT / tag / f'research_rows_{d}.parquet' for d in cv['days_with_markets']]
    C.require(all(p.exists() for p in fps), f'run every day of {tag} first: {cv["days_with_markets"]}')
    X = pd.concat([pd.read_parquet(p) for p in fps], ignore_index=True)
    return X.assign(t=X.t.astype(float), d=X.d.astype(float)), cv


# ------------------------------------------------------------------------------------------------------------------ check pieces
def after_d_trades(X, last, tr):
    """Per research row: the trades with E in (t - AFTER_D_WIN, t] received after the D of its tick's last frame (no hold limit)."""
    D = X[['t', 'd']].merge(last[['t', 'd', 'D_ms']], on=['t', 'd'], how='left', validate='1:1').D_ms.to_numpy(float)
    o = np.argsort(tr.E.to_numpy(float), kind='stable')
    E, Rms = tr.E.to_numpy(float)[o], tr.r.to_numpy()[o] / 1e6
    t = X.t.to_numpy(float)
    lo, hi = np.searchsorted(E, t - AFTER_D_WIN, 'right'), np.searchsorted(E, t, 'right')
    n = np.zeros(len(X), np.int64)
    for i in np.flatnonzero(np.isfinite(D)):
        n[i] = int(np.sum(Rms[lo[i]:hi[i]] > D[i]))
    return n


def at_d(X, Tk, kf, f32, G, tr, bt, warm, mid):
    """G1 (see the docstring) on the research rows outside the after-midnight band, the band apart; with chase_tokyo_replay_fixes' controls."""
    last = Tk.sort_values(['t', 'd', 'recv_ns'], kind='stable').groupby(['t', 'd'], as_index=False).tail(1)
    ren = {f'D_{c}': c for c in kf + ['lim', 'md', 'pred', 'MOVE', 'decision']}
    TD = last[['t', 'd', 'recv_ns', 'key', 'late_after_D_n'] + list(ren)].rename(columns=ren)
    nA = after_d_trades(X, last, tr)
    out = {}
    for nm, sel in (('main', ~mid), ('after_midnight', mid)):
        Xs = X[sel].assign(after_D_trades=nA[sel])
        if not len(Xs):
            out[nm] = dict(rows=0)
            continue
        res, both = CTB.compare(Xs, TD, kf + ['lim', 'md'], f32, -np.inf)
        bad_g = {g: np.zeros(len(both), bool) for g in G}
        for g, cols in G.items():
            for c in cols:
                bad_g[g] |= ~CTB.same_values(both[f'{c}_r'], both[f'{c}_t'], c in f32, CTB.TOL32)
        anybad = np.any(np.stack(list(bad_g.values())), axis=0)
        only_trade = anybad & ~np.any(np.stack([bad_g[g] for g in G if g not in TRADE_GROUPS]), axis=0)
        expl = only_trade & (both.after_D_trades.to_numpy() > 0)
        agree = ~anybad
        pr = CTB.same_values(both.pred_r, both.pred_t, False)
        out[nm] = dict(rows=int(len(Xs)), matched=res['matched'], research_only=res['research_only'], rows_inputs_equal=int(agree.sum()),
                       rows_explained_by_trades_after_D=int(expl.sum()), rows_unexplained=int((anybad & ~expl).sum()),
                       inputs_differ={c: v['differ'] for c, v in res['per_input'].items() if v['differ']},
                       pred_differ_where_inputs_equal=int((agree & ~pr).sum()),
                       unexplained_examples=[dict(t=float(both.t.iat[i]), d=float(both.d.iat[i]), groups=[g for g in G if bad_g[g][i]])
                                             for i in np.flatnonzero(anybad & ~expl)[:8]])
        if nm == 'main':
            main_both, main_expl = both, expl
    A_r = X[X.act.to_numpy(bool) & X.early.to_numpy(bool) & warm & ~mid]
    A_D = Tk[Tk.D_decision == 'act']
    sr, sD = set(zip(A_r.t, A_r.d)), set(zip(A_D.t, A_D.d)) - set(zip(X.t[mid], X.d[mid]))
    ex_rows = set(zip(main_both.t[main_expl], main_both.d[main_expl]))
    keyof = dict(zip(zip(X.t, X.d), X.key))
    ex_moves = {keyof[x] for x in ex_rows if x in keyof}
    diff_a = (sr ^ sD)
    out['actions'] = dict(research_in_scope=int(len(sr)), D_line=int(len(sD)), both=int(len(sr & sD)), differ=int(len(diff_a)),
                          differ_unexplained=int(sum(1 for x in diff_a if keyof.get(x) not in ex_moves)))
    # controls (chase_tokyo_replay_fixes'): the at-frame columns must differ, a planted change must be counted
    Xm = X[~mid]
    TA = last[['t', 'd', 'recv_ns', 'key'] + kf + ['lim', 'md', 'pred', 'MOVE', 'decision']]
    resA, _ = CTB.compare(Xm, TA, kf + ['lim', 'md'], f32, -np.inf)
    TP = TD.copy()
    inR = TP.set_index(['t', 'd']).index.isin(Xm.set_index(['t', 'd']).index)
    v = TP.its_100_btc.to_numpy(float).copy()
    ix = np.flatnonzero(inR & (v > 0) & (v < 100))[:CTRF.PLANT_N]
    C.require(len(ix) == CTRF.PLANT_N, 'too few rows to plant')
    base, _ = CTB.compare(Xm, TD, kf + ['lim', 'md'], f32, -np.inf)
    v[ix] += CTRF.PLANT_ABS
    TP['its_100_btc'] = v
    resP, _ = CTB.compare(Xm, TP, kf + ['lim', 'md'], f32, -np.inf)
    planted = resP['per_input']['its_100_btc']['differ'] - base['per_input']['its_100_btc']['differ']
    C.require(resA['rows_all_inputs_equal'] < resA['matched'], "the control failed: research's rows equal Tokyo's at-frame columns")
    C.require(planted == CTRF.PLANT_N, f'the control failed: {planted} planted differences counted, not {CTRF.PLANT_N}')
    Dm = np.maximum(Tk.t.to_numpy(float) + CTRF.G_AFTER, Tk.recv_ns.to_numpy() / 1e6 + CTRF.D_LEAD)
    lastbook = bt.groupby('E').r.max()
    ft = main_both.t.map(lastbook)
    C.require(ft.notna().all(), "a research row without its tick's bookTicker frame in Tokyo's tape")
    out.update(D_ms_exact=int(CTRF.exact(Dm, Tk.D_ms).sum()), rows=int(len(Tk)),
               receipt_is_last_book_frame=int(np.sum(main_both.recv_ns.astype('int64').to_numpy() == ft.astype('int64').to_numpy())),
               control_at_frame_rows_all_equal=resA['rows_all_inputs_equal'], control_planted_counted=int(planted))
    g1 = (out['main']['research_only'] == 0 and out['main']['rows_unexplained'] == 0 and out['main']['pred_differ_where_inputs_equal'] == 0
          and out['actions']['differ_unexplained'] == 0 and out['D_ms_exact'] == len(Tk))
    return out, bool(g1), last, main_both


def scores_and_line(T, kf, thr, score, label):
    """G2 on one decision file: research's prediction, MOVE and re-derived decisions against T's -> (dict, ok, derived, MOVE)."""
    pred = score(T)
    mv = CTRF.move_of(pred, T.d.to_numpy(float), T.lim.to_numpy(float), T.md.to_numpy(float))
    der = CTRF.derive(T.decision.to_numpy(), T.key.to_numpy(), T.d.to_numpy(float), mv, thr, CTRF.early_of(T.t))
    mis = collections.Counter((a, b) for a, b in zip(T.decision, der) if a != b)
    o = dict(rows=int(len(T)), pred_exact=int(CTRF.exact(pred, T.pred).sum()), move_exact=int(CTRF.exact(mv, T.MOVE).sum()),
             decisions_equal=int(np.sum(T.decision.to_numpy() == der)), mismatches=[[a, b, n] for (a, b), n in mis.items()],
             actions=int(np.sum(der == 'act')), file_actions=int(np.sum(T.decision == 'act')))
    ok = o['pred_exact'] == o['move_exact'] == o['decisions_equal'] == len(T)
    return o, bool(ok), der, mv


def live_vs_replay(Tl, Tk, kf):
    """G3: the live rows against the replay's, on (recv_ns, key)."""
    cols = kf + ['lim', 'md', 'pred', 'MOVE']
    m = Tl[['recv_ns', 'key', 't', 'd', 'decision'] + cols].merge(Tk[['recv_ns', 'key', 't', 'd', 'decision'] + cols], on=['recv_ns', 'key'],
                                                                     how='outer', suffixes=('_l', '_r'), indicator=True, validate='1:1')
    b = m[m._merge == 'both']
    per = {c: int((~CTRF.exact(b[f'{c}_l'], b[f'{c}_r'])).sum()) for c in cols + ['t', 'd']}
    dec = int(np.sum(b.decision_l.astype(str).to_numpy() != b.decision_r.astype(str).to_numpy()))
    o = dict(live_rows=int(len(Tl)), replay_rows=int(len(Tk)), both=int(len(b)), live_only=int((m._merge == 'left_only').sum()),
             replay_only=int((m._merge == 'right_only').sum()), columns_differ={c: n for c, n in per.items() if n}, decisions_differ=dec)
    ok = o['live_only'] == o['replay_only'] == 0 and not o['columns_differ'] and dec == 0
    return o, bool(ok)


def moves_and_attribution(X, Tk, der, last, both, G, score, thr, warm, mid, lastbook):
    """chase_tokyo_replay_fixes 2c / 2d on the run: the at-frame moves against research's in-scope actions (outside the midnight band).
    A deciding row's last_row_of_its_tick: no later row of its (t, d); ms_last_frame: it is its millisecond's last bookTicker frame in the tape
    (a tick's only row can be a provisional state that a later frame of the millisecond undoes)."""
    kp = both.groupby('key_t').key_r.agg(lambda s: sorted(set(s)))
    C.require(all(len(v) == 1 for v in kp), 'a move key pairs with more than one move')
    t2r = {k: v[0] for k, v in kp.items()}
    is_last = np.zeros(len(Tk), bool)
    is_last[last.row.to_numpy()] = True
    Xi = X.set_index(['t', 'd'])
    A_r = X[X.act.to_numpy(bool) & X.early.to_numpy(bool) & warm & ~mid]
    rmv = {k: (t, d) for k, t, d in zip(A_r.key, A_r.t, A_r.d)}
    band = set(zip(X.t[mid], X.d[mid]))
    fmv = {}
    for r_ in Tk[der == 'act'].itertuples():
        if (r_.t, r_.d) not in band:
            fmv[t2r.get(r_.key, 'tokyo:' + r_.key)] = (r_.t, r_.d, r_.row)
    moves, priv = [], []
    for k in sorted(set(rmv) | set(fmv)):
        a_r, a_f = rmv.get(k), fmv.get(k)
        cls = ('same_tick' if a_r[0] == a_f[0] else 'another_tick') if (a_r and a_f) else ('lost' if a_r else 'added')
        m = dict(move=k, cls=cls, research_t=a_r[0] if a_r else None, frame_t=a_f[0] if a_f else None, d=(a_r or a_f)[1])
        if cls == 'same_tick':
            moves.append(m)
            continue
        if cls == 'added' or (cls == 'another_tick' and a_f[0] < a_r[0]):
            i = a_f[2]
        else:
            i = int(last.row.to_numpy()[(last.t.to_numpy(float) == a_r[0]) & (last.d.to_numpy(float) == a_r[1])][0])
        if a_f:
            m['research_row_at_frame_tick'] = bool((a_f[0], a_f[1]) in Xi.index)
        rw = Tk.iloc[i]
        A = {c: float(rw[c]) for c in list(G['trades'] + G['flows'] + G['depth'] + G['book'] + G['predict'])}
        A['d'] = float(rw.d)
        B = {c: float(rw[f'D_{c}']) for c in A if c != 'd'}
        B['d'] = float(rw.d)
        at, pv = CTRF.attribute(A, B, G, score, thr)
        m.update(deciding_row=dict(t=float(rw.t), recv_ns=int(rw.recv_ns), last_row_of_its_tick=bool(is_last[i]),
                                   ms_last_frame=bool(int(lastbook.get(int(rw.t), -1)) == int(rw.recv_ns)), frame_decision=str(rw.decision),
                                   D_decision=str(rw.D_decision)), **at)
        moves.append(m)
        priv.append(dict(move=k, cls=cls, t=float(rw.t), d=float(rw.d), **pv))
    cnt = dict(collections.Counter(m['cls'] for m in moves))
    return dict(counts=cnt, research_in_scope=int(len(rmv)), at_frame=int(len(fmv)), differing=[m for m in moves if m['cls'] != 'same_tick']), priv


def information(Tk, X, last, raw, f32, G, mid):
    """G4b and G4 (chase_tokyo_replay_fixes 2e / 2f) on the replay's rows."""
    bt, tr, dp = CTRF.tape(raw)
    allr = np.concatenate([bt.r.to_numpy(), tr.r.to_numpy(), dp.r.to_numpy()])
    L = CTRF.late_counts(Tk, tr, allr, dp, bt)
    Lk = CTRF.late_counts(Tk, tr, allr, dp, bt, shift_ns=1_000_000)
    eq = dict(n=CTRF.exact(L['n'], Tk.late_n), its=np.isclose(L['its'], Tk.late_its_btc, rtol=1e-12, atol=0.),
              ag=np.isclose(L['ag'], Tk.late_ag_btc, rtol=1e-12, atol=0.), after_D=CTRF.exact(L['after_D'], Tk.late_after_D_n))
    kbl = int(np.sum(~CTRF.exact(Lk['n'], Tk.late_n)))
    C.require(kbl > 0, 'the known-bad (decision 1 ms later) changed no late count')
    is_last = np.zeros(len(Tk), bool)
    is_last[last.row.to_numpy()] = True
    ib = is_last & Tk.set_index(['t', 'd']).index.isin(X[~mid].set_index(['t', 'd']).index)
    D_its = Tk.D_its_100_btc.to_numpy(float)
    late = dict(rows_equal={k: int(v.sum()) for k, v in eq.items()}, known_bad_rows_changed=kbl,
                research_rows=dict(n=int(ib.sum()), rows_with_late_trades=int(np.sum(L['n'][ib] > 0)), late_trades=int(L['n'][ib].sum()),
                                   late_after_D=int(L['after_D'][ib].sum()), its_share=float(L['its'][ib].sum() / max(D_its[ib].sum(), 1e-12)),
                                   rows_depth_fD=int(np.sum(L['depth_fD'][ib] > 0)), rows_book_fD=int(np.sum(L['book_fD'][ib] > 0))),
                all_rows=dict(rows=int(len(Tk)), rows_with_late_trades=int(np.sum(L['n'] > 0)), provisional_rows=int(np.sum(~is_last))))
    prov = ~is_last
    ltr, ldp, lbk = L['trades_fD'] > 0, L['depth_fD'] > 0, L['book_fD'] > 0
    expl = dict(trades=ltr, flows=ltr | prov | lbk, depth=ldp | prov, book=prov | lbk)
    grp = {}
    for g, cols in G.items():
        dif = np.zeros(len(Tk), bool)
        for c in cols:
            dif |= ~CTRF.exact(Tk[c], Tk[f'D_{c}'], c in f32)
        e_ = dict(rows_differ=int(dif.sum()), research_rows_differ=int((dif & ib).sum()))
        if g in expl:
            e_.update(unexplained=int((dif & ~expl[g]).sum()),
                      unexplained_examples=[dict(t=float(Tk.t.iat[i]), d=float(Tk.d.iat[i]), recv_ns=int(Tk.recv_ns.iat[i])) for i in np.flatnonzero(dif & ~expl[g])[:5]])
        grp[g] = e_
    pf = Tk.recv_ns.to_numpy() / 1e6 - Tk.pf_age_ms.to_numpy(float)
    pD = Tk.t.to_numpy(float) + CTRF.G_AFTER - Tk.D_pf_age_ms.to_numpy(float)
    other = ~np.isclose(pf, pD, rtol=0., atol=CTRF.STAMP_TOL) & np.isfinite(pf) & np.isfinite(pD)
    grp['predict'].update(research_rows_other_known_frame=int((other & ib).sum()), research_rows_frame_newer=int((other & ib & (pf > pD)).sum()),
                          research_rows_frame_older=int((other & ib & (pf < pD)).sum()))
    ok = all(v == len(Tk) for v in late['rows_equal'].values()) and all(grp[g]['unexplained'] == 0 for g in expl)
    return late, grp, bool(ok), tr


def latency(Tl, X, warm, mid, raw, edges):
    """The live decision latency at the last frame of research's matched points: lag_us and X = dec - (t + 3.5 ms), by burst stratum."""
    _, trd, _ = CDD.tokyo_frames(raw)                                            # de-duplicated, zero prints dropped, E-sorted
    ll = Tl.sort_values(['t', 'd', 'recv_ns'], kind='stable').groupby(['t', 'd'], as_index=False).tail(1)
    keep = X[warm & ~mid][['t', 'd']]
    m = keep.merge(ll[['t', 'd', 'recv_ns', 'dec_ns', 'lag_us', 'decide_us']], on=['t', 'd'], how='inner', validate='1:1')
    t = m.t.to_numpy(float)
    xl = m.dec_ns.to_numpy() / 1e6 - (t + CTRF.G_AFTER)
    lam = m.recv_ns.to_numpy() / 1e6 - t
    lag = m.lag_us.to_numpy(float) / 1e3
    n20 = CDD.count_in(trd.E.to_numpy(float), t, 20.)
    sb = CDD.strata_of(n20, edges)
    blk = np.floor((t - t.min()) / BLOCK_MS).astype(int)
    by = {int(b): dict(points=int(np.sum(sb == b)), lag_ms=CDD.quant(lag[sb == b]), x_ms=CDD.quant(xl[sb == b]), lam_ms=CDD.quant(lam[sb == b]))
          for b in range(CDD.NSTRATA)}
    sp = float(pd.Series(lag).corr(pd.Series(n20), method='spearman')) if len(lag) > 2 else None
    out = dict(points=int(len(m)), lag_ms=CDD.quant(lag), x_ms=CDD.quant(xl), lam_ms=CDD.quant(lam), by_stratum=by, spearman_lag_n_tr_20=sp,
               all_rows_lag_ms=CDD.quant(Tl.lag_us.to_numpy(float) / 1e3), all_rows_decide_us=CDD.quant(Tl.decide_us.to_numpy(float)),
               samples=dict(x=[round(float(v), 6) for v in xl], stratum=[int(b) for b in sb], block=[int(b) for b in blk]))
    return out


# ------------------------------------------------------------------------------------------------------------------ check
def stage_check(rec, tag, replay_fp, live_fp):
    fp, pv = res_fp(tag), OUT / tag / 'private.json'
    C.require(not fp.exists() and not pv.exists(), f'{fp.name} or {pv.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    CF.frozen()
    kf, thr, mo = model()
    X, cv = research_rows(tag)
    C.require(np.array_equal(CTRF.early_of(X.t), X.early.to_numpy(bool)), "research's scope formula does not give research's stored early")
    f32 = [c for c in kf if str(X[c].dtype) == 'float32']
    G = CTRF.groups(kf)
    score = CTRF.scorer(mo, kf, {c: X[c].dtype for c in kf})
    start_ms, kinds = events_of(rec)
    C.require(start_ms == cv['start_ms'], "the run's start differs from the converted recording's")
    warm = X.t.to_numpy(float) >= start_ms + WARM_MS
    mid = after_midnight(X.t, start_ms)
    raw = OUT / tag / 'data' / 'raw'
    Tk = read_decisions(replay_fp)
    Tl = read_decisions(live_fp, with_d=False)
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), tag=tag, inputs=dict(replay_sha256=CTRF.sha(replay_fp), live_sha256=CTRF.sha(live_fp)),
               start_ms=start_ms, events=kinds, days=cv['days_with_markets'], research_rows=int(len(X)), research_rows_after_midnight=int(mid.sum()))
    bt, trr, _ = CTRF.tape(raw)
    out['G1_at_D'], g1, last, both = at_d(X, Tk, kf, f32, G, trr, bt, warm, mid)
    rs, g2r, der, _ = scores_and_line(Tk, kf, thr, score, 'replay')
    ls, g2l, _, _ = scores_and_line(Tl, kf, thr, score, 'live')
    Kb = Tk[kf].copy()
    Kb['its_100_btc'] = 0.
    kb = int(np.sum(~CTRF.exact(score(Kb), score(Tk))))
    C.require(kb > .01 * len(Tk), f'the known-bad (its_100_btc zeroed) changed only {kb} predictions')
    out['G2_at_frame'] = dict(replay=rs, live=ls, known_bad_changed=kb)
    out['G3_live_vs_replay'], g3 = live_vs_replay(Tl, Tk, kf)
    out['late'], out['G4_at_frame_vs_D'], g4, _ = information(Tk, X, last, raw, f32, G, mid)
    out['moves'], priv = moves_and_attribution(X, Tk, der, last, both, G, score, thr, warm, mid, bt.groupby('E').r.max())
    st1 = json.loads(CTRF.RESULT.read_text())
    same_hour = abs(st1['start_ms'] - start_ms) < 1.
    c0, c1 = out['moves']['counts'], st1['moves']['counts']
    n0, n1 = out['moves']['research_in_scope'], st1['moves']['research_in_scope']
    ch = lambda c: c.get('lost', 0) + c.get('added', 0)  # noqa: E731
    out['churn'] = dict(run=dict(changed=ch(c0), research=n0), stage1=None if same_hour else dict(changed=ch(c1), research=n1),
                        pooled=(ch(c0) / max(n0, 1)) if same_hour else (ch(c0) + ch(c1)) / max(n0 + n1, 1), pooled_with_stage1=not same_hour)
    out['latency'] = latency(Tl, X, warm, mid, raw, json.loads(CDD.TOKYO.read_text())['strata']['edges'])
    out['gate'] = dict(G1=g1, G2=bool(g2r and g2l), G3=g3, G4=g4)
    TR.dump(fp, out)
    pv.parent.mkdir(parents=True, exist_ok=True)
    TR.dump(pv, dict(asof=out['asof'], moves=priv))
    print(json.dumps(TR.clean(dict(gate=out['gate'], at_D={k: out['G1_at_D']['main'].get(k) for k in ('matched', 'research_only', 'rows_unexplained')},
                                   actions=out['G1_at_D']['actions'], moves=out['moves']['counts'], churn=out['churn']['pooled'],
                                   lag=out['latency']['lag_ms'].get('p50'), x=out['latency']['x_ms'].get('mean')))), flush=True)


# ------------------------------------------------------------------------------------------------------------------ reprice
def values():
    """chase_decision_atframe.stage_read's rows and per-grid values on its stored exits (the same code): J, V, Fv, bA, stats."""
    rc = CF.frozen()
    p, sel_p, s_p = CFL.primary(rc)
    C.require(sel_p == CDF.SEL, f"the receipt's primary selection is {sel_p}, the rows here are {CDF.SEL}")
    s = int(s_p)
    ms = CFB.measured()
    days = tuple(d for d in CF.FWD if (CDF.OUT_A / f'{d}.parquet').exists())
    C.require(days == CF.FWD, f'exits_atframe has {len(days)} of {len(CF.FWD)} days')
    A = pd.concat([pd.read_parquet(CDF.OUT_A / f'{d}.parquet') for d in days], ignore_index=True)
    B = pd.concat([pd.read_parquet(CDD.OUT_D / f'{d}.parquet') for d in days], ignore_index=True)
    nk, nS = len(ms['A']), len(ms['S'])
    bcols = [c for c in B.columns if (c.startswith('D') and 'X' in c.split('_')[0]) or (c.startswith('mv_') and '_X' in c)]
    J = A.merge(B[['day', 't', 'd'] + [c for c in bcols if c not in A.columns]], on=['day', 't', 'd'], how='left', validate='1:1', indicator=True)
    inb = (J._merge == 'both').to_numpy()
    C.require(int(inb.sum()) == len(B), "a chase_decision_delay row is missing from the at-frame rows")
    keys, dayk = [], []
    for d in days:
        reg = CF.MM.registry(d).sort_values('ws')
        keys += [f'{d}:{int(m)}' for m in reg.mid]
        dayk += [d] * len(reg)
    keys, dayk = np.array(keys), np.array(dayk)
    CF.SX.DAYS = days
    per_day = len(keys) / len(days)
    wk = (J.day.astype(str) + ':' + J.mid.astype(np.int64).astype(str)).to_numpy()
    big = [k for k, r in enumerate(ms['rows']) if r['probe'] == 'big']
    allk = list(range(nk))
    cov = {k_: np.array([r[k_] for r in ms['rows']], float) for k_ in ('up100', 'n_post200')}
    cma = J.cut_minus_act_ms.to_numpy(float)
    m_all = np.ones(len(J), bool)
    every = lambda ks: np.tile(np.isin(np.arange(nk), ks), (len(J), 1))  # noqa: E731
    ng = len(CDF.GRID_ALL)
    V = {r: np.zeros((ng, len(J))) for r in CDF.READINGS}
    Fv = {r: np.zeros((ng, len(J))) for r in CDF.READINGS}
    for gi, x in enumerate(CDF.GRID_ALL):
        tag = f'Q{CDF.GRID_NEG.index(x)}' if x in CDF.GRID_NEG else f'X{CDD.GRID.index(x)}'
        P = np.stack([np.stack([J[f'D{k}{tag}_S{j}_AB_s{s}_TAKE_C1000_pnl'].to_numpy(float) for j in range(nS)]) for k in range(nk)]).mean(axis=1)
        F = np.stack([(np.nan_to_num(J[f'D{k}{tag}_AB_s{s}_fill'].to_numpy(float)) > 0).astype(float) for k in range(nk)])
        mx = (cma - x) >= EF.EARLY_MS
        up, npost = J[f'mv_up100_{tag}'].to_numpy(float), J[f'mv_n_post200_{tag}'].to_numpy(float)
        sets = dict(R1_BIG=every(big), R2_BIG_MOVE=CFB.strata_sets(up, cov['up100'][big], big, nk, up[mx])[0],
                    R3_BIG_BURST=CFB.strata_sets(npost, cov['n_post200'][big], big, nk, npost[mx])[0], R4_BOTH=every(allk),
                    R5_BOTH_BURST=CFB.strata_sets(npost, cov['n_post200'], allk, nk, npost[mx])[0])
        for r, st_ in sets.items():
            C.require(st_ is not None, f'{r} is not computable at x = {x:g}')
            pn, ff = CFB.set_mean(P, F, st_)
            V[r][gi] = np.where(mx, np.nan_to_num(pn), 0.)
            Fv[r][gi] = np.where(mx, ff, 0.)
    stats = lambda v, f: CFT.stats_of(v, m_all, f, wk, keys, dayk, per_day)  # noqa: E731
    bA = CDD.strata_of(J[CDF.COV].to_numpy(float), json.loads(CDD.TOKYO.read_text())['strata']['edges'])
    return J, V, Fv, bA, stats


def readings_of(W, V, Fv, bA, stats):
    return {r: stats(np.sum(W[bA].T * V[r], axis=0), np.sum(W[bA].T * Fv[r], axis=0)) for r in CDF.READINGS}


def stage_reprice(tag):
    fp = res_fp(tag, 'reprice')
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    ck = json.loads(res_fp(tag).read_text())
    A0 = json.loads(CDF.RESULT.read_text())
    J, V, Fv, bA, stats = values()
    tk = json.loads(CDF.TOKYO.read_text())
    lam, sb = (np.array(tk['samples'][k]) for k in ('lam', 'stratum'))
    xs, sbs = CDF.rule_sample('AT_FRAME', lam, sb)
    W0, _, _, _ = CDF.weights_for(xs, sbs)
    R0 = readings_of(W0, V, Fv, bA, stats)
    flds = ('total_usd', 'days_positive', 'sharpe_day_equiv', 'filled', 'daily_usd')
    par = {r: [k for k in flds if not CDF.close(json.loads(json.dumps(TR.clean(R0[r])))[k], A0['readings'][r]['AT_FRAME'][k])] for r in CDF.READINGS}
    C.require(not any(par.values()), f"the control failed: chase_decision_atframe's AT_FRAME is not reproduced: {par}")
    sm = ck['latency']['samples']
    xl, sl, bl = np.array(sm['x'], float), np.array(sm['stratum'], int), np.array(sm['block'], int)
    C.require(len(xl) > 0, 'no live decision delays')
    WL, lo, hi, empty = CDF.weights_for(xl, sl, fallback=W0)
    RL = readings_of(WL, V, Fv, bA, stats)
    rng = np.random.default_rng(SEED)
    nb = int(bl.max() + 1)
    draws = {r: [] for r in ('R1_BIG', 'R5_BOTH_BURST')}
    emp = 0
    for _ in range(CAL_DRAWS):
        idx = np.concatenate([np.flatnonzero(bl == b) for b in rng.integers(0, nb, nb)])
        Wd, _, _, e = CDF.weights_for(xl[idx], sl[idx], fallback=WL)
        emp += e > 0
        for r in draws:
            draws[r].append(float(np.sum(np.sum(Wd[bA].T * V[r], axis=0)) / 100.))
    q = lambda v: dict(p5=float(np.quantile(v, .05)), median=float(np.median(v)), p95=float(np.quantile(v, .95)), share_positive=float(np.mean(np.array(v) > 0)))  # noqa: E731
    churn = ck['churn']['pooled']
    exp_x = dict(control=float(np.mean([np.dot(W0[b], CDF.GRID_ALL) for b in bA])), live=float(np.mean([np.dot(WL[b], CDF.GRID_ALL) for b in bA])))
    g5 = bool(RL['R1_BIG']['total_usd'] > 0 and RL['R5_BOTH_BURST']['total_usd'] > 0)
    gate = dict(ck['gate'], G5_live_price_positive=g5)
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), tag=tag, rows=int(len(J)), control_reproduced=True,
               readings=dict(control=R0, live=RL), live_minus_control_usd={r: RL[r]['total_usd'] - R0[r]['total_usd'] for r in CDF.READINGS},
               expected_x_ms=exp_x, clamped=dict(below_first=lo, above_last=hi, empty_strata=empty, sample=int(len(xl))),
               calibration_draws=dict(n=CAL_DRAWS, seed=SEED, blocks=nb, empty_stratum_draws=int(emp), **{r: q(v) for r, v in draws.items()}),
               churn_stress=dict(churn=churn, R1_BIG=RL['R1_BIG']['total_usd'] * (1 - churn), R5_BOTH_BURST=RL['R5_BOTH_BURST']['total_usd'] * (1 - churn)),
               gate=gate, go=bool(all(gate.values())))
    TR.dump(fp, out)
    print(json.dumps(TR.clean(dict(live={r: round(v['total_usd'], 1) for r, v in RL.items()}, control={r: round(v['total_usd'], 1) for r, v in R0.items()},
                                   exp_x=exp_x, gate=gate, go=out['go']))), flush=True)


# ------------------------------------------------------------------------------------------------------------------ rehearsal
def stage_rehearsal(tag):
    fp = res_fp(tag, 'rehearsal')
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    X, cv = research_rows(tag)
    X0 = pd.read_parquet(CDD.SRC / 'research_rows.parquet')
    X0 = X0.assign(t=X0.t.astype(float), d=X0.d.astype(float))
    same_cols = list(X0.columns) == list(X.columns)
    eq = {c: bool(X0[c].reset_index(drop=True).equals(X[c].reset_index(drop=True))) for c in X0.columns if c in X.columns}
    ck, r0 = json.loads(res_fp(tag).read_text()), json.loads(CTRF.RESULT.read_text())
    pairs = dict(matched=(ck['G1_at_D']['main']['matched'], r0['at_D']['matched']), actions_both=(ck['G1_at_D']['actions']['both'], r0['at_D']['actions_both']),
                 replay_actions=(ck['G2_at_frame']['replay']['actions'], r0['line_rederived']['actions']), moves=(ck['moves']['counts'], r0['moves']['counts']),
                 late_rows=(ck['late']['research_rows']['rows_with_late_trades'], r0['late']['research_rows']['rows_with_late_trades']),
                 its_share=(round(ck['late']['research_rows']['its_share'], 12), round(r0['late']['research_rows']['its_share_tokyo_def'], 12)),
                 other_known_frame=(ck['G4_at_frame_vs_D']['predict']['research_rows_other_known_frame'], r0['at_frame_vs_D']['predict']['research_rows_other_known_frame']))
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), tag=tag, rows=(int(len(X)), int(len(X0))), same_columns=same_cols,
               columns_equal=int(sum(eq.values())), columns_differ=[c for c, v in eq.items() if not v],
               check_vs_replay_fixes={k: dict(shadow=a, replay_fixes=b, equal=a == b) for k, (a, b) in pairs.items()})
    out['reproduced'] = bool(same_cols and not out['columns_differ'] and len(X) == len(X0) and all(v['equal'] for v in out['check_vs_replay_fixes'].values()))
    TR.dump(fp, out)
    print(json.dumps(TR.clean(dict(reproduced=out['reproduced'], rows=out['rows'], columns_differ=out['columns_differ'],
                                   unequal=[k for k, v in out['check_vs_replay_fixes'].items() if not v['equal']]))), flush=True)


# ------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    C.require(utc_day(1795860000000.) == '20261128' and utc_day(1795910400000. - 1) == '20261128' and utc_day(1795910400000.) == '20261129', 'utc_day')
    t = np.array([1795910400000. + 5000., 1795910400000. + 70000., 1795910400000. - 5000.])
    C.require(after_midnight(t, 0.).tolist() == [True, False, False] and after_midnight(t, t[0] - 1000.).tolist() == [False, False, False], 'after_midnight')
    R = pd.DataFrame(dict(mid=[1, 2], ws=[1795910100., 1795910400.]))
    B = pd.DataFrame(dict(mid=[1, 2, 2, 3], recv_ns=[1, 2, 3, 4]))
    fb = file_books(B, R)
    C.require(sorted(fb) == ['20261128', '20261129'] and fb['20261128'].mid.tolist() == [1] and fb['20261129'].mid.tolist() == [2, 2], f'file_books {fb}')
    kf = ['a', 'b']
    Tk = pd.DataFrame(dict(recv_ns=[1, 2, 3], key=['1:up', '1:up', '2:down'], t=[5., 5., 6.], d=[1., 1., -1.], decision=['act', 'MoveActed', 'BelowThreshold'],
                           a=[.1, .2, .3], b=[1., np.nan, 2.], lim=[50.] * 3, md=[50.5] * 3, pred=[1., 2., 3.], MOVE=[.5, 1.5, 2.5]))
    o, ok = live_vs_replay(Tk.copy(), Tk, kf)
    C.require(ok and o['both'] == 3, f'live_vs_replay same {o}')
    Tl = Tk.copy()
    Tl.loc[1, 'a'] = .25
    Tl.loc[2, 'decision'] = 'act'
    o, ok = live_vs_replay(Tl.iloc[:3], Tk.iloc[:2], kf)
    C.require(not ok and o['columns_differ'] == {'a': 1} and o['live_only'] == 1 and o['decisions_differ'] == 0, f'live_vs_replay planted {o}')
    for fn, args in ((stage_check, ('/nonexistent', 'selftest_tag', '/x', '/y')), (stage_reprice, ('selftest_tag',)), (stage_rehearsal, ('selftest_tag',))):
        keep = globals()['res_fp']
        globals()['res_fp'] = lambda tag, kind='': Path(__file__)
        try:
            fn(*args)
            raise AssertionError(f'{fn.__name__} overwrote its result')
        except ValueError as e_:
            assert 'write-once' in str(e_), e_
        finally:
            globals()['res_fp'] = keep
    print('chase_tokyo_shadow selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    dict(selftest=selftest, convert=lambda: stage_convert(*a_), run=lambda: stage_run(*a_), check=lambda: stage_check(*a_),
         reprice=lambda: stage_reprice(*a_), rehearsal=lambda: stage_rehearsal(*a_))[cmd]()
