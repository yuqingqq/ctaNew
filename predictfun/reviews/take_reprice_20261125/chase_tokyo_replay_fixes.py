"""Research's two checks on Tokyo's replay of the stage-1 hour with both port fixes, and Tokyo's three questions (pmmmrust ba5aae4 /
cb32854: the replay with research's fixes, port 3efb306 merged as f608e51; data branch data/tokyo-chase-stage1-20261128 21d543b,
replay-fixes/decisions.csv.gz). USER 2026-11-28: "Check the new request". P-2026-005, public data, no orders. Declared before the run.
Nothing is chosen and there is no PASS rule: every reported count is computed here.
1. The port at D, row for row. Research's frozen rows on Tokyo's recording (chase_tokyo_2b's research_rows.parquet, 6,370 rows, the sha256
   chase_decision_delay recorded), matched on (t, d) to Tokyo's row at the millisecond's last frame; Tokyo's D_* columns against research's
   with chase_tokyo_2b.compare (no warm-up cut: Tokyo matched all 6,370): the 66 inputs (float32 within 1e-6, exact counted; float64 within
   1e-9 relative), lim and md; the prediction and MOVE exactly; Tokyo's D_decision on research's rows; research's in-scope actions (act,
   early, after Tokyo's warm-up) against the D line's actions (D_decision 'act' on any row); D_ms against max(t + 3.5, f + 2) with f the
   row's receipt, on every row; the matched rows' receipt against the tick's last bookTicker frame in Tokyo's tape; research's scope formula
   (below) against research's stored early. Controls: research's rows against Tokyo's at-frame columns must differ (stage 2b's gaps), and
   its_100_btc planted 0.01 BTC off in ten of research's rows must be counted ten times more.
2. The at-frame line.
   (a) Research's frozen model on Tokyo's at-frame inputs of every row (float32 inputs cast to float32): the prediction against Tokyo's
       exactly, MOVE = pred - cost(lim, md) against Tokyo's; known-bad: its_100_btc zeroed must change > 1% of the predictions.
   (b) The line re-derived on research's scores in Tokyo's row order, in Pilot::decide's order (pmmmrust crates/pf-engine/src/chase.rs):
       WarmUp and BookNotKnown as Tokyo's (decided before decide); the move acted in its direction -> MoveActed; HourEdge, DayStart,
       OffPath, InvalidBook as Tokyo's (score-independent); not finite or below the threshold -> BelowThreshold; else the move is acted, and
       act if research's forward scope holds (window end - LAST_MS - t - lead >= EARLY_MS), WindowEnding if not; against Tokyo's decision
       on every row.
   (c) Moves: the at-frame actions against research's in-scope actions, the moves paired through the matched rows' keys: kept at the same
       tick, at another tick, lost, added (with research's row at the added tick, if any).
   (d) Each differing move's deciding row (the at-frame action's row, or the last frame of research's action tick where the frame line
       did not act there): the D values with one group of at-frame values put in (a group sufficient alone), and the at-frame values with one
       group of D values put in (a group necessary), rescored; groups: trades (its / ag volumes), flows (x_*), depth (d_*, z_*), book (b_*,
       p_tau, the other mv_*), Predict (pf_*, spread_60s, book_updates_60s, lim, md).
   (e) What the at-frame inputs lack, recounted from Tokyo's recorded tape: per row the trades with E <= t received after dec_ns and by the
       first Binance frame received more than 1 s after f (chase_line.rs's row hold), their count, BTC its way and against, and the count
       received after D, against Tokyo's late_n / late_its_btc / late_ag_btc / late_after_D_n; the trades, depth20 and bookTicker frames
       with E <= t received in (f, D]; the share of the its-way 100 ms volume. Known-bad: the decision instant 1 ms later must change counts.
   (f) At-frame against D inputs by group on every row and on research's matched rows, each against its explanation: trades -> trades in
       (f, D]; flows -> trades in (f, D], a provisional frame (not its millisecond's last) or a bookTicker frame in (f, D]; depth -> depth20
       frames in (f, D] or a provisional frame; book -> a provisional frame or a bookTicker frame in (f, D]. A group differing with none of
       its explanations is counted as unexplained. Predict differs with the instant itself (g = f, not t + 3.5): counted, never unexplained.
3. Tokyo's questions. (a) The stamp gap: research's vm_path looks from the frame known at t0 (code, no count). (b) The live gap rule (on the
   path only while g <= the newest received frame's stamp + 1 s): research's rows with Tokyo's decision OffPath at the frame or at D; on
   research's forward days (11-20..26, consumed by this line) the frozen line's in-scope actions whose known Predict frame is over 1 s old
   at the decision (pf_age_ms > 1000; an upper bound: the newest received frame is at least as new as the known one). (c) pf_mid_ret_onset at
   min(onset + 3.5, g): Tokyo's rows with f < onset + 3.5 (onset = t - mv_since_ms), and those whose value differs from D's.
Outputs (write-once): chase_tokyo_replay_fixes.json (counts, no model values); OUT/chase_tokyo_2b/replay_fixes_private.json (the deciding
rows' MOVE against the threshold: model values, never in this repo).
Stages: selftest | check <dir holding tokyo-chase-stage1-20261128>.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_tokyo_replay_fixes.py <stage> ...
"""
import collections
import datetime as dt
import hashlib
import json
import pickle
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_decision_delay as CDD  # noqa: E402
import chase_forward as CF  # noqa: E402
import chase_tokyo_2b as CTB  # noqa: E402

C, TR, CT, TM, EF, E = CF.C, CF.TR, CF.CT, CF.TM, CF.EF, CF.E
RESULT = H / 'chase_tokyo_replay_fixes.json'
PRIVATE = CTB.OUT / 'replay_fixes_private.json'
DEC = Path('tokyo-chase-stage1-20261128') / 'replay-fixes' / 'decisions.csv.gz'
DEC_SHA = '33bf271ec6beea508449b2d7479814683534ca8fcc91358fc731420bccfd1cec'      # ba5aae4's MANIFEST (fb502bda...)
G_AFTER, D_LEAD, HOLD_MS, WIN_MS = TM.FEED + TM.DECIDE, 2.0, 1000., 300000
LEAD = TM.FEED + TM.DECIDE + E.NET + E.HOLD
PRE_DECIDE, SCORE_FREE = ('WarmUp', 'BookNotKnown'), ('HourEdge', 'DayStart', 'OffPath', 'InvalidBook')
PLANT_N, PLANT_ABS, STALE_MS, STAMP_TOL = 10, .01, 1000., .5
GROUPS = ('trades', 'flows', 'depth', 'book', 'predict')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def groups(kf):
    """The input groups of the attribution; they partition the 66 inputs + lim, md."""
    g = dict(trades=['mv_its_btc', 'mv_ag_btc', 'its_100_btc', 'ag_100_btc'], flows=[c for c in kf if c.startswith('x_')],
             depth=[c for c in kf if c.startswith(('d_', 'z_'))],
             predict=[c for c in kf if c.startswith('pf_') or c in ('spread_60s', 'book_updates_60s')] + ['lim', 'md'])
    used = set(sum(g.values(), []))
    g['book'] = [c for c in kf if c not in used]
    C.require(sorted(sum(g.values(), [])) == sorted(list(kf) + ['lim', 'md']), 'the groups do not partition the inputs')
    C.require(all(c.startswith(('b_', 'mv_', 'p_')) for c in g['book']), f"an unexpected input in the book group: {g['book']}")
    return {k: g[k] for k in GROUPS}


def exact(a, b, f32=False):
    """Per row: equal, or both NaN (float32 values compared as float32)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if f32:
        a, b = a.astype(np.float32), b.astype(np.float32)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def move_of(pred, d, lim, md):
    """MOVE = prediction - cost (chase_forward.cost_of; chase.rs cost)."""
    return pred - (d * (lim - md) + CT.fee(lim))


def scorer(model, kf, dtypes):
    """inputs frame (research names) -> prediction, the float32 inputs cast as research's."""
    def f(Z):
        return model.predict(pd.DataFrame({c: Z[c].to_numpy(float).astype(dtypes[c]) for c in kf}))
    return f


def early_of(t):
    """Research's forward scope at tick t (ms): the window's end from the 5-minute grid, as chase_line.rs's window."""
    we = (np.floor(np.asarray(t, float) / WIN_MS) + 1.) * WIN_MS
    return (we - E.LAST_MS - np.asarray(t, float) - LEAD) >= EF.EARLY_MS


# ------------------------------------------------------------------------------------------------------------------ the line
def derive(dec, key, d, mv, thr, early):
    """The at-frame line on research's scores, in Pilot::decide's order -> labels (see the docstring)."""
    acted, out = {}, []
    for lab, k, dd, s, ok in zip(dec, key, d, mv, early):
        if lab in PRE_DECIDE:
            out.append(lab)
        elif acted.get(dd) == k:
            out.append('MoveActed')
        elif lab in SCORE_FREE:
            out.append(lab)
        elif not (np.isfinite(s) and s >= thr):
            out.append('BelowThreshold')
        else:
            acted[dd] = k
            out.append('act' if ok else 'WindowEnding')
    return np.array(out, dtype=object)


# ------------------------------------------------------------------------------------------------------------------ the tape
def tape(raw):
    """Tokyo's recorded Binance frames as research converted them (stage 2b): every frame kept (zero prints and repeats too)."""
    rd = lambda k, cols, names: pd.concat([pd.read_csv(f, header=None, usecols=cols, names=names)  # noqa: E731
                                           for f in sorted((raw / k / 'BTCUSDT').glob('*.csv.gz'))], ignore_index=True)
    bt = rd('bookTicker', [0, 1], ['r', 'E'])
    tr = rd('trade', [0, 1, 3, 5, 6], ['r', 'E', 'id', 'q', 'm'])
    dp = rd('depth20', [0, 1], ['r', 'E'])
    return bt, tr, dp


def late_counts(rows, tr, allr, dp_, bt_, shift_ns=0):
    """Per row (recv_ns, dec_ns, t, d, D_ms): the trades with E <= t received after the decision and by the row's flush (the first Binance
    frame received > HOLD_MS after f, itself included), their BTC its way / against (summed in arrival order), those after D; and the trades,
    depth20 and bookTicker frames with E <= t received in (f, D]."""
    o = np.argsort(tr.r.to_numpy(), kind='stable')
    R, Et, Q, M = tr.r.to_numpy()[o], tr.E.to_numpy(float)[o], tr.q.to_numpy(float)[o], tr.m.to_numpy()[o] == 1
    Rms = R / 1e6
    allr = np.sort(np.asarray(allr))
    allms = allr / 1e6
    f, dec = rows.recv_ns.to_numpy(), rows.dec_ns.to_numpy() + shift_ns
    t, d, Dm = rows.t.to_numpy(float), rows.d.to_numpy(float), rows.D_ms.to_numpy(float)
    fms = f / 1e6
    ci = np.searchsorted(allms, fms + HOLD_MS, 'right')
    a_ = np.searchsorted(R, dec, 'right')
    b_ = np.searchsorted(R, np.where(ci < len(allr), allr[np.minimum(ci, len(allr) - 1)], np.iinfo(np.int64).max), 'right')
    n, its, ag, aft = np.zeros(len(rows), np.int64), np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows), np.int64)
    its100 = np.zeros(len(rows))
    for i in np.flatnonzero(b_ > a_):
        a, b = a_[i], b_[i]
        sel = np.flatnonzero(Et[a:b] <= t[i]) + a
        if not len(sel):
            continue
        w = M[sel] == (d[i] < 0)                                              # its way: a seller-taker for a down move
        n[i] = len(sel)
        its[i] = float(sum(Q[sel][w].tolist()))
        ag[i] = float(sum(Q[sel][~w].tolist()))
        aft[i] = int(np.sum(Rms[sel] > Dm[i]))
        inD = sel[Rms[sel] <= Dm[i]]
        w2 = (M[inD] == (d[i] < 0)) & (Et[inD] > t[i] - 100.)
        its100[i] = float(Q[inD][w2].sum())
    out = dict(n=n, its=its, ag=ag, after_D=aft, its100_fD=its100)
    for nm, fr in (('trades_fD', tr), ('depth_fD', dp_), ('book_fD', bt_)):
        r = fr.r.to_numpy(); oo = np.argsort(r, kind='stable'); r, e = r[oo], fr.E.to_numpy(float)[oo]
        lo, hi = np.searchsorted(r, f, 'right'), np.searchsorted(r / 1e6, Dm, 'right')
        cnt = np.zeros(len(rows), np.int64)
        for i in np.flatnonzero(hi > lo):
            cnt[i] = int(np.sum(e[lo[i]:hi[i]] <= t[i]))
        out[nm] = cnt
    return out


# ------------------------------------------------------------------------------------------------------------------ attribution
def side(mv, thr):
    return bool(np.isfinite(mv) and mv >= thr)


def attribute(rowA, rowB, G, score, thr):
    """One deciding row: A (at-frame values) and B (D values), dicts of the inputs + lim, md, d. -> the groups whose A values move B to A's
    side alone (sufficient) and whose B values move A to B's side alone (necessary), and both rows' MOVE."""
    def mv(z):
        Z = pd.DataFrame({k: [v] for k, v in z.items()})
        return float(move_of(score(Z), np.array([z['d']]), np.array([z['lim']]), np.array([z['md']]))[0])
    ma, mb = mv(rowA), mv(rowB)
    sa, sb = side(ma, thr), side(mb, thr)
    suf, nec = [], []
    for g, cols in G.items():
        zb = dict(rowB, **{c: rowA[c] for c in cols})
        za = dict(rowA, **{c: rowB[c] for c in cols})
        if side(mv(zb), thr) == sa:
            suf.append(g)
        if side(mv(za), thr) == sb:
            nec.append(g)
    return dict(sides_differ=sa != sb, frame_above=sa, D_above=sb, sufficient=suf, necessary=nec), dict(frame_move_minus_thr=ma - thr, D_move_minus_thr=mb - thr)


# ------------------------------------------------------------------------------------------------------------------ the check
def stage_check(src):
    C.require(not RESULT.exists() and not PRIVATE.exists(), f'{RESULT.name} or {PRIVATE.name} exists (write-once)')
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    hd = Path(src) / 'tokyo-chase-stage1-20261128'
    dec_fp = Path(src) / DEC
    C.require(sha(dec_fp) == DEC_SHA, "Tokyo's replay-fixes decisions are not the published file")
    T0 = json.loads(CDD.TOKYO.read_text())
    rows_fp = CDD.SRC / 'research_rows.parquet'
    C.require(sha(rows_fp) == T0['inputs']['research_rows_sha256'], "research's rows changed since chase_decision_delay's Tokyo stage")
    CF.frozen()
    R = json.loads((H / 'chase_forward_refit.json').read_text())
    kf, thr = R['inputs'], float(R['thresholds']['MOVE_top10'])
    with open(CF.FOUT / 'refit' / 'models.pkl', 'rb') as fh:
        model = pickle.load(fh)['move']
    X = pd.read_parquet(rows_fp)
    X = X.assign(t=X.t.astype(float), d=X.d.astype(float))
    f32 = [c for c in kf if str(X[c].dtype) == 'float32']
    G = groups(kf)
    score = scorer(model, kf, {c: X[c].dtype for c in kf})
    ev = [json.loads(ln) for ln in open(hd / 'hour1' / 'events.jsonl') if ln.strip()]
    st = [e for e in ev if e.get('kind') == 'start']
    C.require(len(st) == 1, "expected one start event in Tokyo's events.jsonl")
    start_ms = st[0]['recv_ns'] / 1e6
    Tk = pd.read_csv(dec_fp, float_precision='round_trip')
    Tk = Tk.assign(t=Tk.t.astype(float), d=Tk.d.astype(float), key=Tk.key.astype(str), decision=Tk.decision.astype(str),
                   D_decision=Tk.D_decision.astype(str)).reset_index(drop=True)
    Tk['row'] = np.arange(len(Tk))
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), inputs=dict(research_rows_sha256=sha(rows_fp), tokyo_decisions_sha256=DEC_SHA,
               tokyo='pmmmrust ba5aae4 / cb32854, data 21d543b replay-fixes/, build f608e51'), rows=int(len(Tk)), start_ms=start_ms,
               warm_ms=CTB.WARM_MS, groups={k: len(v) for k, v in G.items()})

    # ---- 1. the port at D
    last = Tk.sort_values(['t', 'd', 'recv_ns'], kind='stable').groupby(['t', 'd'], as_index=False).tail(1)
    ren = {f'D_{c}': c for c in kf + ['lim', 'md', 'pred', 'MOVE', 'decision']}
    TD = last[['t', 'd', 'recv_ns', 'key'] + list(ren)].rename(columns=ren)
    res, both = CTB.compare(X, TD, kf + ['lim', 'md'], f32, -np.inf)
    exact_in = {c: int((~exact(both[f'{c}_r'], both[f'{c}_t'], c in f32)).sum()) for c in kf + ['lim', 'md']}
    pred_ex = int(exact(both.pred_r, both.pred_t).sum())
    mv_ex = int(exact(both.MOVE_r, both.MOVE_t).sum())
    bt, tr, dp = tape(CDD.SRC / 'data' / 'raw')
    lastbook = bt.groupby('E').r.max()
    f_tape = both.t.map(lastbook)
    C.require(f_tape.notna().all(), "a research row without its tick's bookTicker frame in Tokyo's tape")
    C.require(np.array_equal(early_of(X.t), X.early.to_numpy(bool)), "research's scope formula does not give research's stored early")
    Dm = np.maximum(Tk.t.to_numpy(float) + G_AFTER, Tk.recv_ns.to_numpy() / 1e6 + D_LEAD)
    warm = X.t.to_numpy(float) >= start_ms + CTB.WARM_MS
    A_r = X[X.act.to_numpy(bool) & X.early.to_numpy(bool) & warm]
    A_D = Tk[Tk.D_decision == 'act']
    sr, sD = set(zip(A_r.t, A_r.d)), set(zip(A_D.t, A_D.d))
    # controls: the at-frame columns must differ; a planted change must be counted
    TA = last[['t', 'd', 'recv_ns', 'key'] + kf + ['lim', 'md', 'pred', 'MOVE', 'decision']]
    resA, _ = CTB.compare(X, TA, kf + ['lim', 'md'], f32, -np.inf)
    TP = TD.copy()
    inR = TP.set_index(['t', 'd']).index.isin(X.set_index(['t', 'd']).index)
    v = TP.its_100_btc.to_numpy(float).copy()
    ix = np.flatnonzero(inR & (v > 0) & (v < 100))[:PLANT_N]
    C.require(len(ix) == PLANT_N, 'too few rows to plant')
    v[ix] += PLANT_ABS; TP['its_100_btc'] = v
    resP, _ = CTB.compare(X, TP, kf + ['lim', 'md'], f32, -np.inf)
    planted = resP['per_input']['its_100_btc']['differ'] - res['per_input']['its_100_btc']['differ']
    C.require(resA['rows_all_inputs_equal'] < resA['matched'], "the control failed: research's rows equal Tokyo's at-frame columns")
    C.require(planted == PLANT_N, f'the control failed: {planted} planted differences counted, not {PLANT_N}')
    out['at_D'] = dict(matched=res['matched'], research_only=res['research_only'], tokyo_only_last_frames=res['tokyo_only'],
                       rows_all_inputs_equal=res['rows_all_inputs_equal'], inputs_differ={c: v['differ'] for c, v in res['per_input'].items() if v['differ']},
                       inputs_differ_exactly={c: n for c, n in exact_in.items() if n}, pred_exact=pred_ex, move_exact=mv_ex,
                       D_decision_on_research_rows=dict(collections.Counter(both['decision'].astype(str))),
                       research_in_scope_actions=int(len(sr)), D_line_actions=int(len(sD)), actions_both=int(len(sr & sD)),
                       D_ms_exact=int(exact(Dm, Tk.D_ms).sum()), dec_is_receipt=int(np.sum(Tk.dec_ns.to_numpy() == Tk.recv_ns.to_numpy())),
                       receipt_is_last_book_frame=int(np.sum(both.recv_ns.astype('int64').to_numpy() == f_tape.astype('int64').to_numpy())),
                       control_at_frame_rows_all_equal=resA['rows_all_inputs_equal'], control_at_frame_inputs_differ={c: v['differ'] for c, v in resA['per_input'].items() if v['differ']},
                       control_planted_counted=int(planted))

    # ---- 2a. the frozen model on the at-frame inputs
    pred = score(Tk)
    mv = move_of(pred, Tk.d.to_numpy(float), Tk.lim.to_numpy(float), Tk.md.to_numpy(float))
    Kb = Tk[kf].copy(); Kb['its_100_btc'] = 0.
    kb_changed = int(np.sum(~exact(score(Kb), pred)))
    C.require(kb_changed > .01 * len(Tk), f'the known-bad (its_100_btc zeroed) changed only {kb_changed} predictions')
    out['at_frame_scores'] = dict(rows=int(len(Tk)), pred_exact=int(exact(pred, Tk.pred).sum()), move_exact=int(exact(mv, Tk.MOVE).sum()),
                                  pred_max_abs_diff=float(np.nanmax(np.abs(pred - Tk.pred.to_numpy(float)))), known_bad_changed=kb_changed)

    # ---- 2b. the line re-derived
    early = early_of(Tk.t)
    der = derive(Tk.decision.to_numpy(), Tk.key.to_numpy(), Tk.d.to_numpy(float), mv, thr, early)
    dmis = collections.Counter((a, b) for a, b in zip(Tk.decision, der) if a != b)
    derD = derive(Tk.D_decision.to_numpy(), Tk.key.to_numpy(), Tk.d.to_numpy(float),
                  move_of(score(Tk[[f'D_{c}' for c in kf]].set_axis(kf, axis=1)), Tk.d.to_numpy(float), Tk.D_lim.to_numpy(float), Tk.D_md.to_numpy(float)), thr, early)
    dmisD = collections.Counter((a, b) for a, b in zip(Tk.D_decision, derD) if a != b)
    out['line_rederived'] = dict(rows_equal=int(np.sum(Tk.decision.to_numpy() == der)), mismatches=[[a, b, n] for (a, b), n in dmis.items()],
                                 actions=int(np.sum(der == 'act')), tokyo_actions=int(np.sum(Tk.decision == 'act')),
                                 D_rows_equal=int(np.sum(Tk.D_decision.to_numpy() == derD)), D_mismatches=[[a, b, n] for (a, b), n in dmisD.items()],
                                 D_actions=int(np.sum(derD == 'act')))

    # ---- 2c. moves
    kp = both.groupby('key_t').key_r.agg(lambda s: sorted(set(s)))
    kr = both.groupby('key_r').key_t.agg(lambda s: sorted(set(s)))
    C.require(all(len(v) == 1 for v in kp) and all(len(v) == 1 for v in kr), 'a move key pairs with more than one move')
    t2r = {k: v[0] for k, v in kp.items()}
    acts_f = Tk[der == 'act']
    is_last = np.zeros(len(Tk), bool); is_last[last.row.to_numpy()] = True
    Xi = X.set_index(['t', 'd'])
    moves, priv = [], []
    rmv = {k: (t, d) for k, t, d in zip(A_r.key, A_r.t, A_r.d)}
    fmv = {}
    for r_ in acts_f.itertuples():
        fmv[t2r.get(r_.key, 'tokyo:' + r_.key)] = (r_.t, r_.d, r_.row)
    for k in sorted(set(rmv) | set(fmv)):
        a_r, a_f = rmv.get(k), fmv.get(k)
        if a_r and a_f:
            cls = 'same_tick' if a_r[0] == a_f[0] else 'another_tick'
        else:
            cls = 'lost' if a_r else 'added'
        m = dict(move=k, cls=cls, research_t=a_r[0] if a_r else None, frame_t=a_f[0] if a_f else None, d=(a_r or a_f)[1])
        if cls == 'same_tick':
            moves.append(m)
            continue
        if cls == 'added' or (cls == 'another_tick' and a_f[0] < a_r[0]):
            i = a_f[2]
        else:
            i = int(last.row.to_numpy()[(last.t.to_numpy(float) == a_r[0]) & (last.d.to_numpy(float) == a_r[1])][0])
        if cls == 'added':
            hit = (a_f[0], a_f[1]) in Xi.index
            m['research_row_at_frame_tick'] = bool(hit)
            if hit:
                xr = Xi.loc[(a_f[0], a_f[1])]
                m['research_above_at_frame_tick'] = bool(np.isfinite(xr.MOVE) and xr.MOVE >= thr)
                mk_ = X.key.to_numpy() == xr.key
                m['research_move_acted_out_of_scope'] = bool(np.any(mk_ & X.act.to_numpy(bool) & ~(X.early.to_numpy(bool) & warm)))
        rw = Tk.iloc[i]
        A = {c: float(rw[c]) for c in kf + ['lim', 'md']}; A['d'] = float(rw.d)
        B = {c: float(rw[f'D_{c}']) for c in kf + ['lim', 'md']}; B['d'] = float(rw.d)
        at, pv = attribute(A, B, G, score, thr)
        m.update(deciding_row=dict(t=float(rw.t), recv_ns=int(rw.recv_ns), last_frame_of_its_ms=bool(is_last[i]), frame_decision=str(rw.decision),
                                   D_decision=str(rw.D_decision)), **at)
        moves.append(m)
        priv.append(dict(move=k, cls=cls, t=float(rw.t), d=float(rw.d), **pv))
    cnt = collections.Counter(m['cls'] for m in moves)
    out['moves'] = dict(counts=dict(cnt), research_in_scope=int(len(rmv)), at_frame=int(len(fmv)), differing=[m for m in moves if m['cls'] != 'same_tick'])

    # ---- 2e. what the at-frame inputs lack
    allr = np.concatenate([bt.r.to_numpy(), tr.r.to_numpy(), dp.r.to_numpy()])
    L = late_counts(Tk, tr, allr, dp, bt)
    Lk = late_counts(Tk, tr, allr, dp, bt, shift_ns=1_000_000)
    lt = dict(n=exact(L['n'], Tk.late_n), its=np.isclose(L['its'], Tk.late_its_btc, rtol=1e-12, atol=0.), ag=np.isclose(L['ag'], Tk.late_ag_btc, rtol=1e-12, atol=0.),
              after_D=exact(L['after_D'], Tk.late_after_D_n))
    kbl = int(np.sum(~exact(Lk['n'], Tk.late_n)))
    C.require(kbl > 0, 'the known-bad (decision 1 ms later) changed no late count')
    ib = is_last & Tk.set_index(['t', 'd']).index.isin(X.set_index(['t', 'd']).index)
    D_its = Tk.D_its_100_btc.to_numpy(float)
    out['late'] = dict(rows_equal={k: int(v.sum()) for k, v in lt.items()}, rows=int(len(Tk)), known_bad_rows_changed=kbl,
                       research_rows=dict(n=int(ib.sum()), rows_with_late_trades=int(np.sum(L['n'][ib] > 0)), late_trades=int(L['n'][ib].sum()),
                                          late_after_D=int(L['after_D'][ib].sum()),
                                          its_share_tokyo_def=float(L['its'][ib].sum() / D_its[ib].sum()),
                                          its100_share_f_to_D=float(L['its100_fD'][ib].sum() / D_its[ib].sum()),
                                          rows_trades_fD=int(np.sum(L['trades_fD'][ib] > 0)), rows_depth_fD=int(np.sum(L['depth_fD'][ib] > 0)),
                                          rows_book_fD=int(np.sum(L['book_fD'][ib] > 0))),
                       all_rows=dict(rows_with_late_trades=int(np.sum(L['n'] > 0)), rows_depth_fD=int(np.sum(L['depth_fD'] > 0)),
                                     rows_book_fD=int(np.sum(L['book_fD'] > 0)), provisional_rows=int(np.sum(~is_last))))

    # ---- 2f. at-frame against D by group, against the explanations
    prov = ~is_last
    ltr, ldp, lbk = L['trades_fD'] > 0, L['depth_fD'] > 0, L['book_fD'] > 0
    expl = dict(trades=ltr, flows=ltr | prov | lbk, depth=ldp | prov, book=prov | lbk)
    grp = {}
    for g, cols in G.items():
        dif = np.zeros(len(Tk), bool)
        for c in cols:
            dif |= ~exact(Tk[c], Tk[f'D_{c}'], c in f32)
        e_ = dict(rows_differ=int(dif.sum()), research_rows_differ=int((dif & ib).sum()))
        if g in expl:
            e_.update(unexplained=int((dif & ~expl[g]).sum()), research_rows_unexplained=int((dif & ~expl[g] & ib).sum()),
                      unexplained_examples=[dict(t=float(Tk.t.iat[i]), d=float(Tk.d.iat[i]), recv_ns=int(Tk.recv_ns.iat[i]),
                                                 inputs=[c for c in cols if not exact([Tk[c].iat[i]], [Tk[f'D_{c}'].iat[i]], c in f32)[0]])
                                            for i in np.flatnonzero(dif & ~expl[g])[:5]])
        grp[g] = e_
    pstamp_f = Tk.recv_ns.to_numpy() / 1e6 - Tk.pf_age_ms.to_numpy(float)
    pstamp_D = Tk.t.to_numpy(float) + G_AFTER - Tk.D_pf_age_ms.to_numpy(float)
    other = ~np.isclose(pstamp_f, pstamp_D, rtol=0., atol=STAMP_TOL) & np.isfinite(pstamp_f) & np.isfinite(pstamp_D)        # stamps are whole ms
    grp['predict'].update(rows_other_known_frame=int(other.sum()), research_rows_other_known_frame=int((other & ib).sum()),
                          research_rows_frame_newer=int((other & ib & (pstamp_f > pstamp_D)).sum()), research_rows_frame_older=int((other & ib & (pstamp_f < pstamp_D)).sum()))
    out['at_frame_vs_D'] = grp

    # ---- 3. Tokyo's questions
    rr = ib
    onset = Tk.t.to_numpy(float) - Tk.mv_since_ms.to_numpy(float)
    before = Tk.recv_ns.to_numpy() / 1e6 < onset + G_AFTER
    on_dif = ~exact(Tk.pf_mid_ret_onset, Tk.D_pf_mid_ret_onset)
    out['questions'] = dict(
        stamp_gap='vm_path: seg = the stamps from the frame known at t0 (j = searchsorted(kf, t0) - 1) to ws + 298 s; end = min(ws + 298 s, '
                  'the frame before the first gap > 1 s, else the last such frame, + 1 s); a point needs g >= t0 + 1 s and act + 1 s <= end',
        offpath_on_research_rows=dict(at_frame=int(np.sum((Tk.decision.to_numpy() == 'OffPath') & rr)), at_D=int(np.sum((Tk.D_decision.to_numpy() == 'OffPath') & rr))),
        forward_stale=forward_stale(),
        onset=dict(rows_frame_before_onset_plus_3p5=int(before.sum()), of_them_differ_from_D=int((before & on_dif).sum()),
                   differ_from_D_otherwise=int((~before & on_dif).sum()), research_rows_before=int((before & ib).sum()),
                   research_rows_before_differ=int((before & on_dif & ib).sum())))
    TR.dump(RESULT, out)
    PRIVATE.parent.mkdir(parents=True, exist_ok=True)
    TR.dump(PRIVATE, dict(asof=out['asof'], moves=priv))
    print(json.dumps(TR.clean(dict(at_D={k: out['at_D'][k] for k in ('matched', 'research_only', 'rows_all_inputs_equal', 'pred_exact', 'actions_both', 'D_ms_exact')},
                                   scores=out['at_frame_scores'], line={k: out['line_rederived'][k] for k in ('rows_equal', 'actions', 'tokyo_actions', 'mismatches')},
                                   moves=out['moves']['counts'], late=out['late']['rows_equal'], unexplained={g: v.get('unexplained') for g, v in grp.items()}))), flush=True)


def forward_stale():
    """The frozen line's in-scope actions on the forward days whose known Predict frame is over STALE_MS old at the decision."""
    S = pd.read_parquet(CF.FOUT / 'forward_selections.parquet', columns=['day', 't', 'd', CDD.SEL])
    S = S[S[CDD.SEL].to_numpy(bool)]
    pa = []
    for day in CF.FWD:
        s = S[S.day == day].reset_index(drop=True)
        e = pd.read_parquet(CF.FOUT / 'exits_bigmove' / f'{day}.parquet', columns=['t', 'd', 'cut_minus_act_ms'])
        C.require(np.array_equal(e.t.to_numpy(float), s.t.to_numpy(float)) and np.array_equal(e.d.to_numpy(float), s.d.to_numpy(float)),
                  "the stored big-move exits are not the selection's rows")
        s = s[e.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS]
        lb = pd.read_parquet(CF.FOUT / 'rows' / f'{day}_labels.parquet', columns=['t', 'd', 'pf_age_ms'])
        m = s.merge(lb, on=['t', 'd'], how='left', validate='1:1')
        C.require(m.pf_age_ms.notna().all(), 'an action without its pf_age_ms')
        pa.append(m.pf_age_ms.to_numpy(float))
    pa = np.concatenate(pa)
    return dict(actions=int(len(pa)), over_1s=int(np.sum(pa > STALE_MS)), share=float(np.mean(pa > STALE_MS)), pf_age_ms=CDD.quant(pa))


def selftest():
    kf = json.loads((H / 'chase_forward_refit.json').read_text())['inputs']
    G = groups(kf)
    C.require(len(kf) == 66 and sum(len(v) for v in G.values()) == 68 and G['trades'] == ['mv_its_btc', 'mv_ag_btc', 'its_100_btc', 'ag_100_btc'], f'groups {G}')
    C.require(exact([np.nan, 1.], [np.nan, 1.]).all() and not exact([1.], [1. + 1e-12])[0] and exact([0.1], [np.float32(0.1)], True)[0], 'exact')
    # the line: below, act, then the move acted; the window ending acts the move too; a score-free status kept; Tokyo's MoveActed overridden
    dec = np.array(['BelowThreshold', 'act', 'BelowThreshold', 'WarmUp', 'act', 'act', 'OffPath', 'MoveActed'], object)
    key = np.array(['1:up', '1:up', '1:up', '2:up', '2:up', '2:up', '3:down', '4:down'], object)
    d = np.array([1., 1., 1., 1., 1., 1., -1., -1.])
    mv = np.array([0., 5., 5., 5., 5., 5., 5., 0.])
    ok = np.array([True, True, True, True, False, True, True, True])
    got = derive(dec, key, d, mv, 1., ok).tolist()
    C.require(got == ['BelowThreshold', 'act', 'MoveActed', 'WarmUp', 'WindowEnding', 'MoveActed', 'OffPath', 'BelowThreshold'], f'derive {got}')
    C.require(early_of([1795860000000. + 254000.])[0] and not early_of([1795860000000. + 256000.])[0], 'early_of')
    # the late counts: a row at f = 10 ms, t = 5; trades (r ms, E, q, m): before f (in), after f its way, a zero print, after D, after the hold
    ns = lambda ms: int(ms * 1e6)  # noqa: E731
    tr = pd.DataFrame(dict(r=[ns(9), ns(10.5), ns(11), ns(13), ns(1015), ns(1020)], E=[5., 5., 4., 5., 5., 5.], id=[1, 2, 3, 4, 5, 6],
                           q=[1., 2., 0., 4., 8., 16.], m=[0, 0, 0, 1, 0, 0]))
    rows = pd.DataFrame(dict(recv_ns=[ns(10)], dec_ns=[ns(10)], t=[5.], d=[1.], D_ms=[12.]))
    bt = pd.DataFrame(dict(r=[ns(10), ns(11.5)], E=[5., 4.])); dp_ = pd.DataFrame(dict(r=[ns(10.2), ns(12.5)], E=[5., 5.]))
    allr = np.concatenate([bt.r, tr.r, dp_.r])
    L = late_counts(rows, tr, allr, dp_, bt)
    # received after f and by the flush frame (the first > 1 s after f: the trade at 1015 ms, counted): 10.5, 11, 13, 1015
    C.require(L['n'][0] == 4 and L['its'][0] == 2. + 0. + 8. and L['ag'][0] == 4. and L['after_D'][0] == 2, f'late {L}')
    C.require(L['trades_fD'][0] == 2 and L['depth_fD'][0] == 1 and L['book_fD'][0] == 1 and L['its100_fD'][0] == 2., f'(f, D] {L}')
    Lk = late_counts(rows, tr, allr, dp_, bt, shift_ns=ns(1))
    C.require(Lk['n'][0] == 2, f'the 1 ms later decision {Lk}')
    # attribution with a stub model: the score is its_100_btc; the flows do nothing
    g2 = dict(trades=['its_100_btc'], flows=['x_a'])
    stub = lambda Z: Z['its_100_btc'].to_numpy(float)  # noqa: E731
    A = dict(its_100_btc=0., x_a=1., lim=50., md=50., d=1.); B = dict(its_100_btc=5., x_a=2., lim=50., md=50., d=1.)
    at, pv = attribute(A, B, g2, stub, 2.)
    C.require(at['sides_differ'] and not at['frame_above'] and at['D_above'] and at['sufficient'] == ['trades'] and at['necessary'] == ['trades']
              and abs(pv['D_move_minus_thr'] - (5. - 1. - 2.)) < 1e-12, f'attribute {at} {pv}')
    global RESULT
    keep = RESULT
    RESULT = Path(__file__)
    try:
        stage_check('/nonexistent')
        raise AssertionError('check overwrote its result')
    except ValueError as e_:
        assert 'write-once' in str(e_), e_
    finally:
        RESULT = keep
    print('chase_tokyo_replay_fixes selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    dict(selftest=selftest, check=lambda: stage_check(a_[0]))[cmd]()
