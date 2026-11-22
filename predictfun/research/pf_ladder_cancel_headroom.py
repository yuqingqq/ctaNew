"""FEASIBLE-CANCEL-HEADROOM-v1 (P-2026-005; 2026-11-22, session aea5bfb9). Review 3e84ca6d (target 85127b3d) accepted ASYNC-EXECUTION-SENSITIVITY-v1
(assumptions_do_not_rescue) and named ONE next priority: a bounded saved-data audit of the loss a timely cancellation could remove and the good
fills it would remove with it. The USER amendment 6d200317 (plan 20261122_queue_fill_tradeoff_amendment) adds, inside the same audit, the G / L
suffix classes, the bad-then-good / good-then-bad sequences, the queue fields and the 5 s / 30 s / END sensitivity at the same cut. An ORACLE
accounting diagnostic: not a deployable signal, a strategy, a speed / queue variant, an upper bound on market-making policies or a profitability
claim; no tape, replay, fit, threshold, new day or post-hoc market exclusion. Consumed Oct 24-27, the same 19 complete markets. Caps: this round
<= 0.05 core-hours, a 0.02 review reserve, the cumulative line <= 2.50 (the write-once projection gates the run); POSITIVE-PRESSURE-GUARD-v1
stays stopped.
POPULATION: every submitted generation (refused, unfilled, never reviewed included) of CURRENT_ASYNC_FAST10 (primary) and, separately and never
 pooled, FRONT_ASYNC_FAST10 (an artificial-priority reference), from ASYNC-EXECUTION-SENSITIVITY-v1's hash-bound day records (orders, fills,
 fill marks, closures, queue records, command arrays). The unchanged path must reproduce that receipt's committed eligible +1 s and END per
 market; every fill is counted once; the run refuses otherwise.
CHOICES: KEEP the recorded path, or replace ONE recorded live KEEP callback d of the generation (any reason; the counts by reason are reported)
 by a permanent PULL taking effect at d + 10 ms (the profile's live cancel, checked on every record): that generation's recorded fills with
 t >= d + 10 form the deleted suffix S(d); fills before (the request tick included) have happened. No cut before the first live review, no
 deletion of single fills (same-time fills are indivisible), no replacement, re-entry, queue reset or inventory feedback; every other order
 and the external flow stay fixed. No recorded live KEEP -> no_feasible_callback.
VALUES (per fill, native eligibility): v1 = s q (mid(t + 1 s) - p) when t <= END - 1 s and the mark exists (else unvalued); v5 / v30 likewise
 with their own marks (None = censored); vEND = s q (mark_END - p). On a suffix: G = sum max(v1, 0) (good value sacrificed by the cut),
 L = -sum min(v1, 0) (bad value avoided), net = G - L (KEEP minus cancel), counts / shares of positive, negative, exactly-zero and unvalued
 fills. Tolerance TOL = 1e-6 c.
SUFFIX CLASSES (exhaustive, in this precedence): incomplete_valuation (a suffix fill without v1); pure_observed_loss (L > TOL, G <= TOL);
 compensated_mixed (G, L > TOL, G - L > TOL); uncompensated_mixed (G, L > TOL, L - G > TOL); mixed_tie (G, L > TOL, |G - L| <= TOL);
 good_only_or_no_fill (the rest). Descriptive hindsight classes, never trading states or training labels.
ANCHORS (economic totals reconcile once per generation at each): (a) the first feasible callback; (b) the ORACLE: the cut maximizing the
 retained eligible +1 s (unvalued fills count 0); it cuts only if its value exceeds KEEP's by more than TOL, then at the earliest callback
 whose value exceeds KEEP's by more than TOL and is within TOL of the best; anchor-b classes add oracle_keep and no_feasible_callback. Without
 a feasible callback the prefix is the whole generation and the suffix empty at both anchors. END is recomputed for each anchor's cut (not
 re-optimized), late / unvalued fills apart; the 5 s / 30 s / END values of the same cut on native eligibility and on the common cohort
 (fills valued at 1 s, 5 s and 30 s).
SEQUENCES (per generation, overlapping, never summed as disjoint; on distinct fill times, a time's sign from its fills' G / L): bad_then_good
 (a good time after the first bad time) with the latest feasible callback whose effect is at or before that bad time; good_then_bad (a bad
 time after a good one; for the first such bad time and the last good time before it) with the latest callback whose effect lies after the
 good time and at or before the bad time; interleaved (>= 2 sign changes); simultaneous_good_bad (one time with G, L > TOL); the class of the
 whole generation. QUEUE (only what is recorded; unknown stays NaN, never front): submitted size, activation venue ahead (queue_venue.ahead0),
 first-review venue ahead / remaining and its time (queue_venue.first), the client's first-review known ahead (queue_known), a pressure
 pull's venue record; per anchor the order age, the client-known remaining (fills noticed at t + 100 ms) and the venue remaining at the effect
 (hindsight, never a decision input); a same-price generation on the same side within 1 s of a recorded request is a NEW generation (its own
 activation ahead). Decision panel: one row per (generation, feasible callback), a labeled panel of decision states, not independent samples.
 Casebook: per category and arm the 3 generations with the smallest sha256(arm|day|m|gid) (outcome-independent), exact fill / callback times.
READING (declared): per arm and day the oracle's retained eligible +1 s total < -TOL -> no_local_cancel_rescue, else local_cancel_headroom
 (and pooled). A fixed-path deletion accounting bound: optimistic, not an upper bound. no_local_cancel_rescue starts no cancellation-only
 variant; headroom permits only a later causal-discrimination question. Saved data can show a trade-off LINKED under the recorded
 keep-versus-permanent-cancel choice, never that a bad fill was necessary to earn front priority (re-entry is untested: unidentified).
Usage (run_light): pf_ladder_cancel_headroom.py selftest | manifest | timing | project | run [out]"""
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = 'feasible_cancel_headroom_v1'
ARMS = ('CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10')
LIVE, KEEP, FALLBACK, EFFECT_MS, NOTICE_MS, TOL, REPL_MS = 0, 2, 11, 10., 100., 1e-6, 1000.
CLASSES = ('incomplete_valuation', 'pure_observed_loss', 'compensated_mixed', 'uncompensated_mixed', 'mixed_tie', 'good_only_or_no_fill')
B_CLASSES = CLASSES + ('oracle_keep', 'no_feasible_callback')
END_CAUSES = ('refused', 'filled_out', 'recorded_cancellation', 'end_open')
AF_RECEIPT = 'predictfun/reviews/ladder_async_fast_20261122'
AF_OUT = REPO / 'data/predict_fun/research/ladder_async_fast_v1_20261122'
RECEIPT = 'predictfun/reviews/ladder_cancel_headroom_20261122'
OUT = REPO / 'data/predict_fun/research/ladder_cancel_headroom_v1_20261122'
# UNLOGGED_S: ~12 s of record-structure inspection (population counts only) run outside run_light before this design was fixed, plus two
# first-draft mutant batches (0.1 CPU min each by their surviving summaries, <= 18 s) whose per-mutant logs were deleted on 2026-11-22 before
# the rewrite: counted, not dropped. TAIL_S covers the frozen selftest and small post-run jobs.
CAP_ROUND, REVIEW_RESERVE, LINE_CEILING, UNLOGGED_S, TAIL_S = 0.05, 0.02, 2.50, 30., 15.
SOURCES = ('pf_ladder_cancel_headroom.py', 'pf_ladder_cancel_headroom_selftest.py')
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_cancel_headroom.sh', 'ch_checks.py', 'tools/async_identity.py')
TABLES = ('generations', 'generations_core', 'summary', 'classes', 'markets', 'sequences', 'panel_summary', 'cases')
OUTPUTS = tuple(f'{n}.csv' for n in TABLES) + ('panel.csv.gz',)
SG = {0: 1., 1: -1.}
PART_KEYS = ('n', 'q', 'G', 'L', 'n_pos', 'n_neg', 'n_zero', 'n_unv', 'q_pos', 'q_neg', 'q_zero', 'q_unv', 'G5', 'L5', 'n5', 'G30', 'L30', 'n30', 'Gend', 'Lend',
             'vend_unv', 'cc_n', 'cc_q', 'cc_v1', 'cc_v5', 'cc_v30', 'cc_vend', 'same_G', 'same_L', 'thr_G', 'thr_L')
CORE_COLS = ('day', 'm', 'arm', 'gid', 'side', 'rung', 'price', 'size', 'submit', 'act', 'req', 'why', 'end_cause', 'unacked_at_end', 'status', 'n_keeps', 'n_keeps_fallback',
             'first_keep', 'q_ahead_act', 'q_first_t', 'q_first_venue_ahead', 'q_first_known_ahead', 'q_pull_t', 'repl_gid', 'repl_gap', 'repl_ahead_act', 'all_n', 'all_q',
             'all_G', 'all_L', 'all_n_unv', 'v_keep', 'end_keep', 'a_cut', 'a_cls', 'a_age', 'a_rem_known', 'a_rem_venue', 'a_pre_G', 'a_pre_L', 'a_suf_G', 'a_suf_L',
             'a_suf_n', 'a_suf_n_unv', 'a_v', 'a_end', 'b_cut', 'b_cls', 'b_fallback', 'b_age', 'b_rem_known', 'b_rem_venue', 'b_pre_G', 'b_pre_L', 'b_suf_G', 'b_suf_L', 'b_suf_n',
             'b_suf_n_unv', 'b_v', 'b_end', 'seq_bad_then_good', 'seq_btg_cut', 'seq_btg_feasible', 'seq_btg_G', 'seq_btg_L', 'seq_good_then_bad', 'seq_gtb_cut',
             'seq_gtb_feasible', 'seq_gtb_G', 'seq_gtb_L', 'seq_interleaved', 'seq_simultaneous_good_bad', 'gen_class')


def need(cond, msg):
    if not cond:
        raise ValueError(msg)


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def committed(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def sources():
    return {n: sha_b((HERE / n).read_bytes()) for n in SOURCES}


def frozen():
    return bool(all(committed(f'predictfun/research/{n}') == (HERE / n).read_bytes() for n in SOURCES)
                and all(committed(f'{RECEIPT}/{n}') == (REPO / RECEIPT / n).read_bytes() for n in FROZEN_RECEIPT))


# ---------------------------------------------------------------------------------------------------------------------- the accounting
def fills_of(rec, end):
    """Per gid its fills in (t, trade index) order: arrays t, q, thr (through-price), v1 / v5 / v30 (NaN = unvalued / censored), vend."""
    mark = float(rec['mark']); by = {}
    for f, mm, c in zip(rec['fills'], rec['fill_mids'], rec['fill_class']):
        t, s, q, p = float(f[0]), int(f[2]), float(f[3]), float(f[4]); sg = SG[s]; need(c[0] in ('same_price', 'through'), f'a fill class {c[0]}')
        mv = [sg * q * (float(mm[j]) - p) if mm[j] is not None else np.nan for j in (6, 8, 10)]
        by.setdefault(int(f[1]), []).append((t, float(f[5]), q, float(c[0] == 'through'), mv[0] if t <= end - 1000. else np.nan, mv[1], mv[2], sg * q * (mark - p)))
    out = {}
    for g, rows in by.items():
        A = np.array(sorted(rows, key=lambda x: (x[0], x[1])), float).reshape(-1, 8)
        out[g] = dict(t=A[:, 0], q=A[:, 2], thr=A[:, 3] > 0, v1=A[:, 4], v5=A[:, 5], v30=A[:, 6], vend=A[:, 7])
    return out


EMPTY = dict(t=np.zeros(0), q=np.zeros(0), thr=np.zeros(0, bool), v1=np.zeros(0), v5=np.zeros(0), v30=np.zeros(0), vend=np.zeros(0))
NO_KEEPS = (np.zeros(0), np.zeros(0, bool))


def keeps_of(cm):
    """Per gid the recorded live KEEP callbacks: (ascending times, fallback flags)."""
    x = cm[(cm[:, 2] == LIVE) & (cm[:, 7] == KEEP)]; out = {}
    for g in np.unique(x[:, 3]).astype(int):
        y = x[x[:, 3] == g]; o = np.argsort(y[:, 0], kind='stable'); out[int(g)] = (y[o, 0].astype(float), y[o, 8].astype(int) == FALLBACK)
    return out


def part(F, mask):
    """The parts of the fills in mask (header): count, quantity, G / L and the sign counts / shares at +1 s, the 5 s / 30 s / END parts, the
    END value of the unvalued, the common cohort and the same- / through-price G / L."""
    v, q, ok = F['v1'][mask], F['q'][mask], ~np.isnan(F['v1'][mask]); pos, neg, zero = ok & (v > 0), ok & (v < 0), ok & (v == 0); th = F['thr'][mask]
    r = dict(n=int(mask.sum()), q=float(q.sum()), G=float(v[pos].sum()), L=float(-v[neg].sum()), n_pos=int(pos.sum()), n_neg=int(neg.sum()), n_zero=int(zero.sum()),
             n_unv=int((~ok).sum()), q_pos=float(q[pos].sum()), q_neg=float(q[neg].sum()), q_zero=float(q[zero].sum()), q_unv=float(q[~ok].sum()))
    for h in ('5', '30'):
        w = F['v' + h][mask]; k = ~np.isnan(w); r.update({f'G{h}': float(w[k & (w > 0)].sum()), f'L{h}': float(-w[k & (w < 0)].sum()), f'n{h}': int(k.sum())})
    we = F['vend'][mask]; cc = ok & ~np.isnan(F['v5'][mask]) & ~np.isnan(F['v30'][mask])
    r.update(Gend=float(we[we > 0].sum()), Lend=float(-we[we < 0].sum()), vend_unv=float(we[~ok].sum()), cc_n=int(cc.sum()), cc_q=float(q[cc].sum()), cc_v1=float(v[cc].sum()),
             cc_v5=float(F['v5'][mask][cc].sum()), cc_v30=float(F['v30'][mask][cc].sum()), cc_vend=float(we[cc].sum()), same_G=float(v[pos & ~th].sum()),
             same_L=float(-v[neg & ~th].sum()), thr_G=float(v[pos & th].sum()), thr_L=float(-v[neg & th].sum()))
    return r


def suffix_class(p):
    """The declared class of one suffix (header)."""
    if p['n_unv'] > 0:
        return 'incomplete_valuation'
    G, L = p['G'], p['L']
    if L > TOL and G <= TOL:
        return 'pure_observed_loss'
    if G > TOL and L > TOL:
        return 'compensated_mixed' if G - L > TOL else ('uncompensated_mixed' if L - G > TOL else 'mixed_tie')
    return 'good_only_or_no_fill'


def oracle(F, keeps, effect=EFFECT_MS):
    """The oracle's cut (header): None = KEEP."""
    if not len(keeps):
        return None
    cs = np.concatenate([[0.], np.cumsum(np.nan_to_num(F['v1']))]); vals = cs[np.searchsorted(F['t'], np.asarray(keeps) + effect, side='left')]; tot = cs[-1]
    if not vals.max() > tot + TOL:
        return None
    return float(np.asarray(keeps)[np.flatnonzero((vals > tot + TOL) & (vals >= vals.max() - TOL))[0]])


def suffix_mask(F, cut, effect=EFFECT_MS):
    return F['t'] >= cut + effect if cut is not None else np.zeros(len(F['t']), bool)


def sequences(F, keeps, effect=EFFECT_MS):
    """The overlapping sequence descriptors of one generation (header)."""
    ts = np.unique(F['t']); sg = []
    for tj in ts:
        w = F['v1'][F['t'] == tj]; k = ~np.isnan(w); Gj, Lj = w[k & (w > 0)].sum(), -w[k & (w < 0)].sum()
        sg.append(2 if (Gj > TOL and Lj > TOL) else (1 if Gj > TOL else (-1 if Lj > TOL else 0)))
    nz = [s for s in sg if s != 0]; changes = sum(int(a != b) for a, b in zip(nz, nz[1:])) + 2 * sum(int(s == 2) for s in nz)
    bad, good = [j for j, s in enumerate(sg) if s in (-1, 2)], [j for j, s in enumerate(sg) if s in (1, 2)]; keeps = np.asarray(keeps, float)
    r = dict(seq_simultaneous_good_bad=bool(2 in sg), seq_interleaved=bool(changes >= 2), seq_bad_then_good=False, seq_btg_t_bad=np.nan, seq_btg_cut=np.nan,
             seq_good_then_bad=False, seq_gtb_t_good=np.nan, seq_gtb_t_bad=np.nan, seq_gtb_cut=np.nan)
    if bad and any(j > bad[0] for j in good):
        tb = ts[bad[0]]; c = keeps[keeps + effect <= tb]; r.update(seq_bad_then_good=True, seq_btg_t_bad=float(tb), seq_btg_cut=float(c.max()) if len(c) else np.nan)
    jb = [j for j in bad if any(i < j for i in good)]
    if jb:
        tb = ts[jb[0]]; tg = ts[max(i for i in good if i < jb[0])]; c = keeps[(keeps + effect > tg) & (keeps + effect <= tb)]
        r.update(seq_good_then_bad=True, seq_gtb_t_good=float(tg), seq_gtb_t_bad=float(tb), seq_gtb_cut=float(c.max()) if len(c) else np.nan)
    for k_ in ('btg', 'gtb'):
        cut = r[f'seq_{k_}_cut']; p = part(F, suffix_mask(F, cut, effect)) if not math.isnan(cut) else None
        r.update({f'seq_{k_}_feasible': p is not None, f'seq_{k_}_G': p['G'] if p else np.nan, f'seq_{k_}_L': p['L'] if p else np.nan, f'seq_{k_}_class': suffix_class(p) if p else ''})
    r['gen_class'] = suffix_class(part(F, np.ones(len(F['t']), bool))); return r


def end_cause(o, closed, open_gid, end):
    """(cause, unacknowledged at END) from the venue record, cross-checked with the closures."""
    g = str(o['gid']); cl = closed.get(g, [None])[0]
    cause = 'refused' if o['refused'] else ('filled_out' if float(o['rem']) <= 1e-9 else ('recorded_cancellation' if (o['req'] is not None and float(o['remove']) <= end) else 'end_open'))
    need({'refused': cl == 'refused', 'filled_out': cl == 'filled', 'recorded_cancellation': cl in ('removed', None) and g not in open_gid, 'end_open': cl is None and g in open_gid}[cause],
         f'gid {g}: the end cause {cause} disagrees with the closure {cl}')
    return cause, bool(cause == 'recorded_cancellation' and cl is None)


def anchor(F, o, cut, kind, effect=EFFECT_MS, notice=NOTICE_MS):
    """One anchor's fields (header): its cut, class, age, the known / venue remaining, prefix and suffix parts, retained values."""
    sm = suffix_mask(F, cut, effect); pre, suf = part(F, ~sm), part(F, sm); size = float(o['size'])
    r = dict(cut=np.nan if cut is None else cut, cls=kind if kind else suffix_class(suf), age=np.nan if cut is None else cut - float(o['act']),
             rem_known=np.nan if cut is None else size - float(F['q'][F['t'] + notice <= cut].sum()), rem_venue=np.nan if cut is None else size - float(F['q'][~sm].sum()),
             v=pre['G'] - pre['L'], end=float(F['vend'][~sm].sum()), late_end=pre['vend_unv'], v5=pre['G5'] - pre['L5'], v30=pre['G30'] - pre['L30'], net=suf['G'] - suf['L'])
    r.update({f'pre_{k}': pre[k] for k in PART_KEYS}); r.update({f'suf_{k}': suf[k] for k in PART_KEYS}); return r


PANEL_COLS = ('d', 'fallback', 'age', 'rem_known', 'rem_venue', 'G', 'L', 'n_pos', 'n_neg', 'n_zero', 'n_unv')


def panel_rows(F, o, keeps, fbk, effect=EFFECT_MS, notice=NOTICE_MS):
    """The decision-state panel of one generation: per feasible callback its suffix (vectorized; a labeled panel, not samples)."""
    if not len(keeps):
        return np.zeros((0, len(PANEL_COLS)))
    v = F['v1']; ok = ~np.isnan(v); rv = lambda x: np.concatenate([np.cumsum(x[::-1])[::-1], [0.]])
    Gs, Ls = rv(np.where(ok & (v > 0), v, 0.)), rv(np.where(ok & (v < 0), -v, 0.))
    Ps, Ns, Zs, Us = (rv(x.astype(float)) for x in (ok & (v > 0), ok & (v < 0), ok & (v == 0), ~ok))
    k = np.searchsorted(F['t'], keeps + effect, side='left'); cq = np.concatenate([[0.], np.cumsum(F['q'])]); kn = np.searchsorted(F['t'] + notice, keeps, side='right')
    return np.column_stack([keeps, fbk.astype(float), keeps - float(o['act']), float(o['size']) - cq[kn], float(o['size']) - cq[k], Gs[k], Ls[k], Ps[k], Ns[k], Zs[k], Us[k]])


def panel_class(G, L, U):
    """suffix_class on panel columns (vectorized; the same precedence)."""
    out = np.full(len(G), 'good_only_or_no_fill', dtype=object); mixed = (G > TOL) & (L > TOL)
    out[mixed & (G - L > TOL)] = 'compensated_mixed'; out[mixed & (L - G > TOL)] = 'uncompensated_mixed'; out[mixed & (np.abs(G - L) <= TOL)] = 'mixed_tie'
    out[(L > TOL) & (G <= TOL)] = 'pure_observed_loss'; out[U > 0] = 'incomplete_valuation'; return out


def replacement(o, orders):
    """(submit, gid) of the first same-price generation on the same side submitted within REPL_MS of this one's recorded request (a NEW queue), or None."""
    if o['req'] is None:
        return None
    c = sorted((float(x['submit']), int(x['gid'])) for x in orders if int(x['side']) == int(o['side']) and int(x['gid']) != int(o['gid'])
               and float(x['price']) == float(o['price']) and float(o['req']) <= float(x['submit']) <= float(o['req']) + REPL_MS)
    return c[0] if c else None


def generation(day, m, arm, o, F, kp, rec, end):
    """The per-generation record (header)."""
    keeps, fbk = kp; g = str(o['gid']); qv, qk = rec['queue_venue'], rec['queue_known']; cause, unacked = end_cause(o, rec['closed'], rec['open_at_end_gid'], end)
    a0, f1, pl, k1 = qv['ahead0'].get(g), qv['first'].get(g), qv['pull'].get(g), qk.get(g) or {}
    nn = lambda x: np.nan if x is None else float(x); rp = replacement(o, rec['orders'])
    r = dict(day=day, m=m, arm=arm, gid=int(o['gid']), side=int(o['side']), rung=float(o['rung']), price=float(o['price']), size=float(o['size']), submit=float(o['submit']),
             act=float(o['act']), req=nn(o['req']), why=o['why'] or '', refused=bool(o['refused']), rem_final=float(o['rem']), end_cause=cause, unacked_at_end=unacked,
             q_ahead_act=nn(a0[1]) if a0 else np.nan, q_first_t=nn(f1[0]) if f1 else np.nan, q_first_venue_ahead=nn(f1[1]) if f1 else np.nan, q_first_venue_rem=nn(f1[2]) if f1 else np.nan,
             q_first_known_ahead=nn(k1.get('first')), q_known_n=nn(k1.get('n')), q_known_front=nn(k1.get('front')), q_known_unknown=nn(k1.get('unknown')),
             q_pull_t=nn(pl[0]) if pl else np.nan, q_pull_venue_ahead=nn(pl[1]) if pl else np.nan, q_pull_venue_rem=nn(pl[2]) if pl else np.nan,
             repl_gid=rp[1] if rp else -1, repl_gap=rp[0] - float(o['req']) if rp else np.nan,
             repl_ahead_act=nn(qv['ahead0'][str(rp[1])][1]) if rp and str(rp[1]) in qv['ahead0'] else np.nan,
             n_keeps=int(len(keeps)), n_keeps_fallback=int(fbk.sum()), first_keep=float(keeps[0]) if len(keeps) else np.nan, status='feasible' if len(keeps) else 'no_feasible_callback')
    al = part(F, np.ones(len(F['t']), bool)); r.update({f'all_{k}': al[k] for k in PART_KEYS})
    r.update(v_keep=al['G'] - al['L'], end_keep=float(F['vend'].sum()), late_end_keep=al['vend_unv'], v5_keep=al['G5'] - al['L5'], v30_keep=al['G30'] - al['L30'])
    ca = float(keeps[0]) if len(keeps) else None; cb = oracle(F, keeps)
    for X, cut, kind in (('a', ca, None if len(keeps) else 'no_feasible_callback'), ('b', cb, 'no_feasible_callback' if not len(keeps) else ('oracle_keep' if cb is None else None))):
        r.update({f'{X}_{k}': v for k, v in anchor(F, o, cut, kind).items()})
    r['b_fallback'] = bool(cb is not None and bool(fbk[np.flatnonzero(keeps == cb)[0]])); r.update(sequences(F, keeps)); return r


def timeline(F, keeps, effect=EFFECT_MS):
    """The casebook timeline: per distinct fill time its fills (t:q:v1:S/T), preceded by |d=..| for the latest callback whose effect lies in
    (previous fill time, this fill time]; then the number of callbacks whose effect follows the last fill."""
    ts = np.unique(F['t']); out, prev = [], -np.inf; keeps = np.asarray(keeps, float)
    for tj in ts:
        c = keeps[(keeps + effect > prev) & (keeps + effect <= tj)]; w = F['t'] == tj
        out.append((f'|d={c.max():.0f}|' if len(c) else '') + ','.join(f'{tj:.0f}:{q:.2f}:{"NA" if np.isnan(v) else f"{v:+.3f}"}:{"T" if th else "S"}'
                                                                  for q, v, th in zip(F['q'][w], F['v1'][w], F['thr'][w])))
        prev = tj
    return ' '.join(out) + f' |after_last={int(np.sum(keeps + effect > prev))}|'


# ---------------------------------------------------------------------------------------------------------------------- tables
def reproduce(G_, arms_ref):
    """The unchanged path = the accepted per-market eligible +1 s and END (refuses otherwise)."""
    tot = G_.groupby(['day', 'm', 'arm']).agg(v=('v_keep', 'sum'), end=('end_keep', 'sum')).reset_index()
    J = tot.merge(arms_ref[['day', 'm', 'arm', 'all_id_mo1s', 'value']], on=['day', 'm', 'arm'], how='left')
    bad = [(r.day, r.m, r.arm) for r in J.itertuples() if not (abs(r.v - r.all_id_mo1s) <= 1e-6 * max(1., abs(r.all_id_mo1s)) and abs(r.end - r.value) <= 1e-6 * max(1., abs(r.value)))]
    need(not bad and len(J) == len(arms_ref), f'the unchanged path does not reproduce the accepted totals: {bad[:4]}'); return int(len(J))


def reconcile(G_):
    """Per generation and anchor: prefix + suffix = the generation (count, quantity, G, L, unvalued, END, 5 s, 30 s); retained = KEEP - net."""
    out = {}
    for X in ('a', 'b'):
        ok = True
        for k in ('n', 'q', 'G', 'L', 'n_unv', 'Gend', 'Lend', 'G5', 'L5', 'G30', 'L30'):
            ok &= bool(np.allclose(G_[f'{X}_pre_{k}'] + G_[f'{X}_suf_{k}'], G_[f'all_{k}'], rtol=0, atol=1e-6))
        ok &= bool(np.allclose(G_[f'{X}_v'], G_['v_keep'] - G_[f'{X}_net'], rtol=0, atol=1e-6))
        ok &= bool(np.allclose(G_[f'{X}_end'] + G_[f'{X}_suf_Gend'] - G_[f'{X}_suf_Lend'], G_['end_keep'], rtol=0, atol=1e-6)); out[X] = ok
    out['b_classes'] = bool(set(G_.b_cls) <= {'pure_observed_loss', 'uncompensated_mixed', 'incomplete_valuation', 'oracle_keep', 'no_feasible_callback'})
    out['a_classes'] = bool(set(G_.a_cls) <= set(CLASSES) | {'no_feasible_callback'}); out['oracle_ge_keep'] = bool((G_.b_v >= G_.v_keep - 1e-9).all())
    need(all(out.values()), f'a reconciliation failed: {out}'); return out


SUMS = ('v_keep', 'a_v', 'b_v', 'end_keep', 'a_end', 'b_end', 'late_end_keep', 'a_late_end', 'b_late_end', 'v5_keep', 'a_v5', 'b_v5', 'v30_keep', 'a_v30', 'b_v30', 'all_G', 'all_L',
        'all_n', 'all_q', 'all_n_unv', 'a_pre_G', 'a_pre_L', 'a_suf_G', 'a_suf_L', 'b_pre_G', 'b_pre_L', 'b_suf_G', 'b_suf_L', 'b_suf_q_pos', 'b_suf_q_neg', 'b_suf_q', 'b_suf_n',
        'a_suf_same_G', 'a_suf_same_L', 'a_suf_thr_G', 'a_suf_thr_L', 'b_suf_same_G', 'b_suf_same_L', 'b_suf_thr_G', 'b_suf_thr_L', 'all_same_G', 'all_same_L', 'all_thr_G', 'all_thr_L',
        'b_suf_cc_n', 'b_suf_cc_v1', 'b_suf_cc_v5', 'b_suf_cc_v30', 'b_suf_cc_vend', 'a_suf_cc_n', 'a_suf_cc_v1', 'a_suf_cc_v5', 'a_suf_cc_v30', 'a_suf_cc_vend',
        'b_suf_Gend', 'b_suf_Lend', 'b_suf_G5', 'b_suf_L5', 'b_suf_G30', 'b_suf_L30', 'b_suf_n_unv', 'b_suf_vend_unv')


def sign(x):
    return int(x > TOL) - int(x < -TOL)


def group_row(g, d):
    """One summary row (header): totals, statuses, end causes, the anchors' class G / L and loss shares, concentration, the horizon signs, the reading."""
    d.update(generations=int(len(g)), feasible=int((g.status == 'feasible').sum()), no_feasible_callback=int((g.status == 'no_feasible_callback').sum()),
             with_fills=int((g.all_n > 0).sum()), oracle_cuts=int(g.b_cut.notna().sum()), b_fallback_cuts=int(g.b_fallback.sum()))
    d.update({f'end_{c}': int((g.end_cause == c).sum()) for c in END_CAUSES}); d.update({k: float(g[k].sum()) for k in SUMS})
    d.update(headroom_oracle=d['b_v'] - d['v_keep'], headroom_first=d['a_v'] - d['v_keep'])
    for X, cl in (('a', CLASSES + ('no_feasible_callback',)), ('b', B_CLASSES)):
        for c in cl:
            h = g[g[f'{X}_cls'] == c]; d.update({f'{X}_{c}_n': int(len(h)), f'{X}_{c}_G': float(h[f'{X}_suf_G'].sum()), f'{X}_{c}_L': float(h[f'{X}_suf_L'].sum())})
            d[f'{X}_{c}_Lshare'] = d[f'{X}_{c}_L'] / d['all_L'] if d['all_L'] > 0 else np.nan
    for X in ('a', 'b'):
        for h in ('1', '5', '30', 'end'):
            d[f'{X}_cc_sign_{h}'] = sign(d[f'{X}_suf_cc_v{h}'])
            d[f'{X}_loss_recovers_by_{h}'] = bool(h != '1' and d[f'{X}_suf_cc_v1'] < -TOL and d[f'{X}_suf_cc_v{h}'] > TOL)
    byM = g.groupby(['day', 'm']).b_suf_L.sum().sort_values(ascending=False); byP = g[g.a_cls == 'pure_observed_loss'].groupby(['day', 'm']).a_suf_L.sum().sort_values(ascending=False)
    d['b_removed_top3_market_share'] = float(byM.head(3).sum() / byM.sum()) if byM.sum() > TOL else np.nan
    d['a_pure_top3_market_share'] = float(byP.head(3).sum() / byP.sum()) if len(byP) and byP.sum() > TOL else np.nan
    d['reading'] = 'no_local_cancel_rescue' if d['b_v'] < -TOL else 'local_cancel_headroom'; return d


def levels(G_, fn, keys=(('arm', 'day', 'side'), ('arm', 'day'), ('arm', 'side'), ('arm',))):
    rows = []
    for ks in keys:
        for k, g in G_.groupby(list(ks)):
            k = k if isinstance(k, tuple) else (k,); d = dict(zip(ks, k)); d.setdefault('day', 'all'); d.setdefault('side', 'all'); d['side'] = str(d['side']); rows.append(fn(g, d))
    return pd.DataFrame(rows)


def summarize(G_):
    """summary.csv and the declared readings."""
    T = levels(G_, group_row); rd = {}
    for arm in sorted(G_.arm.unique()):
        by = {r.day: r.reading for r in T[(T.arm == arm) & (T.day != 'all') & (T.side == 'all')].itertuples()}
        rd[arm] = dict(by_day=by, pooled=T[(T.arm == arm) & (T.day == 'all') & (T.side == 'all')].iloc[0].reading, days_no_rescue=int(sum(v == 'no_local_cancel_rescue' for v in by.values())))
    return T, rd


def class_table(G_):
    """classes.csv: per arm x day x side x anchor x class the suffix parts, prefix G / L, end causes and the median order age (long format)."""
    rows = []
    for ks in (('arm', 'day', 'side'), ('arm', 'day'), ('arm', 'side'), ('arm',)):
        for X in ('a', 'b'):
            for k, g in G_.groupby(list(ks) + [f'{X}_cls']):
                d = dict(zip(ks, k[:-1])); d.setdefault('day', 'all'); d.setdefault('side', 'all'); d.update(side=str(d['side']), anchor=X, cls=k[-1], generations=int(len(g)))
                d.update({k_: float(g[f'{X}_suf_{k_}'].sum()) for k_ in PART_KEYS}); d.update({f'pre_{k_}': float(g[f'{X}_pre_{k_}'].sum()) for k_ in ('n', 'G', 'L')})
                d.update({f'end_{c}': int((g.end_cause == c).sum()) for c in END_CAUSES}); d['age_median'] = float(g[f'{X}_age'].median()) if g[f'{X}_age'].notna().any() else np.nan
                rows.append(d)
    return pd.DataFrame(rows)


def market_table(G_):
    return G_.groupby(['arm', 'day', 'm']).agg(generations=('gid', 'size'), oracle_cuts=('b_cut', 'count'), v_keep=('v_keep', 'sum'), a_v=('a_v', 'sum'), b_v=('b_v', 'sum'),
                                              end_keep=('end_keep', 'sum'), b_end=('b_end', 'sum'), b_suf_L=('b_suf_L', 'sum'), b_suf_G=('b_suf_G', 'sum'),
                                              a_suf_L=('a_suf_L', 'sum'), a_suf_G=('a_suf_G', 'sum')).reset_index()


SEQ_FLAGS = ('seq_bad_then_good', 'seq_good_then_bad', 'seq_interleaved', 'seq_simultaneous_good_bad')
SEQ_CUT = dict(seq_bad_then_good='btg', seq_good_then_bad='gtb')


def sequence_table(G_):
    """sequences.csv: per arm x day the overlapping sequence flags (not additive across flags) and the whole-generation classes."""
    rows = []
    for (arm, day), g in list(G_.groupby(['arm', 'day'])) + [((a, 'all'), G_[G_.arm == a]) for a in sorted(G_.arm.unique())]:
        for f in SEQ_FLAGS:
            h = g[g[f]]; d = dict(arm=arm, day=day, flag=f, generations=int(len(h)))
            if f in SEQ_CUT:
                k_ = SEQ_CUT[f]; fe = h[h[f'seq_{k_}_feasible']]; net = fe[f'seq_{k_}_G'] - fe[f'seq_{k_}_L']
                d.update(feasible=int(len(fe)), G=float(fe[f'seq_{k_}_G'].sum()), L=float(fe[f'seq_{k_}_L'].sum()), compensated=int((net > TOL).sum()), uncompensated=int((net < -TOL).sum()))
                d.update({f'cls_{c}': int((fe[f'seq_{k_}_class'] == c).sum()) for c in CLASSES})
            rows.append(d)
        for c in CLASSES:
            h = g[g.gen_class == c]; rows.append(dict(arm=arm, day=day, flag=f'gen_{c}', generations=int(len(h)), G=float(h.all_G.sum()), L=float(h.all_L.sum())))
    return pd.DataFrame(rows)


def panel_summary(P_):
    """panel_summary.csv: the decision-state panel by arm x day x side x class (rows are decision states, NOT independent samples)."""
    rows = []
    for ks in (('arm', 'day', 'side'), ('arm', 'day'), ('arm',)):
        for k, g in P_.groupby(list(ks) + ['cls']):
            d = dict(zip(ks, k[:-1])); d.setdefault('day', 'all'); d.setdefault('side', 'all')
            d.update(side=str(d['side']), cls=k[-1], rows=int(len(g)), generations=int(g.groupby(['day', 'm', 'gid']).ngroups), G=float(g.G.sum()), L=float(g.L.sum()),
                     note='decision-state panel: every recorded live KEEP of a generation is a row; not independent samples'); rows.append(d)
    return pd.DataFrame(rows)


CASES = (('a_pure_observed_loss', lambda g: g.a_cls == 'pure_observed_loss'), ('a_compensated_mixed', lambda g: g.a_cls == 'compensated_mixed'),
         ('a_uncompensated_mixed', lambda g: g.a_cls == 'uncompensated_mixed'), ('a_mixed_tie', lambda g: g.a_cls == 'mixed_tie'),
         ('a_incomplete_valuation', lambda g: g.a_cls == 'incomplete_valuation'), ('bad_then_good_feasible', lambda g: g.seq_bad_then_good & g.seq_btg_feasible),
         ('bad_then_good_infeasible', lambda g: g.seq_bad_then_good & ~g.seq_btg_feasible), ('good_then_bad_feasible', lambda g: g.seq_good_then_bad & g.seq_gtb_feasible),
         ('good_then_bad_infeasible', lambda g: g.seq_good_then_bad & ~g.seq_gtb_feasible), ('interleaved', lambda g: g.seq_interleaved))


def case_key(arm, day, m, gid):
    return hashlib.sha256(f'{arm}|{day}|{int(m)}|{int(gid)}'.encode()).hexdigest()


def casebook(G_, lines):
    """cases.csv: per category and arm the 3 generations with the smallest case_key (outcome-independent), with their timelines."""
    rows = []
    for name, sel in CASES:
        for arm in sorted(G_.arm.unique()):
            g = G_[(G_.arm == arm) & sel(G_)].copy(); g['key'] = [case_key(*x) for x in zip(g.arm, g.day, g.m, g.gid)]
            for r in g.sort_values('key').head(3).to_dict('records'):
                rows.append(dict(category=name, **{k: r[k] for k in ('arm', 'day', 'm', 'gid', 'side', 'size', 'act', 'end_cause', 'n_keeps', 'first_keep', 'q_ahead_act',
                                                                  'q_first_venue_ahead', 'q_first_known_ahead', 'a_cut', 'a_cls', 'a_suf_G', 'a_suf_L', 'b_cut', 'b_cls', 'b_suf_G',
                                                                  'b_suf_L', 'seq_btg_cut', 'seq_gtb_cut', 'key')}, timeline=lines[(r['arm'], r['day'], int(r['m']), int(r['gid']))]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------- inputs, manifest, projection and the run
def af_records():
    """The ASYNC receipt's day records (sha256 of each), their gzip / npz bytes checked."""
    out = {}
    for part_ in ('pilot', 'rest'):
        for p_ in sorted((AF_OUT / part_).glob('*.json')):
            rec = json.loads(p_.read_text()); need(sha_b((p_.parent / (p_.stem + '.jsonl.gz')).read_bytes()) == rec['sha256']['gz']
                                                   and sha_b((p_.parent / (p_.stem + '_arrays.npz')).read_bytes()) == rec['sha256']['npz'], f'{p_}: bytes differ from the record')
            out[f'{part_}/{p_.stem}'] = sha_b(p_.read_bytes())
    return out


def arms_ref():
    """The ASYNC receipt's committed arms copy (its raw sha256 = its committed audit)."""
    ab = committed(f'{AF_RECEIPT}/outputs/audit.json'); gb = committed(f'{AF_RECEIPT}/outputs/arms.csv.gz'); need(ab is not None and gb is not None, 'the ASYNC receipt copies are not committed')
    raw = gzip.decompress(gb); need(sha_b(raw) == json.loads(ab)['outputs']['arms.csv'], 'the ASYNC arms copy differs from its audit'); return pd.read_csv(io.BytesIO(raw), dtype={'day': str})


def markets(gm):
    """Yield (day, m, record line, arrays) of every ASYNC market in the manifest."""
    for key in sorted(gm['af_records']):
        part_, day = key.split('/'); z = np.load(AF_OUT / part_ / f'{day}_arrays.npz')
        for ln in gzip.decompress((AF_OUT / part_ / f'{day}.jsonl.gz').read_bytes()).decode().splitlines():
            r = json.loads(ln); yield day, int(r['m']), r, z


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); need(not path.exists(), 'the manifest exists (write-once)')
    sys.path.insert(0, str(HERE)); import pf_ladder_rest as M_; import pf_ladder_pressure as P_     # noqa: E401  the frozen encodings, checked once
    need(M_.CMD_KINDS.index('live') == LIVE and M_.CMD_ACTIONS.index('KEEP') == KEEP and P_.REASONS.index('fallback') == FALLBACK, 'the command encoding differs from the declared one')
    am = json.loads((REPO / AF_RECEIPT / 'manifest.json').read_text())
    man = dict(version=VERSION, review='3e84ca6d', amendment='6d200317', arms=list(ARMS), effect_ms=EFFECT_MS, notice_ms=NOTICE_MS, tol=TOL, repl_ms=REPL_MS, classes=list(CLASSES),
               b_classes=list(B_CLASSES), end_causes=list(END_CAUSES), encoding=dict(live=LIVE, keep=KEEP, fallback=FALLBACK, reasons=list(P_.REASONS)), af_records=af_records(),
               af_manifest_sha256=sha_b((REPO / AF_RECEIPT / 'manifest.json').read_bytes()), days=am['days'], complete=am['complete'], sources=sources(),
               line_receipts=am['line_receipts'] + ['ladder_async_fast_20261122'], line_reviews=am['line_reviews'] + ['20261122_async_fast_review', '20261122_queue_fill_tradeoff_amendment'],
               caps=dict(round_core_hours=CAP_ROUND, review_reserve=REVIEW_RESERVE, line_ceiling=LINE_CEILING, unlogged_inspection_s=UNLOGGED_S, tail_s=TAIL_S),
               asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def cpu_of(globs):
    """(core-hours, peak RSS MiB) of every run_light line in the logs matched by globs."""
    import glob as G
    import re
    rl = re.compile(r'^\S+ tag=\S+ rc=\d+ peak_rss_mib=(\d+) wall_s=[\d.]+ cpu_user_s=([\d.]+)'); tot, peak = 0., 0
    for p in sorted({x for gb in globs for x in G.glob(gb, recursive=True)}):
        for ln in Path(p).read_text(errors='replace').splitlines():
            mm = rl.match(ln)
            if mm:
                tot += float(mm.group(2)); peak = max(peak, int(mm.group(1)))
    return tot / 3600., peak


def ledger(gm):
    """(this round's logged core-hours + the unlogged-inspection allowance, the line before this round, this round's peak)."""
    own, peak = cpu_of([str(REPO / RECEIPT / 'logs' / '**' / '*.log')])
    line, _ = cpu_of([str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log') for n in gm['line_receipts']])
    rev, _ = cpu_of([str(REPO / 'predictfun/reviews/automatic' / n / '*.log') for n in gm['line_reviews']])
    return own + gm['caps']['unlogged_inspection_s'] / 3600., line + rev, peak


def timing():
    """The projection's input: parse every ASYNC record and build its fill tables and callbacks (no value, class or table is computed or printed)."""
    gm = json.loads((REPO / RECEIPT / 'manifest.json').read_text()); n = 0
    for day, m, r, z in markets(gm):
        for arm in ARMS:
            n += len(fills_of(r[arm], float(r['path_info']['end_rel']))); keeps_of(z[f'{m}__{arm}__commands'])
    print(json.dumps(dict(generations_with_fills=n)))


def project(path=None):
    """The write-once projection: this round's spent + the run, checker and identity jobs at 3 + 3 + 1 times the timing job + the tail <= the
    round cap; the line + that + the review reserve <= the ceiling; the peak below the 1,536 MiB checkpoint."""
    path = Path(path or REPO / RECEIPT / 'logs' / 'projection.json'); need(not path.exists(), 'the projection exists (write-once)')
    gm = json.loads((REPO / RECEIPT / 'manifest.json').read_text()); own, line, peak = ledger(gm); tj, _ = cpu_of([str(REPO / RECEIPT / 'logs' / 'ch_timing.log')])
    need(tj > 0, 'the timing job has not run'); proj = own + 7. * tj + gm['caps']['tail_s'] / 3600.; cp = gm['caps']
    res = dict(spent_round=own, timing_core_hours=tj, projected_round=proj, line_before=line, projected_line=line + proj + cp['review_reserve'], peak_mib=peak,
               fits=bool(proj <= cp['round_core_hours'] and line + proj + cp['review_reserve'] <= cp['line_ceiling'] and peak < 1536))
    path.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1)); return res


def run(out=None, require_frozen=True, man_path=None, proj_path=None):
    """The audit (header) -> OUTPUTS and audit.json in out, write-once."""
    if require_frozen:
        need(frozen(), 'the headroom builder / selftest / declaration / manifest / launcher / checker / identity tool are not the committed ones')
    mp = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(mp.read_text()); need(gm['sources'] == sources(), 'a source differs from the frozen one')
    pp = Path(proj_path or REPO / RECEIPT / 'logs' / 'projection.json'); need(pp.exists() and json.loads(pp.read_text()).get('fits') is True, 'the projection is missing or does not fit')
    od = Path(out or OUT); need(not (od / 'audit.json').exists(), 'the audit exists (write-once)'); od.mkdir(parents=True, exist_ok=True); t0 = time.time()
    need(af_records() == gm['af_records'], 'an ASYNC record differs from the manifest'); rows, pan, meta, lines, seen, nf = [], [], [], {}, set(), 0
    for day, m, r, z in markets(gm):
        end = float(r['path_info']['end_rel']); need((day, m) not in seen, f'{day} {m}: a market twice'); seen.add((day, m))
        for arm in ARMS:
            rec = r[arm]; need(float(rec['profile']['cancel']) == EFFECT_MS and float(rec['profile']['fill_notice']) == NOTICE_MS, f'{day} {m} {arm}: the profile differs')
            fl = fills_of(rec, end); kp = keeps_of(z[f'{m}__{arm}__commands']); gids = {int(o['gid']) for o in rec['orders']}
            need(set(fl) <= gids and set(kp) <= gids and sum(len(F['t']) for F in fl.values()) == len(rec['fills']), f'{day} {m} {arm}: fills / callbacks outside the orders')
            nf += len(rec['fills'])
            for o in rec['orders']:
                g = int(o['gid']); F = fl.get(g, EMPTY); k_ = kp.get(g, NO_KEEPS)
                rows.append(generation(day, m, arm, o, F, k_, rec, end)); lines[(arm, day, m, g)] = timeline(F, k_[0])
                pr = panel_rows(F, o, *k_)
                if len(pr):
                    pan.append(pr); meta.append((arm, day, m, g, int(o['side']), len(pr)))
    need(sorted(seen) == sorted((d, int(m)) for d in gm['days'] for m in gm['complete'][d]), 'the markets are not the 19 complete ones')
    G_ = pd.DataFrame(rows); n_rep = reproduce(G_, arms_ref()[lambda x: x.arm.isin(ARMS)]); rc = reconcile(G_); need(int(G_.all_n.sum()) == nf, 'a fill counted twice or dropped')
    P_ = pd.DataFrame(np.concatenate(pan), columns=PANEL_COLS); rep_ = lambda j, dt: np.repeat(np.array([x[j] for x in meta], dtype=dt), [x[5] for x in meta])
    P_ = P_.assign(arm=rep_(0, object), day=rep_(1, object), m=rep_(2, np.int64), gid=rep_(3, np.int64), side=rep_(4, np.int64))
    P_['cls'] = panel_class(P_.G.values, P_.L.values, P_.n_unv.values)
    need(len(P_) == int(G_.n_keeps.sum()), 'the panel rows differ from the callbacks'); T, rd = summarize(G_)
    tabs = dict(generations=G_, generations_core=G_[list(CORE_COLS)], summary=T, classes=class_table(G_), markets=market_table(G_), sequences=sequence_table(G_),
                panel_summary=panel_summary(P_), cases=casebook(G_, lines))
    for n in TABLES:
        tabs[n].to_csv(od / f'{n}.csv', index=False)
    with gzip.open(od / 'panel.csv.gz', 'wt', compresslevel=1) as fh:
        P_.to_csv(fh, index=False)
    (od / 'audit.json').write_text(json.dumps(dict(version=VERSION, asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), reproduced_markets=n_rep, generations=int(len(G_)), fills=nf,
                                                   panel_rows=int(len(P_)), reconciliation=rc, reading=rd, keep_reasons=dict(fallback=int(G_.n_keeps_fallback.sum()), all=int(G_.n_keeps.sum())),
                                                   outputs={n: sha_b((od / n).read_bytes()) for n in OUTPUTS}, sources=sources(), manifest_sha256=sha_b(mp.read_bytes()),
                                                   projection_sha256=sha_b(pp.read_bytes()), elapsed_s=time.time() - t0,
                                                   note='an ORACLE fixed-path deletion accounting diagnostic; not a signal, strategy, upper bound or profitability claim'), indent=1, default=str) + '\n')
    print(json.dumps(dict(generations=len(G_), panel_rows=len(P_), reproduced_markets=n_rep, reading=rd), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_cancel_headroom_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1500])
    elif cmd == 'timing':
        timing()
    elif cmd == 'project':
        project()
    elif cmd == 'run':
        run(*(sys.argv[2:3] or [None]))
    else:
        sys.exit('usage: pf_ladder_cancel_headroom.py selftest | manifest | timing | project | run [out]')
