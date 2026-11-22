"""ARRIVAL-QUEUE-DECOMPOSITION-v1 (P-2026-005; 2026-11-22, session aea5bfb9). The USER's commits aad19914 / ed7c3db5 / 13a70491 (review-only)
recorded a saved-data horizon / condition diagnostic and named ONE next priority for the development receipt: one declared saved-data
decision-to-fill x post-fill horizon decomposition with queue position as a mandatory axis (reviews/automatic/20261122_horizon_conditions/
report.md; reviews/automatic/20261122_queue_conditions/plan.md). No replay, tape load, fit, threshold, rule or new day; the stopped
POSITIVE-PRESSURE-GUARD-v1 stays stopped. Consumed Oct 24-27 development days; exploratory description, no skill or promotion claim.
POPULATION (hash-bound; label BASELINE_neutral_live; no FULL-policy state enters): every neutral BASELINE live snapshot of the audit (the
 committed states.csv.gz, sha256 STATES_SHA256; 4,840 rows, one sampled live review per generation, no-fill rows kept) with its own
 generation's fills from the pressure receipt's disabled arm (= the original ladder; parent day records checked by
 pf_ladder_pressure_lag.parent_day against the committed copies and the residual manifest's sha256s). The snapshot's gid / side / rung /
 size must be its ledger order's (refused otherwise); the states carry no price, so the same price is the same gid (an order's price is
 fixed for its life).
ARRIVAL (state_record): the snapshot's cancel effect e (the audit's: max(d + 37, submit + 95) before activation, else d + 37) opens the
 window of the generation's fills with t_f >= e (fills in [d, e) are pre_effect, before d before_decision - both counted, outside the
 window); each window fill falls in exactly one interval I0 [e, e + 250), I1 [e + 250, e + 1,000), I2 [e + 1,000, e + 5,000), I3
 [e + 5,000, the generation's end). The generation ends at G = min(its removal effect, its full-fill time, END) with its cause (ties:
 full_fill, then end, then removal); every window fill must satisfy t_f < G (t_f = G only for the completing fill; refused otherwise).
 Per interval [a, b) the follow-up status is not_at_risk (G <= a; G < a for a full fill), at_risk_full (G >= b) or, ending inside it,
 removed_in_interval / filled_out_in_interval / end_censored, with its at-risk milliseconds; the union's status is the generation's end
 cause (removed_in_interval / filled_out_in_interval / end_censored) when any interval is at risk, else not_at_risk (an end exactly on a
 boundary leaves the earlier interval at_risk_full and the union still names its cause).
HORIZONS: per fill the side-signed q x (mid(t_f + h) - p) for h in HORIZONS (0 = the fill-time mid, i.e. the execution edge; 50 / 100 ms,
 1 / 5 / 30 s; missing when the saved mid is null or t_f + h > END - never zero) and END = q x (mark - p). Per state and interval a horizon
 value is the sum over its fills, missing if any of them lacks the mark; the fill-level positive / negative parts are kept apart. An
 interval row is in the common cohort when every horizon is valid for every fill (no-fill rows included); the union's when all four
 are. The union must reproduce the audit window (fill count, quantity, END, the +50 / +100 ms / +1 s values, the pre-effect count).
QUEUE (state_record): Q = fp_queue (the client's running-minimum estimate of shares ahead), R = fp_remaining (known remaining after the
 +100 ms fill notice), Q / R when R > 0, o_q_rel (its own normalization), size, known filled = size - R, ctx_generation_age_ms (from
 submission), the activation-relative age d - act, o_depth (distance from the received best), the birth rung and pf_age (the queue
 source's observation age; saved, no staleness threshold chosen). Bands (queue_band; analysis bins, not thresholds): invalid_R (R not
 finite or <= 0), unknown (Q not finite or < 0 - never front), front (Q = 0), within_R (0 < Q <= R), beyond_R (Q > R).
PROGRESS (progress; saved receipt-time states of the same gid only, no lifetime summary, no venue state): a decision before activation is
 not_yet_born; from the generation's first-review snapshot at t_first: < d -> ok (dQ = Q - Q_first, dt), = d -> first_review_is_decision,
 > d -> future_state_refused, none -> no_first_review_saved, another order -> order_mismatch, a non-finite / negative queue ->
 unknown_queue; from the entry snapshot (the pre-activation POST state: its Q is the displayed size the order joined) the same with
 dQ_from_entry. Recent 250 ms depletion (recent): available only when the first review sits exactly at d - 250 (dQ_250); d - 250 before
 activation -> not_yet_born_at_t_minus_250; else unavailable_no_saved_state.
VENUE QUEUE (venue_quality; explanatory, never a predicate or a band): the pressure receipt's disabled-arm venue record [t, ahead, rem] of
 the gid's first live review (read after that tick) beside the first-review snapshot's client Q and R at exactly the same tick
 (time_mismatch / no_venue_entry / no_first_review_saved otherwise; no nearest-time substitution); the known-minus-venue remaining
 must equal the gid's fills in (t - 100, t] (the notice gap; notice_gap_agrees).
KNOWN REMAINING: size minus the generation's fills with t_f + 100 <= d (pf_ladder_rest.known_steps_q's convention) against fp_remaining.
VIEWS (frozen): pressure_flow (Binance adv_bn_r250 as float32, compared as the frozen on_value does: non_adverse <= 0 / positive_below_q5 /
 q5 at or above the side's last frozen edge / missing x Predict adv_ofi250: adverse > 0 / non_adverse / missing), depletion (own_d250 < 0),
 depth (o_depth), age (< 0.5 / 0.5-2 / 2-10 / >= 10 s), inventory (inv_signed: adding / reducing / flat) - the five of the USER's
 diagnostic - plus queue (queue_band), queue_pressure (queue_band x Binance band), queue_pressure_flow (x Predict band, the declared
 secondary split) and progress (the first-review status, its ok rows split by dQ_first: advanced < 0 / unchanged = 0 / increased > 0).
 Exploratory multiplicity: nine views, their band counts recorded in audit.json; sparse cells stay sparse.
TABLES (cell_table; one row per view x band x interval (and the union) x day (and all) x side (and all) x cohort (native / common)):
 opportunities, at-risk states and the follow-up status counts, states with any fill, the any-fill rate (any-fill / at-risk), fills,
 shares, partial (filled, not filled out) and filled-out states, multi-fill states, END net; per horizon h: valid states and their
 shares, net, the state-level positive / negative sums, the fill-level positive / negative sums, net per opportunity (valid states) and
 per filled share. Native = the horizon's own valid rows; common = the fixed cohort of the row type.
Usage (run_light): pf_ladder_arrival_queue.py selftest | manifest | run [out]"""
import collections
import gzip
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_pressure_lag as LG                       # noqa: E402  frozen (b19b7ca0): parent_day (the identity-checked pressure receipt)

P, M, C, pd, LB, A = LG.P, LG.M, LG.C, LG.pd, LG.LB, LG.A
VERSION = 'arrival_queue_decomposition_v1'
POPULATION = 'BASELINE_neutral_live'
STATES_REL = 'predictfun/reviews/ladder_audit_20261121/outputs/states.csv.gz'
STATES_SHA256 = 'ec2eaf7cd53628170fe485bb55a9889ae77e7f85188e592b078de6f1ec9a3043'
RESIDUAL_MANIFEST = 'predictfun/reviews/ladder_pressure_residual_20261121/manifest.json'
HORIZONS = (0, 50, 100, 1000, 5000, 30000)
INTERVALS = (('I0', 0., 250.), ('I1', 250., 1000.), ('I2', 1000., 5000.), ('I3', 5000., np.inf))
NOTICE, CANCEL, PENDING, RECENT = 100., 37., 95., 250.
QUEUE_BANDS = ('front', 'within_R', 'beyond_R', 'unknown', 'invalid_R')
FOLLOW = ('not_at_risk', 'at_risk_full', 'removed_in_interval', 'filled_out_in_interval', 'end_censored')
CAUSE_STATUS = dict(removal='removed_in_interval', full_fill='filled_out_in_interval', end='end_censored')
VIEWS = ('pressure_flow', 'depletion', 'depth', 'age', 'inventory', 'queue', 'queue_pressure', 'queue_pressure_flow', 'progress')
USER_COMMITS = dict(protocol='aad19914', results='ed7c3db5', queue_plan='13a70491')
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_arrival_queue_20261122'
OUT = A.DATA / 'ladder_arrival_queue_v1_20261122'
SOURCES = ('pf_ladder_arrival_queue.py', 'pf_ladder_arrival_queue_selftest.py', 'pf_ladder_pressure_lag.py', 'pf_ladder_pressure.py', 'pf_ladder_rest.py')
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_arrival_queue.sh')
SG = {0: 1., 1: -1.}
NAN = float('nan')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_arrival_queue.py', 'pf_ladder_arrival_queue_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in FROZEN_RECEIPT))


def fin(x):
    return x is not None and bool(np.isfinite(float(x)))


# ---------------------------------------------------------------------------------------------------------------------- pieces
def effect_of(d, o):
    """The audit's cancel effect of a decision at d for order o."""
    return max(d + CANCEL, float(o['submit']) + PENDING) if d < float(o['act']) else d + CANCEL


def generation_end(o, fills_g, end):
    """(G, cause): G = min(removal effect, full-fill time, END); ties full_fill, then end, then removal."""
    cum, full = 0., np.inf
    for f in sorted(fills_g, key=lambda f: (float(f[0]), int(f[5]))):
        cum += float(f[3])
        if cum >= float(o['size']) - LB.TOL:
            full = float(f[0]); break
    G = min(float(o['remove']), full, end)
    return G, ('full_fill' if full == G else 'end' if end == G else 'removal')


def fill_values(f, mm, mark, end):
    """One fill's horizon values (None when missing) and END value."""
    t, s, q, p = float(f[0]), int(f[2]), float(f[3]), float(f[4]); out = {}
    for j, h in enumerate(HORIZONS):
        v = mm[2 * j]
        out[h] = None if (v is None or not np.isfinite(float(v)) or t + h > end) else SG[s] * q * (float(v) - p)
    out['END'] = SG[s] * q * (mark - p); return out


def interval_of(t_f, e):
    """The arrival interval of a fill at t_f for an effect e (None before e)."""
    for name, a, b in INTERVALS:
        if e + a <= t_f < e + b:
            return name
    return None


def follow_up(e, G, cause):
    """Per interval: (status, at-risk ms)."""
    out = {}
    for name, a, b in INTERVALS:
        lo, hi = e + a, e + b
        if (G < lo) if cause == 'full_fill' else (G <= lo):
            out[name] = ('not_at_risk', 0.)
        elif G >= hi:
            out[name] = ('at_risk_full', hi - lo)
        else:
            out[name] = (CAUSE_STATUS[cause], G - lo)
    return out


def queue_band(Q, R):
    """Analysis bins (header precedence); a missing or negative Q is never the front."""
    if not fin(R) or R <= 0:
        return 'invalid_R'
    if not fin(Q) or Q < 0:
        return 'unknown'
    if Q == 0:
        return 'front'
    return 'within_R' if Q <= R else 'beyond_R'


def bn_band(x, last):
    """The frozen rule's comparison: float(float32 x) against the side's last edge (on_value: finite, > 0, q5)."""
    if not fin(x):
        return 'missing'
    x = float(np.float32(x))
    return 'non_adverse' if x <= 0 else ('q5' if x >= last else 'positive_below_q5')


def pf_band(x):
    return 'missing' if not fin(x) else ('adverse' if x > 0 else 'non_adverse')


def ident_of(row, o):
    """A saved snapshot's own order identity (the states carry no price: the gid fixes it)."""
    return dict(gid=int(row['gid']), side=int(row['side']), rung=float(row['rung']), size=float(row['size']), price=float(o['price']))


def same_order(o_a, o_b):
    return int(o_a['gid']) == int(o_b['gid']) and int(o_a['side']) == int(o_b['side']) and round(float(o_a['rung'])) == round(float(o_b['rung'])) \
        and float(o_a['size']) == float(o_b['size']) and float(o_a['price']) == float(o_b['price'])


def progress(d, Q, prior, o, ident, kind):
    """dQ from a saved earlier snapshot of the same order -> (status, Q_prior, dQ, dt)."""
    if d < float(o['act']):
        return ('not_yet_born', NAN, NAN, NAN)
    if prior is None:
        return (f'no_{kind}_saved', NAN, NAN, NAN)
    if ident is None or not same_order(o, ident):
        return ('order_mismatch', NAN, NAN, NAN)
    tp, Qp = float(prior['t']), float(prior['fp_queue'])
    if tp > d:
        return ('future_state_refused', NAN, NAN, NAN)
    if tp == d:
        return (f'{kind}_is_decision', Qp, NAN, 0.)
    if not fin(Qp) or not fin(Q) or Qp < 0 or Q < 0:
        return ('unknown_queue', Qp, NAN, d - tp)
    return ('ok', Qp, Q - Qp, d - tp)


def recent(d, Q, first, o):
    """dQ over the last 250 ms, only from the first review exactly at d - 250."""
    if d - RECENT < float(o['act']):
        return ('not_yet_born_at_t_minus_250', NAN)
    if first is not None and float(first['t']) == d - RECENT and same_order(o, ident_of(first, o)) and fin(first['fp_queue']) and fin(Q):
        return ('available', Q - float(first['fp_queue']))
    return ('unavailable_no_saved_state', NAN)


def known_remaining(o, fills_g, t):
    """size - the fills notified by t (t_f + 100 <= t)."""
    return max(0., float(o['size']) - sum(float(f[3]) for f in fills_g if float(f[0]) + NOTICE <= t))


def venue_quality(first, venue, fills_g, o):
    """The client estimate at the first review against the venue record of the same gid at exactly that tick (explanatory only)."""
    out = dict(venue_status='no_first_review_saved', venue_ahead=NAN, client_minus_venue=NAN, venue_rem=NAN, known_minus_venue_rem=NAN, unnotified_q=NAN, notice_gap_agrees=None)
    if first is None:
        return out
    if venue is None:
        return dict(out, venue_status='no_venue_entry')
    tf = float(first['t'])
    if float(venue[0]) != tf or not same_order(o, ident_of(first, o)):
        return dict(out, venue_status='time_mismatch' if float(venue[0]) != tf else 'order_mismatch')
    Qf, Rf, va, vr = float(first['fp_queue']), float(first['fp_remaining']), float(venue[1]), float(venue[2])
    un = sum(float(f[3]) for f in fills_g if tf - NOTICE < float(f[0]) <= tf)
    return dict(venue_status='available', venue_ahead=va, client_minus_venue=Qf - va if fin(Qf) else NAN, venue_rem=vr, known_minus_venue_rem=Rf - vr, unnotified_q=un,
                notice_gap_agrees=bool(abs((Rf - vr) - un) <= 1e-6 * max(1., abs(Rf))))


def state_record(z, o, fills_g, mids_g, L_, first, entry, venue, last):
    """One live snapshot: identity, queue inputs and bands, conditions, progress, venue quality, known remaining, and its four interval rows."""
    C.require(int(z['side']) == int(o['side']) and int(z['rung']) == round(float(o['rung'])) and float(z['size']) == float(o['size']) and int(z['gid']) == int(o['gid']),
              'the snapshot is not its ledger order')
    end, mark = float(L_['end']), float(L_['mark']); d = float(z['t']); e = effect_of(d, o); C.require(e == float(z['effect']), 'the effect differs from the audit\'s')
    G, cause = generation_end(o, fills_g, end); fu = follow_up(e, G, cause); Q, R = float(z['fp_queue']), float(z['fp_remaining']); size = float(o['size'])
    s = int(z['side']); bb, pb = bn_band(z['adv_bn_r250'], last[s]), pf_band(z['adv_ofi250']); qb = queue_band(Q, R); age = float(z['ctx_generation_age_ms'])
    depl, depth, inv = float(z['own_d250']), float(z['o_depth']), float(z['inv_signed'])
    pr_f = progress(d, Q, first, o, ident_of(first, o) if first is not None else None, 'first_review')
    pr_e = progress(d, Q, entry, o, ident_of(entry, o) if entry is not None else None, 'entry'); rc_ = recent(d, Q, first, o); kr = known_remaining(o, fills_g, d)
    rec = dict(population=POPULATION, day=z['day'], m=int(z['m']), side=s, gid=int(z['gid']), rung=int(round(float(o['rung']))), price=float(o['price']), t=d, effect=e, act=float(o['act']),
               gen_end=G, gen_end_cause=cause, size=size, Q=Q, R=R, Q_over_R=Q / R if (fin(Q) and fin(R) and R > 0) else NAN, o_q_rel=float(z['o_q_rel']), known_filled=size - R,
               age_ms=age, age_since_activation_ms=d - float(o['act']), o_depth=depth, pf_age=float(z['pf_age']) if fin(z['pf_age']) else NAN, queue_band=qb, bn_band=bb, pf_band=pb,
               pressure_flow=f'{bb}|PF_{pb}', depletion='missing' if not np.isfinite(depl) else ('depleting' if depl < 0 else 'not_depleting'), depth='missing' if not np.isfinite(depth) else str(depth),
               age='missing' if not np.isfinite(age) else ('<0.5s' if age < 500 else '0.5-2s' if age < 2000 else '2-10s' if age < 10000 else '>=10s'),
               inventory='missing' if not np.isfinite(inv) else ('adding' if inv > 0 else 'reducing' if inv < 0 else 'flat'), queue=qb, queue_pressure=f'{qb}|{bb}', queue_pressure_flow=f'{qb}|{bb}|PF_{pb}',
               progress=pr_f[0] if pr_f[0] != 'ok' else ('advanced' if pr_f[2] < 0 else 'unchanged' if pr_f[2] == 0 else 'increased'),
               first_status=pr_f[0], Q_first=pr_f[1], dQ_first=pr_f[2], dt_first=pr_f[3], entry_status=pr_e[0], Q_entry=pr_e[1], dQ_from_entry=pr_e[2], dt_entry=pr_e[3],
               recent_status=rc_[0], dQ_250=rc_[1], R_recomputed=kr, R_agrees=bool(fin(R) and abs(kr - R) <= 1e-6 * max(1., R)), **venue_quality(first, venue, fills_g, o))
    rows = []
    for name, a, b in INTERVALS:
        rows.append(dict(day=z['day'], m=int(z['m']), gid=int(z['gid']), interval=name, cause=cause, follow_up=fu[name][0], at_risk_ms=fu[name][1], fills=0, shares=0., filled_out=False, vEND=0.,
                         **{f'v{h}': 0. for h in HORIZONS}, **{f'ok{h}': True for h in HORIZONS}, **{f'fpos{h}': 0. for h in HORIZONS}, **{f'fneg{h}': 0. for h in HORIZONS}))
    by = {r['interval']: r for r in rows}; pre = before = 0
    for f, mm in zip(fills_g, mids_g):
        t_f = float(f[0]); iv = interval_of(t_f, e)
        if iv is None:
            pre += int(d <= t_f < e); before += int(t_f < d); continue
        C.require(t_f < G or (cause == 'full_fill' and t_f == G), 'a window fill after the generation end')
        r = by[iv]; v = fill_values(f, mm, mark, end); r['fills'] += 1; r['shares'] += float(f[3]); r['vEND'] += v['END']; r['filled_out'] |= bool(cause == 'full_fill' and t_f == G)
        for h in HORIZONS:
            if v[h] is None:
                r[f'ok{h}'] = False
            else:
                r[f'v{h}'] += v[h]; r[f'fpos{h}'] += max(v[h], 0.); r[f'fneg{h}'] += min(v[h], 0.)
    for r in rows:
        for h in HORIZONS:
            if not r[f'ok{h}']:
                r[f'v{h}'] = NAN
        r['common'] = bool(all(r[f'ok{h}'] for h in HORIZONS))
    rec['pre_effect_fills'] = pre; rec['before_decision_fills'] = before; return rec, rows


def reconcile(S_, I_, states):
    """Every live snapshot exactly once; per state the interval union reproduces the audit's window -> dict (refuses otherwise)."""
    keys = list(zip(S_.day, S_.m, S_.gid)); C.require(len(keys) == len(set(keys)) == len(states), 'the live snapshots are not each present exactly once')
    C.require(sorted(keys) == sorted(zip(states.day, states.m.astype(int), states.gid.astype(int))), 'the snapshot membership differs from the audit\'s')
    C.require(len(I_) == len(INTERVALS) * len(S_) and bool((I_.groupby(['day', 'm', 'gid']).interval.nunique() == len(INTERVALS)).all()), 'an interval row is missing or extra')
    nan_sum = lambda x: x.sum() if x.notna().all() else NAN
    U = I_.groupby(['day', 'm', 'gid']).agg(fills=('fills', 'sum'), shares=('shares', 'sum'), vEND=('vEND', 'sum'), v50=('v50', nan_sum), v100=('v100', nan_sum), v1000=('v1000', nan_sum))
    J = states.set_index(['day', 'm', 'gid']).join(U, how='left').join(S_.set_index(['day', 'm', 'gid'])[['pre_effect_fills']], how='left')
    cl = lambda a, b: (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(float(a) - float(b)) <= 1e-6 * max(1., abs(float(b))))
    bad = [k for k, r in J.iterrows() if not (int(r.fills) == int(r.win_n) and int(r.pre_effect_fills) == int(r.pre_n) and cl(r.shares, r.win_q) and cl(r.vEND, r.win_end)
                                             and cl(r.v1000, r.win_mo1s) and cl(r.v50, r.win_mo50) and cl(r.v100, r.win_mo100))]
    C.require(not bad, f'the interval union does not reproduce the audit window: {bad[:5]}')
    return dict(states=int(len(S_)), interval_rows=int(len(I_)), window_fills=int(I_.fills.sum()), window_shares=float(I_.shares.sum()))


def union_rows(X_):
    """Per state the union of its four interval rows (values missing when any interval's is; status = the end cause's when any interval is at risk)."""
    nan_sum = lambda x: x.sum() if x.notna().all() else NAN
    own = ('day', 'm', 'gid', 'interval', 'follow_up', 'at_risk_ms', 'fills', 'shares', 'filled_out', 'vEND', 'common')
    agg = {c: (c, 'first') for c in X_.columns if c not in own and not any(c == f'{p}{h}' for p in ('v', 'ok', 'fpos', 'fneg') for h in HORIZONS)}
    agg.update(fills=('fills', 'sum'), shares=('shares', 'sum'), filled_out=('filled_out', 'any'), vEND=('vEND', 'sum'), common=('common', 'all'), at_risk_ms=('at_risk_ms', 'sum'),
               _r=('_r', 'any'))
    agg.update({f'v{h}': (f'v{h}', nan_sum) for h in HORIZONS}); agg.update({f'ok{h}': (f'ok{h}', 'all') for h in HORIZONS})
    agg.update({f'fpos{h}': (f'fpos{h}', 'sum') for h in HORIZONS}); agg.update({f'fneg{h}': (f'fneg{h}', 'sum') for h in HORIZONS})
    U = X_.assign(_r=X_.follow_up != 'not_at_risk').groupby(['day', 'm', 'gid'], as_index=False, sort=True).agg(**agg)
    U['follow_up'] = np.where(U._r, U.cause.map(CAUSE_STATUS), 'not_at_risk'); U['interval'] = 'union'; return U.drop(columns='_r')


def cell_table(S_, I_):
    """The view tables (header), one row per view x band x interval (and the union) x day x side x cohort."""
    X_ = I_.merge(S_[['day', 'm', 'gid', 'side'] + list(VIEWS)], on=['day', 'm', 'gid'], how='left', validate='many_to_one'); C.require(X_.side.notna().all(), 'an interval row without its state')
    F = pd.concat([X_, union_rows(X_)], ignore_index=True); F['side'] = F.side.astype(int).astype(str); F['risk'] = F.follow_up != 'not_at_risk'
    F['filled_out'] = F.filled_out.astype(bool); F['any_fill'] = F.fills > 0; F['multi'] = F.fills > 1; F['partial'] = F.any_fill & ~F.filled_out
    new = {f'n_{st}': F.follow_up == st for st in FOLLOW}; mcols = []
    for coh in ('native', 'common'):
        for h in HORIZONS:
            ok = F[f'ok{h}'].astype(bool) if coh == 'native' else F['common'].astype(bool); v = F[f'v{h}'].where(ok, 0.).fillna(0.); k = f'{coh}__{h}__'
            cols = {k + 'valid': ok.astype(int), k + 'valid_shares': F.shares.where(ok, 0.), k + 'net': v, k + 'state_positive': v.clip(lower=0.), k + 'state_negative': v.clip(upper=0.),
                    k + 'fill_positive': F[f'fpos{h}'].where(ok, 0.), k + 'fill_negative': F[f'fneg{h}'].where(ok, 0.)}
            new.update(cols); mcols += list(cols)
    F = pd.concat([F, pd.DataFrame(new, index=F.index)], axis=1)
    FA = pd.concat([F, F.assign(day='all'), F.assign(side='all'), F.assign(day='all', side='all')], ignore_index=True)
    base = dict(opportunities=('gid', 'size'), at_risk=('risk', 'sum'), any_fill=('any_fill', 'sum'), fills=('fills', 'sum'), shares=('shares', 'sum'), partial=('partial', 'sum'),
                filled_out=('filled_out', 'sum'), multi_fill=('multi', 'sum'), end_net=('vEND', 'sum'), **{f'n_{st}': (f'n_{st}', 'sum') for st in FOLLOW})
    out = []
    for view in VIEWS:
        G = FA.groupby([view, 'interval', 'day', 'side']).agg(**base, **{c: (c, 'sum') for c in mcols}).reset_index().rename(columns={view: 'band'})
        G['any_fill_rate'] = np.where(G.at_risk > 0, G.any_fill / G.at_risk.where(G.at_risk > 0, 1), NAN)
        for coh in ('native', 'common'):
            D_ = {'view': view, **{c: G[c] for c in ['band', 'interval', 'day', 'side'] + list(base) + ['any_fill_rate']}, 'cohort': coh}
            for h in HORIZONS:
                k = f'{coh}__{h}__'; n, q = G[k + 'valid'], G[k + 'valid_shares']
                for x in ('valid', 'valid_shares', 'net', 'state_positive', 'state_negative', 'fill_positive', 'fill_negative'):
                    D_[f'h{h}_{x}'] = G[k + x]
                D_[f'h{h}_per_opportunity'] = np.where(n > 0, G[k + 'net'] / n.where(n > 0, 1), NAN); D_[f'h{h}_per_share'] = np.where(q > 0, G[k + 'net'] / q.where(q > 0, 1.), NAN)
            out.append(pd.DataFrame(D_))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------- inputs and the run
def load_states(path=None):
    p = Path(path or REPO / STATES_REL); b = p.read_bytes(); C.require(hashlib.sha256(b).hexdigest() == STATES_SHA256, 'the audit states differ from the committed ones')
    return pd.read_csv(io.BytesIO(gzip.decompress(b)), dtype={'day': str}, float_precision='round_trip')


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)'); rm = json.loads((REPO / RESIDUAL_MANIFEST).read_text())
    man = dict(version=VERSION, population=POPULATION, user_commits=USER_COMMITS, states=dict(path=STATES_REL, sha256=STATES_SHA256), parent_day_records=rm['parent_day_records'],
               days=rm['days'], edges=rm['edges'], edges_sha256=rm['edges_sha256'], horizons_ms=list(HORIZONS), intervals=[[n, a, b if np.isfinite(b) else None] for n, a, b in INTERVALS],
               queue_bands=list(QUEUE_BANDS), follow_up=list(FOLLOW), views=list(VIEWS), sources=sources(), cap_core_hours=0.10, line_cap_core_hours=2.0, asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def summary(S_, I_, T):
    """The audit.json status counts (statuses, never drops) and the multiplicity record."""
    U = union_rows(I_.merge(S_[['day', 'm', 'gid']], on=['day', 'm', 'gid'])); va = S_[S_.venue_status == 'available']; cmv = va.client_minus_venue
    return dict(population=POPULATION, queue_bands=S_.queue_band.value_counts().to_dict(), first_review=S_.first_status.value_counts().to_dict(), entry=S_.entry_status.value_counts().to_dict(),
                recent_250=S_.recent_status.value_counts().to_dict(), venue=S_.venue_status.value_counts().to_dict(), venue_notice_gap_agrees=int(va.notice_gap_agrees.astype(bool).sum()),
                venue_client_minus_venue=dict(n=int(cmv.notna().sum()), median=float(cmv.median()) if cmv.notna().any() else None, above=int((cmv > 0).sum()), equal=int((cmv == 0).sum()),
                                              below=int((cmv < 0).sum())),
                known_remaining_agree=int(S_.R_agrees.sum()), gen_end_cause=S_.gen_end_cause.value_counts().to_dict(),
                follow_up={iv: g.follow_up.value_counts().to_dict() for iv, g in I_.groupby('interval')}, union_follow_up=U.follow_up.value_counts().to_dict(),
                union_common_all_horizons=int(U.common.sum()), union_common_1_5_30=int((U.ok1000 & U.ok5000 & U.ok30000).sum()), union_no_fill=int((U.fills == 0).sum()),
                pre_effect_fills=int(S_.pre_effect_fills.sum()), before_decision_fills=int(S_.before_decision_fills.sum()),
                multiplicity=dict(views=len(VIEWS), bands={v: int(S_[v].nunique()) for v in VIEWS}, cell_rows=int(len(T))))


def run(out=None, require_frozen=True, man_path=None):
    """The decomposition (header) -> states.csv, intervals.csv, cells.csv, audit.json, write-once."""
    if require_frozen:
        C.require(frozen(), 'the arrival-queue builder / selftest / declaration / manifest / launcher are not the committed ones')
    mp = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(mp.read_text()); C.require(gm['sources'] == sources(), 'a source differs from the frozen one')
    od = Path(out or OUT); C.require(not (od / 'audit.json').exists(), 'the decomposition exists (write-once)'); od.mkdir(parents=True, exist_ok=True); t0 = time.time()
    last = {int(s): float(v[-1]) for s, v in gm['edges'].items()}; C.require(last == {s: float(v[-1]) for s, v in P.frozen_edges().items()}, 'the edges differ from the frozen ones')
    S = load_states(); live = S[S.kind == 'live'].copy(); recs, irows = [], []; byk = {k: g for k, g in S[S.kind != 'live'].groupby(['day', 'm', 'gid'])}; done = 0
    for day in gm['days']:
        _, pmk, _ = LG.parent_day(day, gm['parent_day_records'][day])
        for m, r in pmk.items():
            L_ = r['disabled']; od_ = {int(o['gid']): o for o in L_['orders']}; fl = collections.defaultdict(list); vf = L_['queue_venue']['first']
            for f, mm in zip(L_['fills'], L_['fill_mids']):
                fl[int(f[1])].append((f, mm))
            for z in live[(live.day == day) & (live.m == m)].to_dict('records'):
                g = int(z['gid']); grp = byk.get((day, m, g)); fr = grp[grp.kind == 'first'] if grp is not None else grp; en = grp[grp.kind == 'entry'] if grp is not None else grp
                nf, ne = (0 if x is None else len(x) for x in (fr, en)); C.require(nf <= 1 and ne <= 1, 'more than one first / entry snapshot of a generation')
                rec, rows = state_record(z, od_[g], [f for f, _ in fl[g]], [mm for _, mm in fl[g]], L_, fr.iloc[0].to_dict() if nf else None, en.iloc[0].to_dict() if ne else None,
                                         vf.get(str(g)), last)
                recs.append(rec); irows += rows; done += 1
    C.require(done == len(live), 'a live snapshot without its parent market')
    S_, I_ = pd.DataFrame(recs), pd.DataFrame(irows)
    rc = reconcile(S_, I_, live[['day', 'm', 'gid', 'win_n', 'win_q', 'win_end', 'win_mo1s', 'win_mo50', 'win_mo100', 'pre_n']].assign(m=live.m.astype(int), gid=live.gid.astype(int)))
    T = cell_table(S_, I_)
    for n, D_ in (('states.csv', S_), ('intervals.csv', I_), ('cells.csv', T)):
        D_.to_csv(od / n, index=False)
    C.dump(od / 'audit.json', dict(version=VERSION, asof=C.stamp(), reconcile=rc, summary=summary(S_, I_, T), elapsed_s=time.time() - t0,
                                   outputs={n: C.sha(od / n) for n in ('states.csv', 'intervals.csv', 'cells.csv')}, sources=sources(), manifest_sha256=C.sha(mp),
                                   note='exploratory saved-data description; no rule, threshold or skill claim'))
    print(json.dumps(dict(reconcile=rc, elapsed_s=round(time.time() - t0, 1)), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_arrival_queue_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1200])
    elif cmd == 'run':
        run(*(sys.argv[2:3] or [None]))
    else:
        sys.exit('usage: pf_ladder_arrival_queue.py selftest | manifest | run [out]')
