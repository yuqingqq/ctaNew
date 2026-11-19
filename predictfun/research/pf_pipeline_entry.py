"""ENTRY WITH A FROZEN DOWNSTREAM CANCELLATION RULE - POST VERSUS NO ENTRY ON ONE COMMON TARGET (2026-11-19, session aea5bfb9). Review
1adf6a33's one next priority: "Evaluate entry using a frozen downstream cancellation policy and a common economic target ... keep the current
size, price, thresholds and model fixed for this diagnostic. Compare POST versus no entry under MODEL-J and PULL-ALL, decompose loss before
the earliest effective pull versus continuation value, and assess quantity/movement calibration on the orders actually selected. Include
later-pull versus never-pull attribution and bid/ask splits." USER: "Can you check the commit and proceed the next priority". Declared before
any output of this module is read. Consumed days Oct 26 and Oct 27, both already seen (development evidence, not validation); nothing fitted.

TARGET: one opportunity = one committed state of the keep / pull run with a path (a 30-sh JOIN at the best, END = +5 s, one per window-side
 every 5 s). Its value is the local mark at END of the fills it keeps under the downstream cancellation rule, in cents, in each of the bank's
 19 scenarios. The target is the same for every entry rule and every cancellation rule; a post-only refusal (in that scenario) is worth 0;
 NO-ENTRY is 0.
ENTRY RULES (frozen; no threshold fitted):
 POST-ALL    every opportunity (the simulator's schedule, which the keep / pull tests ran)
 MODEL-POST  post iff the benchmark v3's J POST forecast is coherent (J_ok) and its mapped value is above 0: the contract's decide_one on a
             POST target with zero economic terms and no cap; a refused forecast waits (the contract's fallback)
 NO-ENTRY    none (0)
CANCELLATION RULES (downstream, frozen): KEEP (structural exits only: the POST label's continuation), MODEL-J (pf_pipeline_policy v3's rule,
 its triggers from the same predictions), PULL-ALL (pull at the first live step, +100 ms).
VALUES: the bank's own valuation (pf_keep_pull_scenarios.outcome on the opportunities an entry rule posts, with MODEL-J's triggers):
 V(R, KEEP) = KEEP, V(R, MODEL-J) = KEEP + its improvement, V(R, PULL-ALL) = KEEP + PULL-ALL's improvement. A second path, per opportunity
 (each fill's value and the removal rule), gives the null, the splits and the attribution, and must reproduce the bank's totals (Q0b).
MATCHED NULL (exact): the expected value of posting the same number of opportunities at random in each (day, market window, side, minute)
 cell = the sum over cells of k / n x the cell's value. E_entry(C) = V(MODEL-POST, C) - null(C).
READINGS (computed; across the 19 tested scenarios unless stated):
 N1 V(MODEL-POST, MODEL-J) vs NO-ENTRY: ABOVE- / BELOW-NO-ENTRY- / NOT-STABLE-IN-TESTED-SCENARIOS
 N2 V(MODEL-POST, PULL-ALL) vs NO-ENTRY
 N3 V(MODEL-POST, KEEP) vs NO-ENTRY (the POST label's continuation)
 N4 V(POST-ALL, MODEL-J) vs NO-ENTRY
 N5 V(MODEL-POST, MODEL-J) - V(POST-ALL, MODEL-J): ABOVE- / BELOW-POST-ALL- / NOT-STABLE-IN-TESTED-SCENARIOS
 N6 the same under PULL-ALL
 N7 E_entry under MODEL-J: STABLE-POSITIVE- / STABLE-NEGATIVE- / NOT-STABLE-IN-TESTED-SCENARIOS
 N8 E_entry under PULL-ALL
 N1-NY N1 without fills on orders < 100 ms old (FINDINGS C1)
 C2 N1's value and N7's E at M250 vs M50 (HOLDS- / SIGN-DIFFERS-AT-M250)
 C1 N1's value at B weighted by FV3's B1a precision weights by age bin (HOLDS-SIGN / SIGN-DIFFERS)
PRINTED at B, per entry rule, cancellation rule and side:
 - values by fill time after submission: before the first decision (+100 ms); from it to the first possible cancellation effect (+100 ms + L);
   and after (the continuation);
 - MODEL-J against PULL-ALL by MODEL-J's action: pulled at the first live step / pulled later / never pulled with RETAIN predictions / no
   RETAIN prediction;
 - calibration on the opportunities posted: sum E[q] / sum q, sum E[q * move] / sum q * move, the mean predicted vs realized value under
   KEEP, MODEL-J and PULL-ALL, and the prediction error split exactly into the spread-times-quantity and movement terms;
 - cancel-pending fills (C3) and the value by trade-anchor source (C12).
CONTROLS (any failure -> no output):
 Q0a POST-ALL's KEEP / MODEL-J / PULL-ALL values equal the committed pf_pipeline_policy v3 output of the day in every scenario, and that
     output was computed on the same prediction file
 Q0b the per-opportunity path reproduces the bank's values for both entry rules and the three cancellation rules in every scenario; the
     null of POST-ALL equals its value
 Q0c the fill-time windows and the attribution groups sum to their totals; MODEL-J - PULL-ALL is exactly 0 on orders pulled at the first
     live step
 Q0d entry parity on EVERY POST forecast: the contract's verdict (it accepts the rebuilt two-atom forecast and its map_value gives back
     E[q], E[q * move] and the value) equals J_ok, and the opportunities decide_one posts are exactly the ones the rule posts (rows with
     |V_J| < 1e-6 compared on the verdict only)
 Q0e provenance: the predictions match the benchmark v3 receipt; their POST rows are the export's POST rows with a path, one per state the
     bank values; at B every opportunity's KEEP value equals its exported POST label (value_m50), a refused one 0
STATUSES: every committed state is counted by status (no path: the keep / pull run's statuses; refused in a scenario: worth 0).
LIMITS: two consumed days (G = 2, point values); the POST model was trained on the neutral (KEEP) label, so this is not an aligned model;
 each opportunity is valued alone (no shared volume, inventory or re-post); local marks at END, not P&L; queue rule M only (C5); no
 order-level validation (C10); the RETAIN model's own-fill inputs are B's in every scenario.
SELFTEST (synthetic tape, no data):
 - planted POST and RETAIN predictions give the planted entry set and triggers, and every control passes;
 - every B value is recomputed by a per-fill loop;
 - the exact null equals a hand-computed two-cell example and the mean of 400 random matched selections;
 - refusals by their own control with no output: a POST prediction for a state the export lacks and a missing one (Q0e), a changed
   label (Q0e), a changed committed policy value (Q0a), the entry sign reversed and the rule acting on refused forecasts (Q0d),
   zero-quantity POST forecasts with E[q * move] -50 / +50 c flagged coherent (Q0d), the MODEL-J landing moved (Q0b), a policy output on another
   prediction file (Q0a), every executed pull put in the first-step group (Q0c);
 - the fill-time windows and the young-fill exclusion recomputed by the loop; N1 / N2 / N5 / N7 and their ranges recomputed from the CSV; the
   error split adds up; a planted predicted loss at an order's structural end never executes, and counting it is refused (Q0b, the bank's pulls).
 Mutations: pf_pipeline_entry_mutants.py.
 G1 (plan 0c3d2523, predictfun/handoffs/20261119_entry_retention_priority.md step 1; added before any data was read): the five named policies
 N0 = NO-ENTRY, A0 = POST-ALL + PULL-ALL, A1 = POST-ALL + MODEL-J, E0 = MODEL-POST + PULL-ALL, E1 = MODEL-POST + MODEL-J (references AK / EK: the
 KEEP continuation, the POST label's); one attempted generation per original opportunity, no re-entry before its common END. At B, per side
 (and both sides), from the per-opportunity path:
 - the paired per-opportunity ledger (entry_ledger_<day>.parquet): every committed state with its status, admission, values, cancel step,
   live end and cause, shares, inventory and at-best time under each cancellation rule;
 - population manifest and chronology (statuses, opportunities with a path, admitted / activated / refused / filled per policy; the
   benchmark's training days, the prediction, policy and export receipts);
 - five-policy table: total and per original opportunity at B, the range over the 19 scenarios, vs N0 and vs the same-admission PULL-ALL;
 - participation: attempts / opportunities with a path, accepted (not post-only refused) / attempts, filled / accepted, shares filled /
   shares submitted (30 per attempt);
 - quote availability over the valid market clock (5 s per opportunity with a path): live time from activation (+58 ms at B) to the full
   fill, structural landing, policy cancellation landing or END, per side and, over the bid / ask pairs of the same window and time, either
   and both sides; cancel-pending time (request to effect) separately; at-best time on the 50 ms decision grid (o_depth <= 0 at a grid point
   while live, 50 ms each; unknown depth counted; live time before the first grid point reported apart);
 - order lifetime (activation to its end): p10 / p50 / p90 and cause counts (END survivors censored, not cancellations); submission to
   cancel request and request to effect;
 - first-review reversals: MODEL-J pulls executed at the first live step / activated orders still live there, and / all executed pulls;
 - churn: posts, policy cancellations and structural cancellations per valid market hour and per 100 filled shares (NA when none filled);
 - queue preservation at the rule's KEEP decisions (reached grid rows before its pull): age, known queue ahead, known remaining, unknown
   count; at the fill: NOT COMPUTED (the kernel's queue at a fill is not stored); resets 0 (no re-entry in G1);
 - early-fill loss: the three fill-time windows plus the actual cancel-pending fills (request to landing) per policy;
 - tail: per admitted episode, P(value < 0), P(value < -30 c), the severe excess beyond 30 c, p95 / p99 loss, the mean, per original opportunity;
 - inventory: per episode the signed shares kept to END (mean |.|, peak), time-integrated inventory and reservation (share-seconds), no cap;
 - calibration in the plan's fixed bins [-inf, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, inf] c (exact 0 apart): POST forecasts vs the label
   and the value under MODEL-J / PULL-ALL; RETAIN forecasts on the neutral rows and on the rows the rule reaches, vs the label;
 - markouts: the END markout is the value per filled share; +100 ms / +1 s NOT COMPUTED (no mid is stored per fill; a tape read is the
   next module);
 - the plan's economic gate predicates, descriptive here: E1 > N0 and E1 > E0 at B, per side.
 New controls: Q0f branch identity (in every scenario, every opportunity's fills before the first decision and before the earliest
 effect are worth the same under KEEP, MODEL-J and PULL-ALL); Q0g refusal accounting (per scenario and entry rule, admitted = activated +
 post-only refused, activated = the bank's eligible); a duplicated fill is caught by Q0b (KEEP is the bank's order value). Falsifiers: a
 duplicated fill (Q0b), refused posts kept (Q0g), the MODEL-J landing at the decision (Q0f). A failed run names every failing control.
 v3.1 parity: the rebuilt E[q | fill] is set to the known remaining when it exceeds it by no more than 1e-9 relative (the benchmark clipped
 it there; E[q] / P rounds an ulp above), as pf_pipeline_policy v3.1.
Usage (run_light, PF_SLOT=claude): pf_pipeline_entry.py selftest | day <day>
"""
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_fv3_check as FV                                # noqa: E402  FINDINGS C1's B1a precision weights by age bin (M 0 / M 50)
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_scenarios as SC                       # noqa: E402
import pf_keep_pull_variants as V                         # noqa: E402
import pf_pipeline_contract as PC                         # noqa: E402  the shared contract (imported, never edited)
import pf_pipeline_policy as P                            # noqa: E402  the frozen downstream cancellation rule (v3)
import pf_queue_place_test as QP                          # noqa: E402
import pf_queue_place_ledger as QL                        # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'pipeline_entry_20261119'
MARK = 'PIPELINE ENTRY'
ENTRY, CANCEL = ('POST-ALL', 'MODEL-POST'), ('KEEP', 'MODEL-J', 'PULL-ALL')
B_NAME, M250_NAME = V.B_NAME, V.M250_NAME
CELL = ['day', 'm', 'side', 'phase']
GROUPS = ('pulled at the first live step', 'pulled later', 'never pulled, with RETAIN predictions', 'no RETAIN prediction')
WINDOWS = ('before the first decision (+100 ms)', 'first decision to its earliest effect (+100 ms + L)', 'after it (continuation)')
PRED_COLS = ['day', 'oid', 'kind', 'j', 'side', 't', 'decision_ms', 'p', 'pf_mid', 'fp_remaining', 'q_m50', 'qm_m50', 'value_m50',
             'J_P', 'J_Eq', 'J_Eqm', 'J_V', 'J_ok']
FLIP_ENTRY = RAW_ENTRY = NULL_GLOBAL = False              # planted-error switches (selftest only): the entry sign reversed; the rule acting on
LAND_SHIFT = 0.                                           # refused forecasts; an unmatched (global) null; the MODEL-J landing moved (ms);
GROUP_ANY = ACT_ANY = False                               # every executed pull in the first-live-step group; a pull at or after the order's
                                                          # structural end counted as executed;
DUP_FILL = IGNORE_REFUSED = False                         # a fill counted twice; post-only refused opportunities kept as activated
POL = {'A0': ('POST-ALL', 'PULL-ALL'), 'A1': ('POST-ALL', 'MODEL-J'), 'E0': ('MODEL-POST', 'PULL-ALL'), 'E1': ('MODEL-POST', 'MODEL-J'),
       'AK': ('POST-ALL', 'KEEP'), 'EK': ('MODEL-POST', 'KEEP')}           # N0 = NO-ENTRY (0); AK / EK: the POST label's continuation
CAUSES = ('full fill', 'structural', 'policy cancel', 'END (censored)')
EDGES = (-10., -5., -2., -1., -.5, 0., .5, 1., 2., 5., 10.)                  # the plan's fixed bins, right-closed; exact 0 apart
SLOT_MS = 5000.                                                           # the valid market clock of one opportunity
RET_COLS = ['oid', 'kind', 'j', 'status', 'o_depth', 'o_age', 'fp_queue', 'fp_remaining', 'o_qunk']


def entry_of(pred):
    """MODEL-POST: the (day, oid) of the POST forecasts the rule posts - coherent and a mapped value above 0."""
    p = pred[pred.kind == 'post']; acts = (p.J_V < 0) if FLIP_ENTRY else (p.J_V > 0)
    s = p[acts if RAW_ENTRY else (acts & p.J_ok.astype(bool))]
    return set(zip(s.day.astype(str), s.oid.astype(int)))


def entry_parity(pred, trained):
    """Q0d on every POST forecast: the contract's verdict (it accepts the rebuilt two-atom forecast and map_value gives back E[q], E[q * move]
    and the mapped value) and the opportunities decide_one posts on a POST target (lifecycle absent; wait when the verdict is no)."""
    p = pred[pred.kind == 'post']; tol = lambda x: 1e-9 * max(1., abs(x)); n = ref = rtbad = verdict_off = tiny = 0; posts, tiny_keys = set(), set()
    for r in p.itertuples():
        side = 'bid' if r.side == 0 else 'ask'; t = float(r.t); d = float(r.decision_ms); val = None; key = (str(r.day), int(r.oid))
        tg = PC.Target(f'{r.day}:{int(r.oid)}', side, 'post', d, t + KP.PLACE, t + KP.H, float(r.pf_mid), float(r.p), float(r.fp_remaining), B_NAME,
                       'frozen downstream cancellation (KEEP / MODEL-J / PULL-ALL) to the common END', 'JOIN30/END=birth+5000/queueM/structuralKeeper/scenarios-v6')
        try:
            qf = float(r.J_Eq / r.J_P) if r.J_P > 0 else 0.; mv = float(r.J_Eqm / r.J_Eq) if r.J_Eq > 0 else 0.; rem = float(r.fp_remaining)
            qf = rem if rem < qf <= rem + tol(rem) else qf                                       # v3.1: E[q | fill] clipped at the remaining, rounded above it by E[q] / P
            atoms = (PC.Atom(float(r.J_P), qf, mv), PC.Atom(1. - float(r.J_P), 0., 0.)) if 0. < r.J_P < 1. else (PC.Atom(1., qf if r.J_P >= 1 else 0., mv),)
            fc = PC.Forecast(tg, atoms, d, trained, 'learned', 'pf_pipeline_bench:J'); val = PC.map_value(tg, fc)
            dec = PC.decide_one(tg, fc, PC.ClientState(tg.order_id, side, d, 'absent', 0., 0., 0., 0., 1e9), PC.Economics(tg, 0., 0., 0., 0.)); accepted, action = True, dec.action
        except ValueError:
            ref += 1; accepted, action = False, 'wait'                                          # the contract's fallback: no usable forecast -> wait
        rt = val is not None and abs(val.quantity - r.J_Eq) <= tol(r.J_Eq) and abs(val.movement - r.J_Eqm) <= tol(r.J_Eqm) and abs(val.retained - r.J_V) <= tol(r.J_V)
        coherent = accepted and rt; rtbad += int(accepted and not rt); verdict_off += int(coherent != bool(r.J_ok)); n += 1
        if abs(r.J_V) < 1e-6:
            tiny += 1; tiny_keys.add(key)
        elif coherent and action == 'post':
            posts.add(key)
    return dict(rows=n, refused=ref, accepted_without_round_trip=rtbad, verdict_disagreements=verdict_off, verdict_only=tiny, posts=posts, tiny=tiny_keys)


def null_weights(keys, sel):
    """The exact matched null's weight of each opportunity: k / n of its (day, market window, side, minute) cell (keys: one row per opportunity
    with a path, columns CELL; sel: whether the rule posts it)."""
    sel = np.asarray(sel, bool)
    if NULL_GLOBAL:
        return np.full(len(keys), sel.mean() if len(sel) else 0.)
    g = keys[CELL].astype(str).agg('|'.join, axis=1).values; u, inv = np.unique(g, return_inverse=True)
    n = np.bincount(inv, minlength=len(u)).astype(float); k = np.bincount(inv, weights=sel.astype(float), minlength=len(u))
    return (k / n)[inv]


def per_order(O, Os, Fs, sc, fn):
    """The second path: per opportunity eligible in one scenario, the value it keeps under KEEP / MODEL-J / PULL-ALL in total, by fill-time window,
    by age bin and (MODEL-J) by trade-anchor source; its MODEL-J step and whether that pull executes."""
    Os = Os.rename(columns={c: c + '_s' for c in ('refused', 'v', 'q', 't_full', 'land')})
    M = O.merge(Os, on=['day', 'oid'], how='inner', validate='one_to_one'); E = (M if IGNORE_REFUSED else M[~M.refused_s.astype(bool)]).reset_index(drop=True)
    T = np.asarray(fn(E), int); endj = SC.endj_n(E.t_full_s.values, E.g_struct.values, sc['N']); act = (T >= KP.J0) & (T <= KP.NSTEP) & ((T < endj) | ACT_ANY)
    ix = pd.Series(np.arange(len(E)), index=pd.MultiIndex.from_arrays([E.day.values, E.oid.values]))
    ri = ix.reindex(pd.MultiIndex.from_arrays([Fs.day.values, Fs.oid.values])).values; C.require(not np.isnan(ri).any(), 'Q0b: a fill of an opportunity outside the eligible set')
    ri = ri.astype(int); n = len(E); sg = np.where(E.side.values == 0, 1., -1.); val = Fs.q.values * sg[ri] * (E.mark.values[ri] - E.p.values[ri]); t = Fs.t.values.astype(float)
    lf = np.where(SC.slow_order(E.oid.values[ri], sc['X'][0]), sc['X'][1], sc['L']) if sc.get('X') else np.full(len(ri), float(sc['L']))
    rj = KP.landing(np.clip(T[ri], 1, KP.NSTEP), lf) + LAND_SHIFT; j0 = KP.landing(np.full(len(ri), KP.J0), lf)
    kept = {'KEEP': np.ones(len(ri), bool), 'MODEL-J': ~(act[ri] & (t >= rj)), 'PULL-ALL': ~((KP.J0 < endj[ri]) & (t >= j0))}
    win = np.where(t < KP.STEP * KP.J0, 0, np.where(t < j0, 1, 2)); age = np.digitize(t - float(sc['A']), SC.AGE_EDGES)
    src = Fs.src.values.astype(np.int64) if 'src' in Fs else np.full(len(Fs), -1); out = {}
    for c, k in kept.items():
        v = val * k; out[c] = np.bincount(ri, v, n)
        for w in range(len(WINDOWS)):
            out[f'{c}_w{w}'] = np.bincount(ri, v * (win == w), n)
        for b in range(len(SC.AGE_LABELS)):
            out[f'{c}_a{b}'] = np.bincount(ri, v * (age == b), n)
    for c_ in range(len(SC.SRC_LABELS)):
        out[f'MODEL-J_s{c_}'] = np.bincount(ri, val * kept['MODEL-J'] * (src == c_), n)
    lat = np.where(SC.slow_order(E.oid.values, sc['X'][0]), sc['X'][1], sc['L']) if sc.get('X') else np.full(n, float(sc['L']))      # G1: per opportunity
    inf_ = lambda x: np.where(np.isnan(x), np.inf, x); tf = inf_(E.t_full_s.values.astype(float)); ld = inf_(E.land_s.values.astype(float))
    rc = {'KEEP': np.full(n, np.inf), 'MODEL-J': np.where(act, KP.landing(np.clip(T, 1, KP.NSTEP), lat) + LAND_SHIFT, np.inf),
          'PULL-ALL': np.where(KP.J0 < endj, KP.landing(np.full(n, KP.J0), lat), np.inf)}
    rq = {'KEEP': np.full(n, np.inf), 'MODEL-J': np.where(act, KP.STEP * T, np.inf), 'PULL-ALL': np.where(KP.J0 < endj, KP.STEP * KP.J0, np.inf)}
    for c, k in kept.items():
        full = np.where((tf < rc[c]) & (tf < ld) & (tf <= KP.H), tf, np.inf); cand = np.vstack([full, ld, rc[c], np.full(n, KP.H)])
        end = cand.min(0); out[f'end_{c}'] = end; out[f'cause_{c}'] = cand.argmin(0); out[f'req_{c}'] = rq[c]; out[f'land_{c}'] = rc[c]
        qk = Fs.q.values * k; pm = (t >= rq[c][ri]) & (t < rc[c][ri])
        out[f'sh_{c}'] = np.bincount(ri, qk, n); out[f'tint_{c}'] = np.bincount(ri, qk * np.maximum(KP.H - t, 0.), n)
        out[f'res_{c}'] = KP.SIZE * end - np.bincount(ri, qk * np.maximum(end[ri] - t, 0.), n)
        out[f'pend_{c}'] = np.bincount(ri, val * pm, n); out[f'pendn_{c}'] = np.bincount(ri, pm.astype(float), n)
    D = pd.DataFrame(out); D['day'] = E.day.values.astype(str); D['oid'] = E.oid.values.astype(int); D['side'] = E.side.values.astype(int); D['T'] = T; D['act'] = act
    D['endj'] = endj; D['refused_s'] = E.refused_s.values.astype(bool)
    return D


def subset(O, Os, Fs, keys):
    """The bank's inputs restricted to the opportunities an entry rule posts."""
    k = pd.MultiIndex.from_tuples(sorted(keys), names=['day', 'oid']) if keys else pd.MultiIndex.from_tuples([], names=['day', 'oid'])
    on = lambda df: pd.MultiIndex.from_arrays([df.day.astype(str).values, df.oid.astype(int).values]).isin(k)
    return O[on(O)], Os[on(Os)], Fs[on(Fs)]


def bank_values(O, Os, Fs, sc, fn, keys):
    """The bank's KEEP / MODEL-J / PULL-ALL values of the opportunities in keys (None: all), with its cancel-pending (C3) numbers."""
    if keys is not None:
        O, Os, Fs = subset(O, Os, Fs, keys)
    if not len(Os):
        return {'KEEP': 0., 'MODEL-J': 0., 'PULL-ALL': 0., 'eligible': 0, 'pulls': 0, 'pending_fills': 0, 'pending_value': 0.}
    b = SC.outcome(O, Os, Fs, sc, triggers=fn)
    return {'KEEP': b['KEEP'], 'MODEL-J': b['KEEP'] + b['score'], 'PULL-ALL': b['KEEP'] + b['PULLALL'], 'eligible': b['eligible'], 'pulls': b['pulls'],
            'pending_fills': b['cancel_pending']['fills'], 'pending_value': b['cancel_pending']['value']}


def close(a, b, rel=1e-6):
    return abs(float(a) - float(b)) <= rel * max(1., abs(float(a)), abs(float(b)))


def q3(x):
    """p10 / p50 / p90 (None when empty)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x, (.1, .5, .9))] if len(x) else None


def bins_of(v):
    """The plan's fixed signed bins (right-closed), exact 0 apart: an index per value (-1: exactly 0)."""
    v = np.asarray(v, float); return np.where(v == 0., -1, np.digitize(v, EDGES, right=True))


def bin_label(i):
    e = (-np.inf,) + EDGES + (np.inf,)
    return 'exactly 0' if i < 0 else f'({e[i]:g}, {e[i + 1]:g}]'


def run(day, out=None, sc_out=None, kp=None, bench=None, export=None, policy=None, pred=None):  # noqa: C901
    """Every entry rule x cancellation rule in every scenario of the bank on one day, and the G1 ledger / metrics at B -> entry_<day>.json / .txt /
    .csv / entry_ledger_<day>.parquet."""
    start = time.time(); C.require(day in KP.DAYS, 'undeclared day'); out = Path(out or OUT); sc_out = Path(sc_out or SC.OUT); kp = Path(kp or KP.OUT)
    bench = Path(bench or P.BENCH); export = Path(export or P.EXPORT); policy = Path(policy or P.OUT); C.require(not (out / f'entry_{day}.json').exists(), 'output exists')
    bj = json.loads((bench / 'bench.json').read_text()); pf_ = bench / f'pred_{day}.parquet'; bad = {}
    fail = lambda k, m: bad.setdefault(k, []).append(m)
    if pred is None:
        C.require(bj['outputs'].get(pf_.name) == C.sha(pf_), f'Q0e: the prediction file {pf_.name} does not match the benchmark receipt')
        pred = pd.read_parquet(pf_, columns=PRED_COLS)
    pred = pred[pred.day.astype(str) == day]
    pj = json.loads((policy / f'policy_{day}.json').read_text())
    if pj['inputs']['predictions'] != (C.sha(pf_) if pf_.exists() else None):
        fail('Q0a', 'the committed policy output was computed on another prediction file')
    er = json.loads((export / day / 'receipt.json').read_text()); exp_post = {}; ret = []
    for x, h in er['outputs'].items():
        C.require(C.sha(export / day / x) == h, f'Q0e: export part {x} changed since its receipt')
        g = pd.read_parquet(export / day / x, columns=RET_COLS); exp_post.update(zip(g.oid[g.kind == 'post'].astype(int), g.status[g.kind == 'post'].astype(str)))
        ret.append(g[g.kind == 'retain'].drop(columns=['kind']))
    ret = pd.concat(ret, ignore_index=True) if ret else pd.DataFrame(columns=RET_COLS)
    O, Fl, S, SF, Op, rec = V.load([day], sc_out, kp); O['day'] = O.day.astype(str); Op = Op.assign(day=Op.day.astype(str))
    pp = pred[pred.kind == 'post']; got = set(zip(pp.day.astype(str), pp.oid.astype(int))); want = set(zip(Op.day, Op.oid.astype(int)))
    with_path = {(day, o) for o, s_ in exp_post.items() if s_ in ('eligible', 'refused_post_only')}
    if not (got == want == with_path and len(pp) == len(got)):
        fail('Q0e', f'{len(got - want):,} POST predictions for states the bank does not value, {len(want - got):,} valued states without one, '
                    f'{len(with_path ^ want):,} states where the export\'s POST rows with a path and the bank differ')
    hit = P.triggers_of(pred); fn = P.trigger_fn(hit); sel = entry_of(pred)
    tr = bj.get('train'); C.require(bool(tr), 'Q0e: the benchmark records no training days')
    trained = float(I._ms(f'{max(tr)[:4]}-{max(tr)[4:6]}-{max(tr)[6:]}') + 86400000); par = entry_parity(pred, trained)
    exec_cmp = sel - par['tiny']
    if not (par['rows'] == len(pp) and par['rows'] > par['refused'] and par['verdict_disagreements'] == 0 and exec_cmp == par['posts']):
        fail('Q0d', f"on {par['rows']:,} POST forecasts the contract's verdict differs from J_ok on {par['verdict_disagreements']:,} ({par['refused']:,} refused, "
                    f"{par['accepted_without_round_trip']:,} accepted without giving back the moments); it posts {len(par['posts']):,}, the rule {len(exec_cmp):,} "
                    f'({len(par["posts"] ^ exec_cmp):,} differ)')
    keys = Op[CELL].reset_index(drop=True).assign(oid=Op.oid.values.astype(int)); keys['sel'] = [(d_, int(o)) in sel for d_, o in zip(keys.day, keys.oid)]
    keys['w'] = null_weights(keys, keys.sel.values); wi = keys.set_index(['day', 'oid'])
    rp = set(zip(pred[pred.kind == 'retain'].day.astype(str), pred[pred.kind == 'retain'].oid.astype(int)))
    res = {r: {} for r in ENTRY}; nul = {}; ny = {}; Dbs = {}; pol = pj['values'][P.RULE]; sides = {}
    adm = {'POST-ALL': set(want), 'MODEL-POST': set(sel)}
    for sc in SC.SCEN:
        nm = SC.name(sc); Os, Fs = V.scen_inputs(sc, Op, Fl, S, SF); Os = Os.assign(day=Os.day.astype(str)); Fs = Fs.assign(day=Fs.day.astype(str))
        if DUP_FILL:
            Fs = pd.concat([Fs, Fs], ignore_index=True)
        D = per_order(O, Os, Fs, sc, fn); ix = pd.MultiIndex.from_arrays([D.day.values, D.oid.values]); D['sel'] = ix.isin(pd.MultiIndex.from_tuples(sorted(sel))) if sel else False
        D['w'] = wi.w.reindex(ix).values; refs = set(zip(Os.day[Os.refused.astype(bool)], Os.oid[Os.refused.astype(bool)].astype(int)))
        for r, ks in (('POST-ALL', None), ('MODEL-POST', sel)):
            res[r][nm] = bank_values(O, Os, Fs, sc, fn, ks); m = np.ones(len(D), bool) if ks is None else D.sel.values
            if int(D.act.values[m].sum()) != res[r][nm]['pulls']:                         # Q0b: the executed pulls are the bank's
                fail('Q0b', f"{nm} {r}: {int(D.act.values[m].sum()):,} executed MODEL-J pulls on the per-opportunity path, the bank {res[r][nm]['pulls']:,}")
            for c in CANCEL:                                                                # Q0b
                if not close(res[r][nm][c], D[c].values[m].sum()):
                    fail('Q0b', f'{nm} {r} {c}: the bank {res[r][nm][c]:,.3f} vs the per-opportunity path {D[c].values[m].sum():,.3f}')
            n_ref = len(adm[r] & refs)                                                     # Q0g: refusal accounting
            if not (len(adm[r]) == res[r][nm]['eligible'] + n_ref and int(m.sum()) == res[r][nm]['eligible'] and not D.refused_s.values[m].any()):
                fail('Q0g', f"{nm} {r}: {len(adm[r]):,} admitted vs {res[r][nm]['eligible']:,} activated + {n_ref:,} post-only refused; {int(m.sum()):,} on the per-opportunity path")
            for sd, ms in (('both', m), ('bid', m & (D.side.values == 0)), ('ask', m & (D.side.values == 1))):
                for c in CANCEL:
                    sides[(r, c, sd, nm)] = float(D[c].values[ms].sum())
        for w in (0, 1):                                                                    # Q0f: branch identity before the earliest effect
            if not all(np.allclose(D[f'KEEP_w{w}'].values, D[f'{c}_w{w}'].values, rtol=0, atol=1e-9) for c in CANCEL):
                fail('Q0f', f'{nm}: fills before the {"first decision" if w == 0 else "earliest effect"} are worth different amounts under the cancellation rules')
        nul[nm] = {c: float((D.w.values * D[c].values).sum()) for c in CANCEL}
        ny[nm] = float((D['MODEL-J'].values - D['MODEL-J_a0'].values)[D.sel.values].sum())
        if any(not close(float(D[c].values.sum()), float((D.w.values * D[c].values).sum() if NULL_GLOBAL else D[c].values.sum())) for c in CANCEL) or (
                bool(keys.sel.all()) and any(not close(nul[nm][c], res['POST-ALL'][nm][c]) for c in CANCEL)):
            fail('Q0b', f'{nm}: the matched null of POST-ALL is not its value')
        ref_ = {'KEEP': pol[nm]['KEEP'], 'MODEL-J': pol[nm]['KEEP'] + pol[nm]['score'], 'PULL-ALL': pol[nm]['KEEP'] + pol[nm]['PULLALL']}   # Q0a
        for c in CANCEL:
            if not close(res['POST-ALL'][nm][c], ref_[c]):
                fail('Q0a', f'{nm} {c}: POST-ALL {res["POST-ALL"][nm][c]:,.3f} vs the committed policy output {ref_[c]:,.3f}')
        if nm in (B_NAME, M250_NAME):
            Dbs[nm] = D
        C.guard('pipeline entry')
    Db = Dbs[B_NAME].copy(); Db['has_retain'] = [(d_, int(o)) in rp for d_, o in zip(Db.day, Db.oid)]
    Db['group'] = np.where(Db.act & ((Db['T'] == KP.J0) | GROUP_ANY), 0, np.where(Db.act, 1, np.where(Db.has_retain, 2, 3)))
    lab = pp.set_index(pp.oid.astype(int)); Db['label'] = lab.value_m50.reindex(Db.oid.values).values.astype(float)    # Q0e: KEEP at B = the POST label
    refd = Op[Op.status == 'refused_post_only'].oid.astype(int)
    if not (np.allclose(Db.KEEP.values, Db.label.values, rtol=0, atol=1e-6) and np.allclose(lab.value_m50.reindex(refd).values.astype(float), 0., rtol=0, atol=1e-9)):
        fail('Q0e', f"at B the KEEP value differs from the exported POST label on {int((~np.isclose(Db.KEEP.values, Db.label.values, rtol=0, atol=1e-6)).sum()):,} opportunities")
    dif = Db['MODEL-J'].values - Db['PULL-ALL'].values; gsum = [float(dif[Db.group.values == g].sum()) for g in range(len(GROUPS))]
    wsum = {c: [float(Db[f'{c}_w{w}'].sum()) for w in range(len(WINDOWS))] for c in CANCEL}
    if not (close(sum(gsum), dif.sum()) and gsum[0] == 0. and all(close(sum(wsum[c]), Db[c].sum()) for c in CANCEL)):           # Q0c
        fail('Q0c', f'the attribution groups {gsum} or the windows do not sum to their totals (or the first-live-step group is not 0)')
    if bad:
        sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(f'{k} ({len(v)} failures; first: {v[0]})' for k, v in sorted(bad.items())))
    V_ = lambda r, c: {nm: res[r][nm][c] for nm in res[r]}; Ent = lambda c: {nm: res['MODEL-POST'][nm][c] - nul[nm][c] for nm in nul}
    lo_hi = lambda x: (min(x.values()), max(x.values())); R = {}; RNG = {}
    for k, x, ref in (('N1 V(MODEL-POST, MODEL-J) vs NO-ENTRY', V_('MODEL-POST', 'MODEL-J'), 'NO-ENTRY'), ('N2 V(MODEL-POST, PULL-ALL) vs NO-ENTRY', V_('MODEL-POST', 'PULL-ALL'), 'NO-ENTRY'),
                      ('N3 V(MODEL-POST, KEEP) vs NO-ENTRY', V_('MODEL-POST', 'KEEP'), 'NO-ENTRY'), ('N4 V(POST-ALL, MODEL-J) vs NO-ENTRY', V_('POST-ALL', 'MODEL-J'), 'NO-ENTRY'),
                      ('N5 V(MODEL-POST, MODEL-J) - V(POST-ALL, MODEL-J)', {nm: res['MODEL-POST'][nm]['MODEL-J'] - res['POST-ALL'][nm]['MODEL-J'] for nm in nul}, 'POST-ALL'),
                      ('N6 V(MODEL-POST, PULL-ALL) - V(POST-ALL, PULL-ALL)', {nm: res['MODEL-POST'][nm]['PULL-ALL'] - res['POST-ALL'][nm]['PULL-ALL'] for nm in nul}, 'POST-ALL'),
                      ('N1-NY V(MODEL-POST, MODEL-J) WITHOUT FILLS ON ORDERS < 100 MS OLD vs NO-ENTRY', ny, 'NO-ENTRY')):
        R[k] = V.above(*lo_hi(x), ref); RNG[k] = lo_hi(x)
    for k, c in (('N7 E_entry UNDER MODEL-J (vs the matched null)', 'MODEL-J'), ('N8 E_entry UNDER PULL-ALL (vs the matched null)', 'PULL-ALL')):
        R[k] = SC.stable(*lo_hi(Ent(c))); RNG[k] = lo_hi(Ent(c))
    R['C2 N1 V AT M250 vs M50'] = SC.at_m250(res['MODEL-POST'][B_NAME]['MODEL-J'], res['MODEL-POST'][M250_NAME]['MODEL-J'])
    R['C2 N7 E AT M250 vs M50'] = SC.at_m250(Ent('MODEL-J')[B_NAME], Ent('MODEL-J')[M250_NAME])
    s_ = Db.sel.values; vb = [float(Db[f'MODEL-J_a{b}'].values[s_].sum()) for b in range(len(SC.AGE_LABELS))]
    R['C1 N1 V AT B, B1a-WEIGHTED BY AGE'] = SC.b1a_sign(res['MODEL-POST'][B_NAME]['MODEL-J'], {k: float(sum(w * x for w, x in zip(wt, vb))) for k, wt in FV.B1A.items()})
    for sd in ('both', 'bid', 'ask'):                                                       # the plan's economic gate predicates (descriptive in G1)
        e1, e0 = sides[('MODEL-POST', 'MODEL-J', sd, B_NAME)], sides[('MODEL-POST', 'PULL-ALL', sd, B_NAME)]
        R[f'PLAN GATE AT B, {sd}: E1 vs N0'] = 'POSITIVE' if e1 > 0 else 'NOT-POSITIVE'
        R[f'PLAN GATE AT B, {sd}: E1 vs E0 (same admissions)'] = 'POSITIVE' if e1 - e0 > 0 else 'NOT-POSITIVE'
    R['CONTROLS'] = 'YES'
    # printed at B: windows, attribution, calibration, C3, C12 (per entry rule and side)
    tabs = {'windows': [], 'attribution': [], 'calibration': []}
    for r in ENTRY:
        m0 = np.ones(len(Db), bool) if r == 'POST-ALL' else Db.sel.values
        for sd, ms in (('both', m0), ('bid', m0 & (Db.side.values == 0)), ('ask', m0 & (Db.side.values == 1))):
            for c in CANCEL:
                tabs['windows'].append(dict(entry=r, side=sd, cancel=c, n=int(ms.sum()), total=float(Db[c].values[ms].sum()), **{f'w{w}': float(Db[f'{c}_w{w}'].values[ms].sum()) for w in range(len(WINDOWS))},
                                            pending_fills=int(Db[f'pendn_{c}'].values[ms].sum()), pending_value=float(Db[f'pend_{c}'].values[ms].sum())))
            for g in range(len(GROUPS)):
                mg = ms & (Db.group.values == g)
                tabs['attribution'].append(dict(entry=r, side=sd, group=GROUPS[g], orders=int(mg.sum()), model_j=float(Db['MODEL-J'].values[mg].sum()),
                                                pull_all=float(Db['PULL-ALL'].values[mg].sum()), difference=float(dif[mg].sum())))
    lab_all = pp.assign(oid=pp.oid.astype(int), day=pp.day.astype(str)); lab_all['sel'] = [(d_, o) in sel for d_, o in zip(lab_all.day, lab_all.oid)]
    lab_all = lab_all[lab_all.J_ok.astype(bool)]                                            # calibration on the coherent forecasts (the rest are never used)
    vb_ = Db.set_index('oid'); sg = np.where(lab_all.side.values == 0, 1., -1.); edge = sg * (lab_all.pf_mid.values - lab_all.p.values)
    for r in ENTRY:
        m0 = np.ones(len(lab_all), bool) if r == 'POST-ALL' else lab_all.sel.values
        for sd, ms in (('both', m0), ('bid', m0 & (lab_all.side.values == 0)), ('ask', m0 & (lab_all.side.values == 1))):
            g = lab_all[ms]; e_ = edge[ms]; real = {c: vb_[c].reindex(g.oid.values).fillna(0.).values for c in CANCEL}
            q, qm = float(g.q_m50.sum()), float(g.qm_m50.sum())
            tabs['calibration'].append(dict(entry=r, side=sd, n=int(len(g)), amount_ratio=float(g.J_Eq.sum() / q) if q else None, move_ratio=float(g.J_Eqm.sum() / qm) if qm else None,
                                            predicted_mean=float(g.J_V.mean()) if len(g) else None, label_mean=float(g.value_m50.mean()) if len(g) else None,
                                            **{f'realized_mean_{c}': float(real[c].mean()) if len(g) else None for c in CANCEL},
                                            error_edge_term=float((e_ * (g.J_Eq.values - g.q_m50.values)).sum()), error_move_term=float((g.J_Eqm.values - g.qm_m50.values).sum()),
                                            error_total=float((g.J_V.values - g.value_m50.values).sum()), error_vs_MODEL_J=float((g.J_V.values - real['MODEL-J']).sum())))
    src_b = {sl: float(Db[f'MODEL-J_s{i}'].values[Db.sel.values].sum()) for i, sl in enumerate(SC.SRC_LABELS)}
    st = O.status.value_counts().to_dict()
    # ---- G1 (plan 0c3d2523): the ledger and the metric schema at B
    A_ = float(SC.B0['A']); Oi = O.assign(oid=O.oid.astype(int)).set_index('oid'); Db['m'] = Oi.m.reindex(Db.oid.values).values; Db['t'] = Oi.t.reindex(Db.oid.values).values
    npath = {sd: int(((Op.side == s) if s is not None else np.ones(len(Op), bool)).sum()) for sd, s in (('both', None), ('bid', 0), ('ask', 1))}
    norig = {sd: int(((O.side == s) if s is not None else np.ones(len(O), bool)).sum()) for sd, s in (('both', None), ('bid', 0), ('ask', 1))}
    pairs = Op[Op.side == 0][['m', 't']].merge(Op[Op.side == 1][['m', 't']], on=['m', 't']); npair = len(pairs)
    adm_n = {r: {sd: sum(1 for (d_, o) in adm[r] if sd == 'both' or int(Oi.side.loc[o]) == (0 if sd == 'bid' else 1)) for sd in npath} for r in ENTRY}
    Rr = ret.assign(oid=ret.oid.astype(int)).merge(Db[['oid', 'side', 'm', 't', 'sel'] + [f'end_{c}' for c in CANCEL] + ['T', 'act']], on='oid', how='inner')
    tj = KP.STEP * Rr.j.values.astype(float); dep = Rr.o_depth.values.astype(float); g1 = {}
    for pn, (r, c) in POL.items():
        mA = np.ones(len(Db), bool) if r == 'POST-ALL' else Db.sel.values; mR = np.ones(len(Rr), bool) if r == 'POST-ALL' else Rr.sel.values.astype(bool)
        live = np.maximum(Db[f'end_{c}'].values - A_, 0.); fin_ = np.isfinite(Db[f'land_{c}'].values) & (Db[f'cause_{c}'].values == 2)
        pend_t = np.where(fin_, np.where(fin_, Db[f'land_{c}'].values, 0.) - np.where(fin_, Db[f'req_{c}'].values, 0.), 0.)
        lr = mR & (tj >= A_) & (tj < Rr[f'end_{c}'].values); ab = lr & (dep <= 1e-9); unk = lr & ~np.isfinite(dep)
        pre = np.maximum(np.minimum(Db[f'end_{c}'].values, KP.STEP * KP.J0) - A_, 0.)
        lb = pd.DataFrame(dict(m=Db.m.values, t=Db.t.values, side=Db.side.values, live=np.where(mA, live, 0.)))
        pl = pairs.merge(lb[lb.side == 0][['m', 't', 'live']], on=['m', 't'], how='left').merge(lb[lb.side == 1][['m', 't', 'live']], on=['m', 't'], how='left', suffixes=('_b', '_a')).fillna(0.)
        gab = pd.DataFrame(dict(m=Rr.m.values[ab], t=Rr.t.values[ab], j=Rr.j.values[ab], side=Rr.side.values[ab])).groupby(['m', 't', 'j']).side.nunique()
        x = dict(availability=dict(both_sides_ms=float(np.minimum(pl.live_b, pl.live_a).sum()), either_side_ms=float(np.maximum(pl.live_b, pl.live_a).sum()), pairs=npair,
                                   both_sides=float(np.minimum(pl.live_b, pl.live_a).sum() / (npair * SLOT_MS)) if npair else None,
                                   either_side=float(np.maximum(pl.live_b, pl.live_a).sum() / (npair * SLOT_MS)) if npair else None,
                                   at_best_both_sides=float((gab == 2).sum() * KP.STEP / (npair * SLOT_MS)) if npair else None,
                                   at_best_either_side=float((gab >= 1).sum() * KP.STEP / (npair * SLOT_MS)) if npair else None))
        for sd, s in (('both', None), ('bid', 0), ('ask', 1)):
            ms = mA & ((Db.side.values == s) if s is not None else True); mr = (Rr.side.values == s) if s is not None else np.ones(len(Rr), bool)
            vals = Db[c].values[ms]; sh = Db[f'sh_{c}'].values[ms]; att = adm_n[r][sd]; acc = int(ms.sum()); fl = int((sh > 0).sum())
            ep = np.r_[vals, np.zeros(att - acc)] if att >= acc else vals                    # admitted episodes (a post-only refusal is worth 0)
            loss = -ep; hrs = npath[sd] * SLOT_MS / 3.6e6; filled_sh = float(sh.sum()); cz = Db[f'cause_{c}'].values[ms]
            life = (Db[f'end_{c}'].values - A_)[ms]; pol_c = cz == 2; inv = sh * (1. if s == 0 else -1. if s == 1 else np.where(Db.side.values[ms] == 0, 1., -1.))
            rev = None
            if c == 'MODEL-J':
                live_j0 = ms & (Db.endj.values > KP.J0); first = ms & Db.act.values & (Db['T'].values == KP.J0); allp = ms & Db.act.values
                rev = dict(first_review_pulls=int(first.sum()), live_at_first_review=int(live_j0.sum()), all_pulls=int(allp.sum()),
                           of_live=float(first.sum() / live_j0.sum()) if live_j0.sum() else None, of_pulls=float(first.sum() / allp.sum()) if allp.sum() else None)
            elif c == 'PULL-ALL':
                live_j0 = ms & (Db.endj.values > KP.J0); rev = dict(first_review_pulls=int(live_j0.sum()), live_at_first_review=int(live_j0.sum()), all_pulls=int(live_j0.sum()),
                                                                   of_live=1. if live_j0.sum() else None, of_pulls=1. if live_j0.sum() else None)
            keepd = None
            if c == 'MODEL-J':                                                              # the rule's KEEP decisions it reaches (before its pull)
                kd = mR & mr & (tj < Rr['end_MODEL-J'].values) & np.where(Rr.act.values.astype(bool), Rr.j.values < Rr['T'].values, True)
                keepd = dict(decisions=int(kd.sum()), age_ms=q3(Rr.o_age.values[kd]), queue_ahead_known=q3(Rr.fp_queue.values[kd]), remaining_known=q3(Rr.fp_remaining.values[kd]),
                             queue_unknown=int((kd & (~np.isfinite(Rr.fp_queue.values.astype(float)) | (Rr.o_qunk.values.astype(float) > 0))).sum()), resets=0)
            x[sd] = dict(value=float(vals.sum()), per_original_opportunity=float(vals.sum() / norig[sd]) if norig[sd] else None,
                         participation=dict(opportunities_with_path=npath[sd], attempts=att, accepted=acc, filled_orders=fl, shares_submitted=float(KP.SIZE * att), shares_filled=filled_sh,
                                            accepted_of_attempts=acc / att if att else None, filled_of_accepted=fl / acc if acc else None, shares_filled_of_submitted=filled_sh / (KP.SIZE * att) if att else None),
                         availability=dict(any_resting=float(live[ms].sum() / (npath[sd] * SLOT_MS)) if npath[sd] else None, cancel_pending_ms=float(pend_t[ms].sum()),
                                           live_ms=float(live[ms].sum()), at_best_grid=float(ab[mr].sum() * KP.STEP / (npath[sd] * SLOT_MS)) if npath[sd] else None,
                                           unknown_depth_grid_ms=float(unk[mr].sum() * KP.STEP), live_before_first_grid_ms=float(pre[ms].sum())),
                         lifetime=dict(p10_p50_p90_ms=q3(life), causes={CAUSES[i]: int((cz == i).sum()) for i in range(len(CAUSES))},
                                       request_after_submission_p50_ms=q3(Db[f'req_{c}'].values[ms][pol_c])[1] if pol_c.any() else None,
                                       effect_after_request_p50_ms=q3(Db[f'land_{c}'].values[ms][pol_c] - Db[f'req_{c}'].values[ms][pol_c])[1] if pol_c.any() else None),
                         reversals=rev, queue_at_keep=keepd,
                         end_markout_c_per_share=float(vals.sum() / filled_sh) if filled_sh > 0 else None,
                         churn=dict(posts_per_hour=att / hrs if hrs else None, posts_per_100_filled_shares=(100. * att / filled_sh) if filled_sh > 0 else None,
                                    policy_cancels_per_hour=int(pol_c.sum()) / hrs if hrs else None,
                                    structural_cancels_per_hour=int((cz == 1).sum()) / hrs if hrs else None, replacements=0,
                                    policy_cancels_per_100_filled_shares=(100. * int(pol_c.sum()) / filled_sh) if filled_sh > 0 else None,
                                    structural_per_100_filled_shares=(100. * int((cz == 1).sum()) / filled_sh) if filled_sh > 0 else None),
                         tail=dict(episodes=int(len(ep)), mean=float(ep.mean()) if len(ep) else None, p_loss=float((ep < 0).mean()) if len(ep) else None,
                                   p_severe=float((ep < -KP.SEVERE).mean()) if len(ep) else None, severe_excess=float(np.maximum(loss - KP.SEVERE, 0.).sum()),
                                   loss_p95=float(np.quantile(loss, .95)) if len(ep) else None, loss_p99=float(np.quantile(loss, .99)) if len(ep) else None),
                         inventory=dict(mean_abs_terminal=float(np.abs(inv).mean()) if len(inv) else None, peak_abs_terminal=float(np.abs(inv).max()) if len(inv) else None,
                                        time_integrated_share_s=float(Db[f'tint_{c}'].values[ms].sum() / 1e3), reservation_share_s=float(Db[f'res_{c}'].values[ms].sum() / 1e3), cap_breaches=0),
                         scenario_range=lo_hi({nm: sides[(r, c, sd, nm)] for nm in nul}))
        g1[pn] = x
    for pn, (r, c) in POL.items():                                                          # paired differences: vs N0 (the value) and vs the same-admission PULL-ALL
        for sd in ('both', 'bid', 'ask'):
            g1[pn][sd]['vs_N0'] = g1[pn][sd]['value']; g1[pn][sd]['vs_same_admission_PULL_ALL'] = g1[pn][sd]['value'] - g1[('A0' if r == 'POST-ALL' else 'E0')][sd]['value']
    cal = {'post': [], 'retain_neutral': [], 'retain_reached': []}                          # the plan's fixed bins
    pc = lab_all[lab_all.J_ok.astype(bool)]; rmj = vb_['MODEL-J'].reindex(pc.oid.values).fillna(0.).values; rpa = vb_['PULL-ALL'].reindex(pc.oid.values).fillna(0.).values
    for sd, s in (('bid', 0), ('ask', 1)):
        bi = bins_of(pc.J_V.values)
        for b in sorted(set(bi)):
            mb = (bi == b) & (pc.side.values == s)
            if mb.any():
                pv, lv = pc.J_V.values[mb], pc.value_m50.values[mb]
                cal['post'].append(dict(side=sd, bin=bin_label(b), n=int(mb.sum()), predicted=float(pv.sum()), label=float(lv.sum()), MODEL_J=float(rmj[mb].sum()), PULL_ALL=float(rpa[mb].sum()),
                                        bias=float((pv - lv).mean()), mae=float(np.abs(pv - lv).mean())))
    rr = pred[(pred.kind == 'retain') & pred.J_ok.astype(bool)].assign(oid=lambda z: z.oid.astype(int)); fh = hit.reset_index(); fh.columns = ['day', 'oid', 'first']
    rr = rr.merge(fh.assign(day=fh.day.astype(str), oid=fh.oid.astype(int)), on=['day', 'oid'], how='left'); reach = rr.j.values <= rr['first'].fillna(10 ** 6).values
    for key, mk in (('retain_neutral', np.ones(len(rr), bool)), ('retain_reached', reach)):
        bi = bins_of(rr.J_V.values)
        for sd, s in (('bid', 0), ('ask', 1)):
            for b in sorted(set(bi)):
                mb = mk & (bi == b) & (rr.side.values == s)
                if mb.any():
                    pv, lv = rr.J_V.values[mb], rr.value_m50.values[mb]
                    cal[key].append(dict(side=sd, bin=bin_label(b), n=int(mb.sum()), predicted=float(pv.sum()), label=float(lv.sum()), bias=float((pv - lv).mean()), mae=float(np.abs(pv - lv).mean())))
    manifest = dict(original_opportunities=norig, by_status={str(k): int(v) for k, v in st.items()}, with_path=npath, bid_ask_pairs_with_path=npair,
                    refused_at_B=int((Op.status == 'refused_post_only').sum()), admitted={r: adm_n[r] for r in ENTRY},
                    chronology=dict(train_days=list(tr), evaluation_day=day, benchmark_asof=bj.get('asof'), policy_asof=pj.get('asof'), export_receipt=C.sha(export / day / 'receipt.json'),
                                    bank_days=rec), not_computed=dict(markouts_100ms_1s='no mid is stored per fill (a tape read is the next module); the END markout is the value per filled share',
                                                                       queue_at_fill='the kernel\'s queue position at a fill is not stored'))
    led = O[['day', 'oid', 'm', 'side', 't', 'phase', 'status']].assign(oid=O.oid.astype(int)).merge(Db.drop(columns=[c_ for c_ in Db.columns if c_.startswith('MODEL-J_s') or c_ in ('m', 't')]),
                                                                                                    on=['day', 'oid', 'side'], how='left')
    led['admitted_MODEL_POST'] = [(d_, int(o)) in sel for d_, o in zip(led.day, led.oid)]; led['has_path'] = [(d_, int(o)) in want for d_, o in zip(led.day, led.oid)]
    led = led.merge(pp[['oid', 'J_V', 'J_ok']].assign(oid=pp.oid.astype(int)).rename(columns={'J_V': 'post_J_V', 'J_ok': 'post_J_ok'}), on='oid', how='left')
    for pn, (r, c) in POL.items():
        led[f'value_{pn}'] = np.where(led.has_path & (led.admitted_MODEL_POST if r == 'MODEL-POST' else True), led[c].fillna(0.), 0.)
    rows = [dict(entry=r, cancel=c, side=sd, scenario=nm, V=sides[(r, c, sd, nm)], null=nul[nm][c] if (r == 'MODEL-POST' and sd == 'both') else None,
                 E_entry=(res[r][nm][c] - nul[nm][c]) if (r == 'MODEL-POST' and sd == 'both') else None, eligible=res[r][nm]['eligible'], pulls=res[r][nm]['pulls'])
            for r in ENTRY for c in CANCEL for sd in ('both', 'bid', 'ask') for nm in res[r]]
    L = [f'ENTRY WITH A FROZEN DOWNSTREAM CANCELLATION RULE (pf_pipeline_entry.py; {day}; c summed; local marks at END, not P&L; review 1adf6a33\'s next priority, plan 0c3d2523 G1)',
         f"  {len(O):,} committed states: " + ', '.join(f'{k} {v:,}' for k, v in sorted(st.items())) + f'; {len(Op):,} opportunities with a path; MODEL-POST posts '
         f"{len(sel):,}; entry parity: {par['rows']:,} POST forecasts, {par['refused']:,} refused, {par['verdict_disagreements']:,} verdict disagreements, "
         f"{par['verdict_only']:,} on the verdict only; CONTROLS: YES (Q0a-Q0g)", '',
         f"  {'scenario':<24} {'A0 ALL+PA':>11} {'A1 ALL+MJ':>11} {'AK ALL+KEEP':>12} {'E0 MP+PA':>9} {'E1 MP+MJ':>9} {'EK MP+KEEP':>10} {'null E1':>9} {'E MJ':>8} {'E PA':>8} {'E1 ny':>8}"]
    for nm in nul:
        a, b = res['POST-ALL'][nm], res['MODEL-POST'][nm]
        L.append(f"  {nm:<24} {a['PULL-ALL']:>+11,.0f} {a['MODEL-J']:>+11,.0f} {a['KEEP']:>+12,.0f} {b['PULL-ALL']:>+9,.0f} {b['MODEL-J']:>+9,.0f} {b['KEEP']:>+10,.0f} "
                 f"{nul[nm]['MODEL-J']:>+9,.0f} {b['MODEL-J'] - nul[nm]['MODEL-J']:>+8,.0f} {b['PULL-ALL'] - nul[nm]['PULL-ALL']:>+8,.0f} {ny[nm]:>+8,.0f}")
    L += ['', '  AT B, VALUE BY FILL TIME AFTER SUBMISSION (c): ' + ' | '.join(f'w{i} {w}' for i, w in enumerate(WINDOWS)) + '; actual cancel-pending fills (request to landing)']
    L += [f"    {x['entry']:<10} {x['side']:<4} {x['cancel']:<8} n {x['n']:>6,}  total {x['total']:>+11,.1f}  " + '  '.join(f"w{w} {x[f'w{w}']:>+11,.1f}" for w in range(len(WINDOWS)))
          + f"  pending {x['pending_fills']:,} fills {x['pending_value']:+,.1f}" for x in tabs['windows']]
    L += ['', '  AT B, MODEL-J AGAINST PULL-ALL BY MODEL-J\'S ACTION (c)']
    L += [f"    {x['entry']:<10} {x['side']:<4} {x['group']:<38} orders {x['orders']:>6,}  MODEL-J {x['model_j']:>+11,.1f}  PULL-ALL {x['pull_all']:>+11,.1f}  difference {x['difference']:>+10,.1f}"
          for x in tabs['attribution']]
    L += ['', '  AT B, CALIBRATION ON THE OPPORTUNITIES POSTED (the J POST forecast vs the M50 label; realized under each cancellation rule)']
    f_ = lambda z, k=3: 'n/a' if z is None else f'{z:+.{k}f}'
    L += [f"    {x['entry']:<10} {x['side']:<4} n {x['n']:>6,}  E[q] / q {f_(x['amount_ratio'])}  E[qm] / qm {f_(x['move_ratio'])}  predicted {f_(x['predicted_mean'], 2)}  label {f_(x['label_mean'], 2)}  "
          f"realized KEEP / MODEL-J / PULL-ALL {f_(x['realized_mean_KEEP'], 2)} / {f_(x['realized_mean_MODEL-J'], 2)} / {f_(x['realized_mean_PULL-ALL'], 2)}  error: edge x quantity "
          f"{x['error_edge_term']:+,.0f}, movement {x['error_move_term']:+,.0f} (total {x['error_total']:+,.0f}; vs MODEL-J {x['error_vs_MODEL_J']:+,.0f})" for x in tabs['calibration']]
    bm = res['MODEL-POST'][B_NAME]
    L += ['', f"  C3 at B (MODEL-POST + MODEL-J, the bank): cancel-pending {bm['pending_fills']:,} fills / {bm['pending_value']:+,.1f} c; C12 by trade-anchor source: "
          + ' | '.join(f'{k} {v:+,.1f}' for k, v in src_b.items()) + '; C1 by age: ' + ' | '.join(f'{lab_} {x:+,.1f}' for lab_, x in zip(SC.AGE_LABELS, vb))]
    pc_ = lambda z: 'n/a' if z is None else f'{100 * z:.2f}%'; nz = lambda z, k=1: 'n/a' if z is None else f'{z:,.{k}f}'
    L += ['', f"  G1 AT B (plan 0c3d2523): N0 = 0; {len(O):,} original opportunities ({npath['both']:,} with a path, {npair:,} bid / ask pairs); per policy and side"]
    for pn in POL:
        for sd in ('both', 'bid', 'ask'):
            x = g1[pn][sd]; pa = x['participation']; av = x['availability']; li = x['lifetime']; ch = x['churn']; ta = x['tail']; iv = x['inventory']; rv = x['reversals']
            L.append(f"    {pn} {sd:<4} value {x['value']:>+11,.1f} ({x['per_original_opportunity']:+.3f} / opportunity; scenarios {x['scenario_range'][0]:+,.0f} .. {x['scenario_range'][1]:+,.0f}; "
                     f"vs same-admission PULL-ALL {x['vs_same_admission_PULL_ALL']:+,.1f}) | attempts {pa['attempts']:,}, accepted {pc_(pa['accepted_of_attempts'])}, filled {pc_(pa['filled_of_accepted'])}, "
                     f"shares {pc_(pa['shares_filled_of_submitted'])} | resting {pc_(av['any_resting'])}, at best (grid) {pc_(av['at_best_grid'])}, cancel-pending {nz(av['cancel_pending_ms'] / 1e3)} s | "
                     f"lifetime p10/p50/p90 {li['p10_p50_p90_ms'] and '/'.join(f'{v:.0f}' for v in li['p10_p50_p90_ms'])} ms, causes {li['causes']} | "
                     + (f"first-review {pc_(rv['of_live'])} of live, {pc_(rv['of_pulls'])} of pulls | " if rv else '')
                     + f"posts {nz(ch['posts_per_hour'])}/h, policy cancels {nz(ch['policy_cancels_per_hour'])}/h ({nz(ch['policy_cancels_per_100_filled_shares'], 2)} per 100 sh), "
                     f"structural {nz(ch['structural_cancels_per_hour'])}/h | loss P {pc_(ta['p_loss'])}, severe P {pc_(ta['p_severe'])}, excess {ta['severe_excess']:,.0f}, p95/p99 "
                     f"{nz(ta['loss_p95'], 2)}/{nz(ta['loss_p99'], 2)} | inventory mean |.| {nz(iv['mean_abs_terminal'], 2)}, peak {nz(iv['peak_abs_terminal'], 0)}, {iv['time_integrated_share_s']:,.0f} share-s")
        a_ = g1[pn]['availability']
        L.append(f"    {pn} pairs: either side {pc_(a_['either_side'])}, both sides {pc_(a_['both_sides'])}; at best (grid) either {pc_(a_['at_best_either_side'])}, both {pc_(a_['at_best_both_sides'])}"
                 + (f"; KEEP decisions reached {g1[pn]['both']['queue_at_keep']['decisions']:,} (age / queue ahead / remaining p10-p50-p90: {g1[pn]['both']['queue_at_keep']['age_ms']} / "
                    f"{g1[pn]['both']['queue_at_keep']['queue_ahead_known']} / {g1[pn]['both']['queue_at_keep']['remaining_known']}; queue unknown {g1[pn]['both']['queue_at_keep']['queue_unknown']:,})"
                    if g1[pn]['both']['queue_at_keep'] else ''))
    L += ['', '  CALIBRATION IN THE PLAN\'S FIXED BINS (c; coherent forecasts; POST: label and the value under MODEL-J / PULL-ALL; RETAIN: label on the rows the rule reaches)']
    L += [f"    POST {x['side']} {x['bin']:<14} n {x['n']:>6,} predicted {x['predicted']:>+10,.1f} label {x['label']:>+10,.1f} MODEL-J {x['MODEL_J']:>+10,.1f} PULL-ALL {x['PULL_ALL']:>+10,.1f} "
          f"bias {x['bias']:+.2f} MAE {x['mae']:.2f}" for x in cal['post']]
    L += [f"    RETAIN reached {x['side']} {x['bin']:<14} n {x['n']:>7,} predicted {x['predicted']:>+11,.1f} label {x['label']:>+11,.1f} bias {x['bias']:+.2f} MAE {x['mae']:.2f}" for x in cal['retain_reached']]
    L += ['', 'READINGS (declared before this module\'s outputs were read)'] + [f'  {k}: {v}' for k, v in R.items()]
    L += ['', 'NOT COMPUTED: ' + '; '.join(f'{k}: {v}' for k, v in manifest['not_computed'].items()),
          'LIMITS: consumed days (G = 1 per output); the POST model was trained on the neutral (KEEP) label; each opportunity valued alone (no shared volume, '
          'inventory or re-post); local marks, not P&L; queue rule M only (C5); no order-level validation (C10); the RETAIN inputs\' own fills are B\'s in every scenario; '
          'at-best time on the 50 ms grid.']
    out.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out / f'entry_{day}.csv', index=False); (out / f'entry_{day}.txt').write_text('\n'.join(L) + '\n')
    led.to_parquet(out / f'entry_ledger_{day}.parquet', index=False)
    C.dump(out / f'entry_{day}.json', dict(asof=C.stamp(), day=day, readings=R, reading_ranges=RNG, values=res, null=nul, no_young=ny, tables=tabs, g1=g1, calibration_bins=cal, manifest=manifest,
                                           c3=dict(fills=bm['pending_fills'], value=bm['pending_value']), c12=src_b, c1_by_age=dict(zip(SC.AGE_LABELS, vb)),
                                           statuses={str(k): int(v) for k, v in st.items()}, posted=len(sel), parity={k: v for k, v in par.items() if k not in ('posts', 'tiny')},
                                           inputs=dict(bench=C.sha(bench / 'bench.json'), predictions=C.sha(pf_) if pf_.exists() else None, policy=C.sha(policy / f'policy_{day}.json'),
                                                       export_receipt=C.sha(export / day / 'receipt.json'), days=rec),
                                           outputs={x: C.sha(out / x) for x in (f'entry_{day}.txt', f'entry_{day}.csv', f'entry_ledger_{day}.parquet')},
                                           source_sha256=C.sha(Path(__file__).resolve()), provenance=KP.provenance(), seconds=time.time() - start))
    print('\n'.join(L), flush=True)
    return res, R, Db, nul


def selftest():  # noqa: C901
    """Synthetic tape only (see the module's SELFTEST)."""
    global FLIP_ENTRY, RAW_ENTRY, NULL_GLOBAL, LAND_SHIFT, GROUP_ANY, ACT_ANY, DUP_FILL, IGNORE_REFUSED
    bad = []; src = KP.Syn(); days = list(KP.DAYS); d0 = days[0]
    k2 = pd.DataFrame(dict(day=['a'] * 6, m=[1, 1, 2, 2, 2, 2], side=[0] * 6, phase=[1] * 6))             # the null by hand: cell 1 values (10, 0) with
    w2 = null_weights(k2, [True, False, True, False, False, False])                                        # 1 posted -> 5; cell 2 (-4 x 4) with 1 -> -4
    if not abs(float((w2 * np.r_[10., 0., -4., -4., -4., -4.]).sum()) - 1.) < 1e-12:
        bad.append(f'the matched null on the hand example: {float((w2 * np.r_[10., 0., -4., -4., -4., -4.]).sum())} (want 1; a global null gives -2)')
    env = lambda d_: dict(h=600., q01=-900., q99=1050., inputs={}); syn_set = lambda d_: (pd.DataFrame(dict(t_set_ms=np.zeros(0), set_status=np.zeros(0, int))), {})
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp)
        with QL._v1_on(src, tmp):
            for d in days:
                QP.day_run(d)
        tim = SC.SynTiming(src, contra=((108, 0, 150500.),))
        with contextlib.redirect_stdout(io.StringIO()):
            for d in days:
                KP.day_run(d, src=src, v1=tmp / 'v1', out=tmp / 'kp')
            KP.report(tmp / 'kp')
            SC.day_run(d0, src=src, timing=tim, kp=tmp / 'kp', out=tmp / 'sc', env_of=env, set_of=syn_set, st_lag=150.)
            SC.report(out=tmp / 'sc', kp=tmp / 'kp', days=[d0], expect=None)
        O = pd.read_parquet(tmp / 'kp' / d0 / 'orders.parquet'); el = O[O.status == 'eligible']; Opx = O[O.status.isin(('eligible', 'refused_post_only'))]
        endj = SC.A.endj_of(el.t_full.values, el.g_struct.values)
        rows = []; want_t = {}; late = set()
        for o, e in zip(el.itertuples(), endj):                                              # planted RETAIN predictions (as the policy selftest)
            js = np.arange(KP.J0, max(int(e), KP.J0))
            k = (js[0] if int(o.oid) % 5 == 0 else js[len(js) // 2] if int(o.oid) % 3 == 0 else None) if len(js) else None   # first-step and later pulls
            for j in js:
                qm_ = -6. if k is not None and j >= k else 6.
                rows.append(dict(day=d0, oid=int(o.oid), kind='retain', j=int(j), side=int(o.side), t=float(o.t), decision_ms=float(o.t) + KP.STEP * j, p=float(o.p),
                                 pf_mid=float(o.p) + (.5 if o.side == 0 else -.5), fp_remaining=30., q_m50=0., qm_m50=0., value_m50=0., J_P=.4, J_Eq=4., J_Eqm=qm_, J_V=.5 * 4. + qm_, J_ok=True))
            if k is not None:
                want_t[(d0, int(o.oid))] = int(k)
            elif int(o.oid) % 11 == 0 and KP.J0 <= int(e) <= KP.NSTEP:                          # a predicted loss at the order's structural end: the
                rows.append(dict(rows[-1] if js.size else dict(day=d0, oid=int(o.oid), kind='retain', side=int(o.side), t=float(o.t), p=float(o.p), fp_remaining=30.,   # pull never executes
                                                               pf_mid=float(o.p) + (.5 if o.side == 0 else -.5), q_m50=0., qm_m50=0., value_m50=0., J_P=.4, J_Eq=4., J_ok=True),
                                 j=int(e), decision_ms=float(o.t) + KP.STEP * int(e), J_Eqm=-6., J_V=-4.))
                want_t[(d0, int(o.oid))] = int(e); late.add((d0, int(o.oid)))
        want_e = set()
        for o in Opx.itertuples():                                                           # planted POST predictions: post every 4th; every 7th's
            sg = 1. if o.side == 0 else -1.; v_ = float(o.v) if o.status == 'eligible' else 0.; q_ = float(o.q) if o.status == 'eligible' else 0.   # forecast refused
            up = int(o.oid) % 4 == 0; off = int(o.oid) % 7 == 0; jm = (1e4 if up else -1e4) if off else (1. if up else -5.)   # a refused forecast: its implied
            rows.append(dict(day=d0, oid=int(o.oid), kind='post', j=0, side=int(o.side), t=float(o.t), decision_ms=float(o.t), p=float(o.p), pf_mid=float(o.p) + .5 * sg,   # endpoint
                             fp_remaining=30., q_m50=q_, qm_m50=v_ - .5 * q_, value_m50=v_, J_P=.4, J_Eq=4., J_Eqm=jm, J_V=2. + jm, J_ok=not off))   # mid is far outside [0, 100]
            if up and not off:
                want_e.add((d0, int(o.oid)))
        pred = pd.DataFrame(rows); po_ = pred.index[pred.kind == 'post']
        up_ = [i for i in po_ if pred.loc[i, 'J_ok'] and pred.loc[i, 'J_V'] > 0]; dn_ = [i for i in po_ if pred.loc[i, 'J_ok'] and pred.loc[i, 'J_V'] < 0]
        pb = next(x for x in np.linspace(.05, .95, 9001) if (x * 30.) / x > 30.)                # v3.1 accepted cases: E[q | fill] at the remaining (E[q] / P
        pred.loc[up_[0], ['J_P', 'J_Eq', 'J_Eqm', 'J_V']] = [pb, pb * 30., .1 * pb * 30., .6 * pb * 30.]   # rounds above it); a nonfinite forecast flagged
        pred.loc[dn_[0], ['J_Eqm', 'J_V', 'J_ok']] = [np.nan, np.nan, False]                      # incoherent; a move of 5e-10 c at E[q] = 0 (inside the
        pred.loc[dn_[1], ['J_Eq', 'J_Eqm', 'J_V']] = [0., 5e-10, 5e-10]; want_e.add((d0, int(pred.loc[dn_[1], 'oid'])))   # tolerance: posted)
        if not (entry_of(pred) == want_e and len(want_e) > 0 and dict(P.triggers_of(pred).items()) == want_t):
            bad.append(f'planted entry set / triggers: {len(entry_of(pred))} of {len(want_e)} / {len(P.triggers_of(pred))} of {len(want_t)}')
        exp = tmp / 'exp' / d0; exp.mkdir(parents=True)
        st_ = dict(zip(Opx.oid.astype(int), Opx.status.astype(str)))
        rq = np.random.default_rng(4); ne = len(pred)                                          # the order inputs the G1 metrics read (planted)
        pred[['oid', 'kind', 'j']].assign(status=[('eligible' if k_ == 'retain' else st_[int(o_)]) for k_, o_ in zip(pred.kind, pred.oid)], o_depth=rq.integers(0, 2, ne).astype(float),
                                          o_age=KP.STEP * pred.j.values - KP.PLACE, fp_queue=np.where(rq.random(ne) < .1, np.nan, rq.uniform(0., 300., ne)), fp_remaining=30.,
                                          o_qunk=0.).to_parquet(exp / 'rows_000.parquet', index=False)
        C.dump(exp / 'receipt.json', dict(outputs={'rows_000.parquet': C.sha(exp / 'rows_000.parquet')})); bj = tmp / 'bench'; bj.mkdir()
        C.dump(bj / 'bench.json', dict(outputs={}, gate=dict(retain='PASS'), diagnostic=dict(retain='PASS'), train=['20231206']))   # the synthetic clock starts 2023-12-07
        with contextlib.redirect_stdout(io.StringIO()):
            P.run(d0, out=tmp / 'pol', sc_out=tmp / 'sc', kp=tmp / 'kp', bench=bj, export=tmp / 'exp', report=tmp / 'sc' / f'scenarios_{d0}.json', pred=pred)
        def go(out_, **kw):
            with contextlib.redirect_stdout(io.StringIO()):
                return run(d0, out=out_, sc_out=tmp / 'sc', kp=tmp / 'kp', bench=bj, export=tmp / 'exp', policy=kw.pop('policy', tmp / 'pol'), pred=kw.pop('pred', pred))
        R = None
        try:
            res, R, Db, nul = go(tmp / 'e')
        except (SystemExit, ValueError) as e:
            bad.append(f'the synthetic entry run refused: {str(e)[:400]}')
        if R is not None:
            O_, Fl, S, SF, Op, _ = V.load([d0], tmp / 'sc', tmp / 'kp'); Os, Fs = V.scen_inputs(SC.B0, Op, Fl, S, SF)   # B recomputed by a per-fill loop
            ref = Os.set_index('oid'); M = O_.set_index('oid'); brute = {c: {} for c in CANCEL}; en = SC.endj_n(ref.t_full.values, M.g_struct.reindex(ref.index).values, SC.B0['N'])
            bw = np.zeros(len(WINDOWS)); bny = 0.; bsh = {c: {} for c in CANCEL}
            endo = dict(zip(ref.index.astype(int), en))
            for f in Fs.itertuples():
                o = int(f.oid)
                if bool(ref.refused.loc[o]):
                    continue
                sg = 1. if M.side.loc[o] == 0 else -1.; v = float(f.q) * sg * (float(M.mark.loc[o]) - float(M.p.loc[o])); T = want_t.get((d0, o))
                pulled = T is not None and KP.J0 <= T < endo[o] and float(f.t) >= KP.STEP * T + SC.B0['L']
                pa = KP.J0 < endo[o] and float(f.t) >= KP.STEP * KP.J0 + SC.B0['L']
                for c, gone in (('KEEP', False), ('MODEL-J', pulled), ('PULL-ALL', pa)):
                    brute[c][o] = brute[c].get(o, 0.) + (0. if gone else v); bsh[c][o] = bsh[c].get(o, 0.) + (0. if gone else float(f.q))
                bw[0 if float(f.t) < 100. else 1 if float(f.t) < 100. + SC.B0['L'] else 2] += v               # POST-ALL, KEEP, by fill time
                bny += v if ((d0, o) in want_e and not pulled and float(f.t) - SC.B0['A'] >= 100.) else 0.  # MODEL-POST + MODEL-J, orders >= 100 ms old
            for r, ks in (('POST-ALL', None), ('MODEL-POST', want_e)):
                for c in CANCEL:
                    b_ = sum(v for o, v in brute[c].items() if ks is None or (d0, o) in ks)
                    if not close(b_, res[r][B_NAME][c]):
                        bad.append(f'B by a per-fill loop: {r} {c} {b_:,.3f} vs {res[r][B_NAME][c]:,.3f}')
            rng = np.random.default_rng(5); kk = Op[CELL].assign(day=Op.day.astype(str), oid=Op.oid.astype(int)).reset_index(drop=True)   # the null vs 400 draws
            kk['sel'] = [(d_, o) in want_e for d_, o in zip(kk.day, kk.oid)]; vv = np.array([brute['MODEL-J'].get(int(o), 0.) for o in kk.oid]); draws = []
            g_ = kk[CELL].astype(str).agg('|'.join, axis=1).values
            for _ in range(400):
                m_ = np.zeros(len(kk), bool)
                for gk in np.unique(g_):
                    ii = np.flatnonzero(g_ == gk); kc = int(kk.sel.values[ii].sum())
                    if kc:
                        m_[rng.choice(ii, kc, replace=False)] = True
                draws.append(vv[m_].sum())
            se = float(np.std(draws) / np.sqrt(len(draws))); ex = nul[B_NAME]['MODEL-J']
            if not (abs(float(np.mean(draws)) - ex) <= 4. * se + 1e-9 and se > 0):
                bad.append(f'the exact matched null {ex:,.3f} vs the mean of 400 matched draws {float(np.mean(draws)):,.3f} (se {se:,.3f})')
            Jp = json.loads((tmp / 'e' / f'entry_{d0}.json').read_text()); n_off = int((Opx.oid.astype(int) % 7 == 0).sum())
            w_ = next(x for x in Jp['tables']['windows'] if x['entry'] == 'POST-ALL' and x['side'] == 'both' and x['cancel'] == 'KEEP')
            if not (all(close(w_[f'w{w}'], bw[w]) for w in range(len(WINDOWS))) and close(Jp['no_young'][B_NAME], bny)):
                bad.append(f"by a per-fill loop: the windows {[w_[f'w{w}'] for w in range(len(WINDOWS))]} vs {bw.tolist()}, no-young {Jp['no_young'][B_NAME]:,.3f} vs {bny:,.3f}")
            Dq = Db.set_index('oid'); off_g1 = []; A_ = float(SC.B0['A']); H_ = float(KP.H)            # G1 at B by a per-order loop: end, cause, shares
            for o in ref.index.astype(int):
                if bool(ref.refused.loc[o]):
                    continue
                tf = float(ref.t_full.loc[o]) if np.isfinite(float(ref.t_full.loc[o])) else np.inf; ld = float(ref.land.loc[o]) if np.isfinite(float(ref.land.loc[o])) else np.inf
                T = want_t.get((d0, o)); ex = T is not None and KP.J0 <= T < endo[o]
                for c, rc in (('KEEP', np.inf), ('MODEL-J', KP.STEP * T + SC.B0['L'] if ex else np.inf), ('PULL-ALL', KP.STEP * KP.J0 + SC.B0['L'] if KP.J0 < endo[o] else np.inf)):
                    cand = [tf if (tf < rc and tf < ld and tf <= H_) else np.inf, ld, rc, H_]; e_ = min(cand)
                    if not (abs(float(Dq.loc[o, f'end_{c}']) - e_) < 1e-9 and int(Dq.loc[o, f'cause_{c}']) == cand.index(e_) and abs(float(Dq.loc[o, f'sh_{c}']) - bsh[c].get(o, 0.)) < 1e-9):
                        off_g1.append((o, c))
            Jg = json.loads((tmp / 'e' / f'entry_{d0}.json').read_text())['g1']; npb = {s_: int((Op.side == s_).sum()) for s_ in (0, 1)}
            live_b = {s_: sum(max(float(Dq.loc[o, 'end_MODEL-J']) - A_, 0.) for (d_, o) in want_e if o in Dq.index and int(Dq.loc[o, 'side']) == s_) for s_ in (0, 1)}
            liv0 = [o for o in Dq.index if endo[o] > KP.J0]; fr = [o for o in liv0 if want_t.get((d0, o)) == KP.J0]
            chk_g1 = dict(per_order=not off_g1, any_resting=all(close(Jg['E1'][sd]['availability']['any_resting'], live_b[s_] / (npb[s_] * SLOT_MS)) for sd, s_ in (('bid', 0), ('ask', 1))),
                          reversals=Jg['A1']['both']['reversals']['first_review_pulls'] == len(fr) and Jg['A1']['both']['reversals']['live_at_first_review'] == len(liv0),
                          attempts=Jg['E1']['both']['participation']['attempts'] == len(want_e) and Jg['A0']['both']['participation']['attempts'] == len(Op),
                          tail=close(Jg['E1']['both']['tail']['mean'] * Jg['E1']['both']['tail']['episodes'], Jg['E1']['both']['value']) and Jg['E1']['both']['tail']['episodes'] == len(want_e)
                          and Jg['A0']['both']['tail']['episodes'] == len(Op))
            if not all(chk_g1.values()):
                bad.append(f'G1 by a per-order loop: failed {[k_ for k_, v in chk_g1.items() if not v]}; per-order mismatches {off_g1[:5]}')
            cv = pd.read_csv(tmp / 'e' / f'entry_{d0}.csv'); cs = lambda r_, c_, k_='V': cv[(cv.entry == r_) & (cv.cancel == c_) & (cv.side == 'both')].set_index('scenario')[k_]
            mm = lambda x: (float(x.min()), float(x.max())); n5 = cs('MODEL-POST', 'MODEL-J') - cs('POST-ALL', 'MODEL-J')
            again = {'N1 V(MODEL-POST, MODEL-J) vs NO-ENTRY': V.above(*mm(cs('MODEL-POST', 'MODEL-J')), 'NO-ENTRY'), 'N5 V(MODEL-POST, MODEL-J) - V(POST-ALL, MODEL-J)': V.above(*mm(n5), 'POST-ALL'),
                     'N7 E_entry UNDER MODEL-J (vs the matched null)': SC.stable(*mm(cs('MODEL-POST', 'MODEL-J', 'E_entry'))),
                     'N2 V(MODEL-POST, PULL-ALL) vs NO-ENTRY': V.above(*mm(cs('MODEL-POST', 'PULL-ALL')), 'NO-ENTRY')}
            rng_ = {'N1 V(MODEL-POST, MODEL-J) vs NO-ENTRY': mm(cs('MODEL-POST', 'MODEL-J')), 'N5 V(MODEL-POST, MODEL-J) - V(POST-ALL, MODEL-J)': mm(n5),
                    'N7 E_entry UNDER MODEL-J (vs the matched null)': mm(cs('MODEL-POST', 'MODEL-J', 'E_entry')), 'N2 V(MODEL-POST, PULL-ALL) vs NO-ENTRY': mm(cs('MODEL-POST', 'PULL-ALL'))}
            if any(R[k_] != v for k_, v in again.items()) or not all(close(Jp['reading_ranges'][k_][0], v[0]) and close(Jp['reading_ranges'][k_][1], v[1]) for k_, v in rng_.items()) \
                    or not all(close(x['error_edge_term'] + x['error_move_term'], x['error_total']) for x in Jp['tables']['calibration'] if x['n']):
                bad.append(f'readings recomputed from the CSV {again} vs {R}, or the calibration error split does not add up')
            chk = dict(controls=R['CONTROLS'] == 'YES', posted=Jp['posted'] == len(want_e),
                       readings=all(k_ in R for k_ in ('N1 V(MODEL-POST, MODEL-J) vs NO-ENTRY', 'N7 E_entry UNDER MODEL-J (vs the matched null)', 'C1 N1 V AT B, B1a-WEIGHTED BY AGE', 'C2 N1 V AT M250 vs M50')),
                       refused=Jp['parity']['refused'] == n_off + 1 > 1, verdicts=Jp['parity']['verdict_disagreements'] == 0, groups=len(Db) > 0 and {0, 1, 2} <= set(Db.group),
                       late=len(late) > 0 and not Db[[(d0, int(o)) in late for o in Db.oid]].act.any())
            if not all(chk.values()):
                bad.append(f"synthetic entry run: failed {[k_ for k_, v in chk.items() if not v]}; refused {Jp['parity']['refused']} of {n_off} planted, groups {sorted(set(Db.group))}")
        def refused(name, msg, **kw):
            o_ = tmp / ('z' + str(abs(hash(name)) % 100000))
            try:
                go(o_, **kw); bad.append(f'the entry run accepted {name}')
            except (SystemExit, ValueError) as e:
                if msg not in str(e):
                    bad.append(f'{name} refused by another control: {str(e)[:300]}')
            if (o_ / f'entry_{d0}.json').exists():
                bad.append(f'a refused entry run wrote output ({name})')
        post = pred.index[pred.kind == 'post']
        refused('a POST prediction for a state the export lacks', 'Q0e', pred=pd.concat([pred, pred.loc[post[:1]].assign(oid=10 ** 9)], ignore_index=True))
        refused('a missing POST prediction', 'Q0e', pred=pred.drop(post[:1]))
        lb = pred.copy(); i_ = lb.index[(lb.kind == 'post') & (lb.value_m50 != 0)][0]; lb.loc[i_, 'value_m50'] += 1.
        refused('a POST label that is not the bank\'s value at B', 'Q0e', pred=lb)
        pb = tmp / 'polb'; pb.mkdir(); Jx = json.loads((tmp / 'pol' / f'policy_{d0}.json').read_text()); Jx['values'][P.RULE][B_NAME]['score'] += 1.
        (pb / f'policy_{d0}.json').write_text(json.dumps(Jx)); refused('a changed committed policy value', 'Q0a', policy=pb)
        pc = tmp / 'polc'; pc.mkdir(); Jy = json.loads((tmp / 'pol' / f'policy_{d0}.json').read_text()); Jy['inputs']['predictions'] = 'another file'
        (pc / f'policy_{d0}.json').write_text(json.dumps(Jy)); refused('a policy output computed on another prediction file', 'Q0a', policy=pc)
        zc = pred.index[(pred.kind == 'post') & pred.J_ok.astype(bool)]
        zn = pred.copy(); zn.loc[zc[0], ['J_Eq', 'J_Eqm', 'J_V']] = [0., -50., -50.]
        refused('a zero-quantity POST forecast with E[q * move] -50 c flagged coherent', 'Q0d', pred=zn)
        zp = pred.copy(); zc2 = zp.index[(zp.kind == 'post') & zp.J_ok.astype(bool) & (zp.J_V < 0)]; zp.loc[zc2[0], ['J_Eq', 'J_Eqm', 'J_V']] = [0., 50., 50.]
        refused('a zero-quantity POST forecast with E[q * move] +50 c flagged coherent', 'Q0d', pred=zp)
        zt = pred.copy(); zt.loc[dn_[2], ['J_Eq', 'J_Eqm', 'J_V']] = [0., 5e-9, 5e-9]                  # v3.1: beyond the tolerance, flagged coherent
        refused('a POST forecast with E[q * move] 5e-9 c at E[q] = 0 flagged coherent', 'Q0d', pred=zt)
        zn2 = pred.copy(); zn2.loc[dn_[0], 'J_ok'] = True
        refused('a nonfinite POST forecast flagged coherent', 'Q0d', pred=zn2)
        for flag, val, name, msg in (('FLIP_ENTRY', True, 'the entry sign reversed', 'Q0d'), ('RAW_ENTRY', True, 'the rule posting on refused forecasts', 'Q0d'),
                                     ('LAND_SHIFT', 1., 'the MODEL-J landing moved by 1 ms', 'Q0b'), ('GROUP_ANY', True, 'every executed pull in the first-step group', 'Q0c'),
                                     ('ACT_ANY', True, 'a pull at the order\'s structural end counted as executed', 'Q0b'),
                                     ('DUP_FILL', True, 'a fill counted twice', 'Q0b'), ('IGNORE_REFUSED', True, 'post-only refused posts kept as activated', 'Q0g'),
                                     ('LAND_SHIFT', -float(SC.B0['L']), 'the MODEL-J landing at the decision', 'Q0f')):
            globals()[flag] = val
            try:
                refused(name, msg)
            finally:
                FLIP_ENTRY, RAW_ENTRY, LAND_SHIFT, GROUP_ANY, ACT_ANY, DUP_FILL, IGNORE_REFUSED = False, False, 0., False, False, False, False
        NULL_GLOBAL = True
        try:
            w3 = null_weights(k2, [True, False, True, False, False, False])
            if abs(float((w3 * np.r_[10., 0., -4., -4., -4., -4.]).sum()) - 1.) < 1e-12:
                bad.append('an unmatched (global) null passes the hand example')
        finally:
            NULL_GLOBAL = False
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print('pipeline entry selftests passed (synthetic only): the planted entry set and triggers; every B value recomputed by a per-fill loop; the exact '
          'matched null = the hand example and the mean of 400 matched draws (a global null fails the hand example); every control passes; refusals by '
          'their own control with no output: a POST prediction for a state the export lacks and a missing one, a label that is not the bank\'s value '
          '(Q0e), a changed committed policy value and a policy output on another prediction file (Q0a), zero-quantity POST forecasts with E[q * move] '
          '-50 / +50 c flagged coherent, the entry sign reversed and the rule posting on refused forecasts (Q0d), the MODEL-J landing moved (Q0b), every '
          'pull in the first-step group (Q0c), a pull at the order\'s structural end counted as executed (Q0b); the windows and the no-young value by the '
          'per-fill loop; readings and their numeric ranges recomputed from the CSV; G1: every opportunity\'s end, cause and shares under each cancellation rule, '
          'the availability, first-review reversals, attempts and tail mean recomputed by a per-order loop; a duplicated fill (Q0b), refused posts kept (Q0g) and '
          'the MODEL-J landing at the decision (Q0f) refused; v3.1 parity: a POST forecast at the remaining-size clip and a 5e-10 c move at E[q] = 0 accepted, '
          '5e-9 c and a nonfinite forecast flagged coherent refused (Q0d)', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'day':
        run(a[1])
    else:
        sys.exit('usage: pf_pipeline_entry.py selftest | day <day>')
