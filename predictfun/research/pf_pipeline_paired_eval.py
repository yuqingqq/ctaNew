"""THE PAIRED CANDIDATE AND ITS CONTROLS THROUGH THE SCENARIO BANK (plan 0c3d2523 steps 3-4, single-order episodes; 2026-11-19, session aea5bfb9).
USER: "Build the new model and get the numbers". Declared before any output of this module is read. Evaluation days Oct 26 and Oct 27 (consumed:
development evidence, not validation); the predictions are pf_pipeline_paired's (pi0 fitted on Oct 23, candidates on Oct 24-25).

POLICIES (entry only at the original opportunities; cancellation at the first eligible 50 ms step - from +100 ms - whose rule says pull; a step
 without a usable forecast keeps; decisions fixed across the 19 timing scenarios: a frozen-trigger sensitivity, the plan's separately named control;
 one order per episode, no re-entry before the common END, no inventory cap):
 N0      never enter (0)
 PA      every opportunity, PULL-ALL (the first live step)
 A       the primary candidate: POST iff its predicted Delta_post > 0; PULL at the first step whose predicted Delta_keep < 0 (a tie or a nonfinite
         prediction waits / keeps, counted)                                                        A-PA: A's admissions with PULL-ALL
 AF, AM  the diagnostic controls' heads, same rule                                                 AF-PA, AM-PA
 PI0     pi0: the Oct 23 J, E1 rule (coherent J POST > 0; the first coherent J RETAIN < 0)         PI0-PA
 NEU     the neutral-target J fitted on Oct 24-25 (the candidate's history), E1 rule               NEU-PA
 E1      the benchmark v3 J (Oct 23-25), E1 rule - the current neutral-target rule of record       E1-PA
VALUES: the bank's own valuation (pf_keep_pull_scenarios.outcome on each policy's admissions with its triggers); a second, per-opportunity path
 (pf_pipeline_entry.per_order) must reproduce every value and executed pull count in every scenario. Local marks at END, cents, not P&L.
READINGS (computed):
 ECONOMIC GATE (the plan's; at B, each day; both sides summed): A > N0, A > A-PA (the same admissions with PULL-ALL) and A > PI0 -> PASS when
   all hold on both days; the same predicates per side are reported (a negative side is not removed)
 MATCHED CONTROL (the plan's prediction gate, economic part): A's value at B above the exact matched null of its admissions (k / n per (day,
   market window, side, minute) cell, under A's own cancellation) on each day
 PRACTICAL GATE (the plan's; at B, A vs A-PA on the same admissions): both-side at-best availability higher, the median live lifetime higher,
   first-review reversals (of the orders live there) lower and policy cancellations per 100 filled shares lower, with the economic gate PASS and
   >= 200 admitted orders per day and side (else INSUFFICIENT-SUPPORT)
 ACROSS THE 19 SCENARIOS: A - N0, A - A-PA, A - PI0, A - NEU, A - E1: ABOVE- / BELOW- / NOT-STABLE-IN-TESTED-SCENARIOS; C2: A's value at M250
   vs M50; C1: A's value at B, B1a-weighted by age (FINDINGS); the paired module's PREDICTION GATE is carried
PRINTED at B per policy and side: value (total and per original opportunity; vs N0, vs the same-admission PULL-ALL, vs PI0), participation,
 availability (resting; at the best on the 50 ms grid; either / both sides), lifetime p10 / p50 / p90 and causes, first-review reversals, policy
 cancellations per hour and per 100 filled shares, tail (P(loss), P(< -30 c), p95 / p99 loss), and the reached-state shift (the RETAIN rows A's
 rule reaches vs the neutral rows: counts, mean predicted and realized Delta_keep).
CONTROLS (any failure -> no output; a failed run names every failing control):
 R0a provenance and coverage: the paired predictions match the paired receipt and the benchmark v3 predictions theirs; together they hold one
     POST row per opportunity with a path and one RETAIN row per eligible exported step, nothing else
 R0b the per-opportunity path reproduces the bank's KEEP / policy / PULL-ALL values and executed pulls, every policy, every scenario
 R0c the paired labels reconcile with the bank: at B the sum of the POST labels (each order's value under pi0's cancellation) equals the bank's
     value of every opportunity under pi0's triggers
 R0d branch identity: in every scenario each opportunity's fills before the first decision and before the earliest effect are worth the same
     under every policy's cancellation
 R0e refusal accounting: per scenario and admission set, admitted = activated + post-only refused, activated = the bank's eligible
 R0f every policy's executed pull steps and admissions recomputed by a per-order loop over its predictions
LIMITS: two consumed days (G = 2, point values, no interval); single-order episodes (no joint bid / ask replay, shared inventory or re-post: not a
 two-sided quoting claim); frozen decisions across scenarios; queue rule M (C5); no order-level validation (C10); markouts at +100 ms / +1 s not
 computed (no mid is stored per fill).
SELFTEST (synthetic tape, no data): planted predictions give the planted admissions and pull steps; the run passes every control; the B values of
 A, A-PA and PI0 recomputed by a per-fill loop; the readings recomputed from the CSV; refusals by their own control with no output: a missing
 prediction (R0a), the per-opportunity landing moved (R0b), a POST label that is not the order's value under pi0 (R0c), the landing at the decision
 (R0d), refused posts kept (R0e), the last instead of the first pull step (R0f). Mutations: pf_pipeline_paired_eval_mutants.py.
v2 (review 0026ab85 / plan v2 stage R0, declared before v2 ran): the predictions are pf_pipeline_paired v2's (the v1 models reused, not
 refitted); every value must reproduce v1 (compare script). The reached-state shift counts only A's own admitted orders (up to the first pull);
 the all-posted counterfactual is reported under its own name. This restricted experiment's POST alternative is NO_ENTRY to END: its zero
 is not evidence about an adaptive WAIT or later entry, which no policy here can do.
Usage (run_light, PF_SLOT=claude): pf_pipeline_paired_eval.py selftest | day <day>
"""
import contextlib
import gc
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_fv3_check as FV                                # noqa: E402
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_scenarios as SC                       # noqa: E402
import pf_keep_pull_variants as V                         # noqa: E402
import pf_pipeline_entry as E                             # noqa: E402  per_order / bank_values / null_weights (imported)
import pf_pipeline_paired as PQ                           # noqa: E402
import pf_pipeline_policy as P                            # noqa: E402
import pf_queue_place_test as QP                          # noqa: E402
import pf_queue_place_ledger as QL                        # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'pipeline_paired_eval_v2_20261119'   # v2: review 0026ab85 (R0); v1 stays as provenance
MARK = 'PIPELINE PAIRED EVAL'
B_NAME, M250_NAME = V.B_NAME, V.M250_NAME
RULES = {'A': ('A_V', None), 'AF': ('AF_V', None), 'AM': ('AM_V', None), 'PI0': ('pi0_V', 'pi0_ok'), 'NEU': ('NEU_V', 'NEU_ok'), 'E1': ('J_V', 'J_ok')}
PRED_COLS = ['day', 'oid', 'kind', 'j', 'side', 'status', 'pi0_V', 'pi0_ok', 'NEU_V', 'NEU_ok', 'A_V', 'AF_V', 'AM_V', 'pair_v', 'value_m50']
SUPPORT = 200                                             # the plan's engineering floor of distinct admitted orders per day and side
LAST_PULL = False                                         # planted-error switch (selftest only): the last instead of the first pull step
CAUSE_POLICY = 2                                          # pf_pipeline_entry.CAUSES: 0 full fill, 1 structural, 2 policy cancel, 3 END


def usable(df, rule):
    """A rule's usable forecast per row: finite, and coherent where the rule's model has a coherence flag."""
    v, ok = RULES[rule]; x = df[v].values.astype(float)
    return np.isfinite(x) & (df[ok].values.astype(bool) if ok else True)


def admissions(post, rule):
    """The opportunities a rule posts: a usable POST forecast above 0 (a tie or an unusable one waits)."""
    m = usable(post, rule) & (post[RULES[rule][0]].values.astype(float) > 0)
    return set(zip(post.day.astype(str).values[m], post.oid.astype(int).values[m]))


def triggers(ret, rule):
    """A rule's first pull step per order: the first eligible RETAIN step with a usable forecast below 0 (none: never)."""
    m = usable(ret, rule) & (ret[RULES[rule][0]].values.astype(float) < 0); g = ret[m].groupby([ret.day.astype(str).values[m], ret.oid.astype(int).values[m]]).j
    return (g.max() if LAST_PULL else g.min()).astype(int)


def stable_vs(x, ref):
    lo, hi = min(x.values()), max(x.values()); return V.above(lo, hi, ref), (lo, hi)


def metrics(D, mask, ret_ok, A_, npath, pairs, adm_n, policy_col):
    """At B, one policy (its per-opportunity values in D[policy_col], live ends end_<rule>, causes cause_<rule>, shares sh_<rule>) restricted to its
    admissions (mask over D): participation, availability, lifetime, reversals, churn and tail, per side and both."""
    out = {}; c = policy_col
    live = np.maximum(D[f'end_{c}'].values - A_, 0.)
    for sd, s in (('both', None), ('bid', 0), ('ask', 1)):
        ms = mask & ((D.side.values == s) if s is not None else True); vals = D[c].values[ms]; sh = D[f'sh_{c}'].values[ms]; att = adm_n[sd]; acc = int(ms.sum())
        ep = np.r_[vals, np.zeros(max(att - acc, 0))]; cz = D[f'cause_{c}'].values[ms]; hrs = npath[sd] * E.SLOT_MS / 3.6e6; fsh = float(sh.sum())
        live0 = ms & (D.endj.values > KP.J0); pol_c = cz == CAUSE_POLICY
        first = (ms & D.act.values & (D['T'].values == KP.J0)) if c == 'MODEL-J' else (live0 if c == 'PULL-ALL' else np.zeros(len(D), bool))
        allp = (ms & D.act.values) if c == 'MODEL-J' else (live0 if c == 'PULL-ALL' else np.zeros(len(D), bool))
        rr = ret_ok[(ret_ok.side == s) if s is not None else np.ones(len(ret_ok), bool)]
        out[sd] = dict(value=float(vals.sum()), attempts=att, accepted=acc, filled=int((sh > 0).sum()), shares_filled=fsh, shares_submitted=float(KP.SIZE * att),
                       resting=float(live[ms].sum() / (npath[sd] * E.SLOT_MS)) if npath[sd] else None,
                       at_best=float(rr[f'ab_{c}'].sum() * KP.STEP / (npath[sd] * E.SLOT_MS)) if npath[sd] else None,
                       lifetime_p10_p50_p90=E.q3((D[f'end_{c}'].values - A_)[ms]), causes={E.CAUSES[i]: int((cz == i).sum()) for i in range(len(E.CAUSES))},
                       first_review_of_live=float(first.sum() / live0.sum()) if live0.sum() else None, first_review_of_pulls=float(first.sum() / allp.sum()) if allp.sum() else None,
                       cancels_per_hour=float(pol_c.sum() / hrs) if hrs else None, cancels_per_100_filled_shares=float(100. * pol_c.sum() / fsh) if fsh > 0 else None,
                       p_loss=float((ep < 0).mean()) if len(ep) else None, p_severe=float((ep < -KP.SEVERE).mean()) if len(ep) else None,
                       loss_p95=float(np.quantile(-ep, .95)) if len(ep) else None, loss_p99=float(np.quantile(-ep, .99)) if len(ep) else None)
    return out


def econ_readings(v, null_b):
    """The economic gate and the matched control from the values at B (v[policy][side]) and A's matched null at B (the plan's predicates)."""
    R = {f'ECONOMIC PREDICATES AT B ({sd}): A > N0 / A > A-PA / A > PI0': [v['A'][sd] > 0, v['A'][sd] > v['A-PA'][sd], v['A'][sd] > v['PI0'][sd]] for sd in ('both', 'bid', 'ask')}
    R['ECONOMIC GATE (this day, both sides)'] = 'PASS' if all(R['ECONOMIC PREDICATES AT B (both): A > N0 / A > A-PA / A > PI0']) else 'FAIL'
    R['MATCHED CONTROL AT B: A above the matched null of its admissions'] = 'PASS' if v['A']['both'] > null_b else 'FAIL'
    return R


def practical_gate(prac, econ, support_ok):
    """The plan's practical gate: every practical predicate, the economic gate, and the admitted-order support floor."""
    return 'INSUFFICIENT-SUPPORT' if not support_ok else ('PASS' if all(prac) and econ == 'PASS' else 'FAIL')


def run(day, out=None, sc_out=None, kp=None, paired=None, bench=None, export=None, pred=None, jpred=None):  # noqa: C901
    """Every policy in every scenario of the bank on one evaluation day -> eval_<day>.json / .txt / .csv."""
    start = time.time(); C.require(day in (PQ.EVAL if pred is None else KP.DAYS), 'undeclared day'); out = Path(out or OUT); sc_out = Path(sc_out or SC.OUT); kp = Path(kp or KP.OUT)
    paired = Path(paired or PQ.OUT); bench = Path(bench or P.BENCH); export = Path(export or P.EXPORT); C.require(not (out / f'eval_{day}.json').exists(), 'output exists')
    bad = {}; fail = lambda k, m: bad.setdefault(k, []).append(m)
    pj = json.loads((paired / 'paired.json').read_text()); bj = json.loads((bench / 'bench.json').read_text()); pf_, jf_ = paired / f'pred_{day}.parquet', bench / f'pred_{day}.parquet'
    if pred is None:
        C.require(pj['outputs'].get(pf_.name) == C.sha(pf_), f'R0a: {pf_.name} does not match the paired receipt'); pred = pd.read_parquet(pf_, columns=PRED_COLS)
    if jpred is None:
        C.require(bj['outputs'].get(jf_.name) == C.sha(jf_), f'R0a: {jf_.name} does not match the benchmark receipt'); jpred = pd.read_parquet(jf_, columns=['day', 'oid', 'kind', 'j', 'J_V', 'J_ok'])
    pred = pred[pred.day.astype(str) == day].assign(day=day, oid=lambda z: z.oid.astype(int), j=lambda z: z.j.astype(int))
    jpred = jpred[jpred.day.astype(str) == day].assign(day=day, oid=lambda z: z.oid.astype(int), j=lambda z: z.j.astype(int))
    pred = pred.merge(jpred, on=['day', 'oid', 'kind', 'j'], how='left', validate='one_to_one'); del jpred      # memory (v2): small types, no copies kept
    for c_ in ('day', 'kind', 'status'):
        pred[c_] = pred[c_].astype(str).astype('category')
    er = json.loads((export / day / 'receipt.json').read_text()); ex = []
    for x, h in er['outputs'].items():
        C.require(C.sha(export / day / x) == h, f'R0a: export part {x} changed since its receipt'); ex.append(pd.read_parquet(export / day / x, columns=['oid', 'kind', 'j', 'status', 'o_depth']))
    ex = pd.concat(ex, ignore_index=True); ex['oid'] = ex.oid.astype(int); ex['j'] = ex.j.astype(int); ex['kind'] = ex.kind.astype(str).astype('category')
    ex['status'] = ex.status.astype(str).astype('category')
    want = set(zip(ex.oid[(ex.kind == 'post') & ex.status.isin(('eligible', 'refused_post_only'))], ex.kind[(ex.kind == 'post') & ex.status.isin(('eligible', 'refused_post_only'))],
                   ex.j[(ex.kind == 'post') & ex.status.isin(('eligible', 'refused_post_only'))])) | set(zip(ex.oid[(ex.kind == 'retain') & (ex.status == 'eligible')], ex.kind[(ex.kind == 'retain') & (ex.status == 'eligible')], ex.j[(ex.kind == 'retain') & (ex.status == 'eligible')]))
    got = set(zip(pred.oid, pred.kind.astype(str), pred.j))
    if got != want or len(got) != len(pred) or pred.J_V.isna().any():
        fail('R0a', f'{len(want - got):,} exported rows without a prediction, {len(got - want):,} predictions for rows the export does not have, {int(pred.J_V.isna().sum()):,} without the benchmark\'s')
    post, ret = pred[pred.kind == 'post'].copy(), pred[pred.kind == 'retain'].copy(); del pred; gc.collect()
    O, Fl, S, SF, Op, rec = V.load([day], sc_out, kp); O['day'] = O.day.astype(str); Op = Op.assign(day=Op.day.astype(str))
    allk = set(zip(Op.day, Op.oid.astype(int))); adm = {'ALL': allk}; adm.update({r: admissions(post, r) for r in RULES}); hit = {r: triggers(ret, r) for r in RULES}
    fns = {r: P.trigger_fn(hit[r]) for r in RULES}
    for r in RULES:                                                                         # R0f: admissions and first pull steps by a per-order loop
        a2 = {(day, int(o)) for o, v, *ok in zip(post.oid, post[RULES[r][0]], *( [post[RULES[r][1]]] if RULES[r][1] else [])) if np.isfinite(v) and v > 0 and (not ok or bool(ok[0]))}
        h2 = {}
        ro, rj_, rv_ = ret.oid.values, ret.j.values, ret[RULES[r][0]].values.astype(float); rk = ret[RULES[r][1]].values.astype(bool) if RULES[r][1] else np.ones(len(ret), bool)
        for i in np.lexsort((rj_, ro)):                                                     # (arrays, not a list of tuples: memory)
            if (day, int(ro[i])) not in h2 and np.isfinite(rv_[i]) and rv_[i] < 0 and rk[i]:
                h2[(day, int(ro[i]))] = int(rj_[i])
        hd = {(str(k[0]), int(k[1])): int(v) for k, v in hit[r].items()}
        if a2 != adm[r] or h2 != hd:
            fail('R0f', f'{r}: {len(a2 ^ adm[r]):,} admissions and {len(set(h2.items()) ^ set(hd.items())):,} pull steps differ from a per-order loop')
    keys = Op[E.CELL].reset_index(drop=True).assign(oid=Op.oid.values.astype(int)); keys['sel'] = [(d_, int(o)) in adm['A'] for d_, o in zip(keys.day, keys.oid)]
    wi = keys.assign(w=E.null_weights(keys, keys.sel.values)).set_index(['day', 'oid']).w
    res, Db, sides, nul = {}, {}, {}, {}; pairs_ = [('ALL', 'A')] + [(r, r) for r in RULES]
    for sc in SC.SCEN:
        nm = SC.name(sc); Os, Fs = V.scen_inputs(sc, Op, Fl, S, SF); Os = Os.assign(day=Os.day.astype(str)); Fs = Fs.assign(day=Fs.day.astype(str))
        refs = set(zip(Os.day[Os.refused.astype(bool)], Os.oid[Os.refused.astype(bool)].astype(int))); Ds = {}
        for r in RULES:
            Ds[r] = E.per_order(O, Os, Fs, sc, fns[r]); Ds[r]['key'] = list(zip(Ds[r].day, Ds[r].oid))
        for w in (0, 1):                                                                    # R0d: branch identity across the policies' cancellations
            base = Ds['A'][f'KEEP_w{w}'].values
            if not all(np.allclose(Ds[r][f'MODEL-J_w{w}'].values, base, rtol=0, atol=1e-9) and np.allclose(Ds[r][f'PULL-ALL_w{w}'].values, base, rtol=0, atol=1e-9) for r in RULES):
                fail('R0d', f'{nm}: fills before the {"first decision" if w == 0 else "earliest effect"} are worth different amounts under the policies')
        for ent, rule in pairs_:
            b = E.bank_values(O, Os, Fs, sc, fns[rule], None if ent == 'ALL' else adm[ent]); D = Ds[rule]; m = np.array([k in adm[ent] for k in D.key])
            res[(ent, rule, nm)] = b
            for c in ('KEEP', 'MODEL-J', 'PULL-ALL'):                                       # R0b
                if not E.close(b[c], D[c].values[m].sum()):
                    fail('R0b', f'{nm} {ent}/{rule} {c}: the bank {b[c]:,.3f}, the per-opportunity path {D[c].values[m].sum():,.3f}')
            if int(D.act.values[m].sum()) != b['pulls']:
                fail('R0b', f"{nm} {ent}/{rule}: {int(D.act.values[m].sum()):,} executed pulls on the per-opportunity path, the bank {b['pulls']:,}")
            n_ref = len(adm[ent] & refs)                                                   # R0e
            if not (len(adm[ent]) == b['eligible'] + n_ref and int(m.sum()) == b['eligible']):
                fail('R0e', f"{nm} {ent}: {len(adm[ent]):,} admitted vs {b['eligible']:,} activated + {n_ref:,} post-only refused; {int(m.sum()):,} on the per-opportunity path")
            for sd, s in (('both', None), ('bid', 0), ('ask', 1)):
                ms = m & ((D.side.values == s) if s is not None else True)
                for c in ('KEEP', 'MODEL-J', 'PULL-ALL'):
                    sides[(ent, rule, c, sd, nm)] = float(D[c].values[ms].sum())
        DA = Ds['A']; nul[nm] = float((wi.reindex(pd.MultiIndex.from_tuples(DA.key)).values * DA['MODEL-J'].values).sum())
        if nm == B_NAME:
            Db = Ds; b0 = E.bank_values(O, Os, Fs, sc, fns['PI0'], None); lab = float(post.pair_v.sum())       # R0c
            if not E.close(b0['MODEL-J'], lab):
                fail('R0c', f"at B the POST labels sum to {lab:,.3f}, the bank's value of every opportunity under pi0's triggers is {b0['MODEL-J']:,.3f}")
        C.guard('pipeline paired eval')
    if bad:
        sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(f'{k} ({len(v)} failures; first: {v[0]})' for k, v in sorted(bad.items())))
    V_ = lambda ent, rule, c, sd='both': {nm: sides[(ent, rule, c, sd, nm)] for nm in nul}
    pol = {'N0': None, 'PA': ('ALL', 'A', 'PULL-ALL')}
    for r in RULES:
        pol[r] = (r, r, 'MODEL-J'); pol[f'{r}-PA'] = (r, r, 'PULL-ALL')
    val = lambda p_, sd='both', nm=B_NAME: 0. if pol[p_] is None else sides[(pol[p_][0], pol[p_][1], pol[p_][2], sd, nm)]
    R = {}
    for d_, ref in (('N0', 'N0'), ('A-PA', 'A-PA'), ('PI0', 'PI0'), ('NEU', 'NEU'), ('E1', 'E1')):
        x = {nm: val('A', 'both', nm) - val(ref, 'both', nm) for nm in nul}; R[f'A - {ref} ACROSS TESTED SCENARIOS'], _ = stable_vs(x, ref)
    R.update(econ_readings({p_: {sd: val(p_, sd) for sd in ('both', 'bid', 'ask')} for p_ in ('A', 'A-PA', 'PI0')}, nul[B_NAME]))
    R['C2 A AT M250 vs M50'] = SC.at_m250(val('A'), val('A', 'both', M250_NAME))
    s_ = np.array([k in adm['A'] for k in Db['A'].key]); vb = [float(Db['A'][f'MODEL-J_a{b}'].values[s_].sum()) for b in range(len(SC.AGE_LABELS))]
    R['C1 A AT B, B1a-WEIGHTED BY AGE'] = SC.b1a_sign(val('A'), {k: float(sum(w * x for w, x in zip(wt, vb))) for k, wt in FV.B1A.items()})
    R['PREDICTION GATE (carried from pf_pipeline_paired)'] = pj['readings'].get('PREDICTION GATE (A, per day and side)', {}).get(day)
    # ---- at B: the plan's metrics per policy and side
    A_ = float(SC.B0['A']); Oi = O.assign(oid=O.oid.astype(int)).set_index('oid')
    npath = {sd: int(((Op.side == s) if s is not None else np.ones(len(Op), bool)).sum()) for sd, s in (('both', None), ('bid', 0), ('ask', 1))}
    norig = {sd: int(((O.side == s) if s is not None else np.ones(len(O), bool)).sum()) for sd, s in (('both', None), ('bid', 0), ('ask', 1))}
    pairs = Op[Op.side == 0][['m', 't']].merge(Op[Op.side == 1][['m', 't']], on=['m', 't']); M = {}
    rr = ex[(ex.kind == 'retain') & (ex.status == 'eligible')][['oid', 'j', 'o_depth']]; del ex; gc.collect()
    for p_, spec in pol.items():
        if spec is None:
            continue
        ent, rule, c = spec; D = Db[rule]; mask = np.array([k in adm[ent] for k in D.key]); col = c
        adm_n = {sd: sum(1 for (_, o) in adm[ent] if sd == 'both' or int(Oi.side.loc[o]) == (0 if sd == 'bid' else 1)) for sd in npath}
        R_ = rr.merge(D[['oid', 'side', f'end_{c}']].assign(sel=mask), on='oid', how='inner'); tj = KP.STEP * R_.j.values
        R_[f'ab_{c}'] = R_.sel.values & (tj >= A_) & (tj < R_[f'end_{c}'].values) & (R_.o_depth.values.astype(float) <= 1e-9)
        x = metrics(D, mask, R_, A_, npath, pairs, adm_n, col)
        lb = pd.DataFrame(dict(m=Oi.m.reindex(D.oid.values).values, t=Oi.t.reindex(D.oid.values).values, side=D.side.values,
                               live=np.where(mask, np.maximum(D[f'end_{c}'].values - A_, 0.), 0.)))
        pl = pairs.merge(lb[lb.side == 0][['m', 't', 'live']], on=['m', 't'], how='left').merge(lb[lb.side == 1][['m', 't', 'live']], on=['m', 't'], how='left', suffixes=('_b', '_a')).fillna(0.)
        gab = R_[R_[f'ab_{c}']].assign(m=lambda z: Oi.m.reindex(z.oid.values).values, t=lambda z: Oi.t.reindex(z.oid.values).values).groupby(['m', 't', 'j']).side.nunique()
        x['pairs'] = dict(either=float(np.maximum(pl.live_b, pl.live_a).sum() / (len(pairs) * E.SLOT_MS)) if len(pairs) else None,
                          both=float(np.minimum(pl.live_b, pl.live_a).sum() / (len(pairs) * E.SLOT_MS)) if len(pairs) else None,
                          at_best_both=float((gab == 2).sum() * KP.STEP / (len(pairs) * E.SLOT_MS)) if len(pairs) else None)
        for sd in ('both', 'bid', 'ask'):
            x[sd]['per_original_opportunity'] = x[sd]['value'] / norig[sd] if norig[sd] else None; x[sd]['vs_N0'] = x[sd]['value']
            x[sd]['vs_same_admission_PULL_ALL'] = x[sd]['value'] - val(f'{ent}-PA' if ent != 'ALL' else 'PA', sd) if ent != 'ALL' else 0.
            x[sd]['vs_PI0'] = x[sd]['value'] - val('PI0', sd); x[sd]['scenario_range'] = list(stable_vs({nm: val(p_, sd, nm) for nm in nul}, 'N0')[1])
        M[p_] = x; del R_, lb, pl, gab, D; gc.collect()
    for sd, s in (('bid', 0), ('ask', 1)):
        if M['A'][sd]['attempts'] < SUPPORT:
            R[f'PRACTICAL GATE SUPPORT ({sd})'] = 'INSUFFICIENT-SUPPORT'
    a, b = M['A'], M['A-PA']; lt = lambda z: (z['lifetime_p10_p50_p90'] or [None, None, None])[1]
    prac = [a['pairs']['at_best_both'] is not None and b['pairs']['at_best_both'] is not None and a['pairs']['at_best_both'] > b['pairs']['at_best_both'],
            lt(a['both']) is not None and lt(b['both']) is not None and lt(a['both']) > lt(b['both']),
            a['both']['first_review_of_live'] is not None and b['both']['first_review_of_live'] is not None and a['both']['first_review_of_live'] < b['both']['first_review_of_live'],
            a['both']['cancels_per_100_filled_shares'] is not None and b['both']['cancels_per_100_filled_shares'] is not None
            and a['both']['cancels_per_100_filled_shares'] < b['both']['cancels_per_100_filled_shares']]
    R['PRACTICAL PREDICATES AT B (A vs A-PA): at-best both sides / median lifetime / first-review reversals / cancels per filled share'] = prac
    R['PRACTICAL GATE (this day)'] = practical_gate(prac, R['ECONOMIC GATE (this day, both sides)'], not any(k.startswith('PRACTICAL GATE SUPPORT') for k in R))
    R['CONTROLS'] = 'YES'
    ra = ret; kix = pd.MultiIndex.from_arrays([ra.day.astype(str).values, ra.oid.astype(int).values])
    fa = pd.Series({(str(k[0]), int(k[1])): int(v) for k, v in hit['A'].items()}, dtype=float).reindex(kix).fillna(10 ** 6).values if len(hit['A']) else np.full(len(ra), 1e6)
    reach_all = ra.j.values <= fa; reach = reach_all & kix.isin(pd.MultiIndex.from_tuples(sorted(adm['A']))) if adm['A'] else np.zeros(len(ra), bool)   # A's own admitted orders (v2)
    shift = dict(neutral_rows=int(len(ra)), reached_rows=int(reach.sum()), reached_rows_if_all_posted=int(reach_all.sum()), neutral_mean_pred=float(ra.A_V.mean()), reached_mean_pred=float(ra.A_V.values[reach].mean()) if reach.any() else None,
                 neutral_mean_label=float(ra.pair_v.mean()), reached_mean_label=float(ra.pair_v.values[reach].mean()) if reach.any() else None)
    rows = [dict(policy=p_, side=sd, scenario=nm, V=val(p_, sd, nm), null_A=nul[nm] if (p_ == 'A' and sd == 'both') else None) for p_ in pol for sd in ('both', 'bid', 'ask') for nm in nul]
    pc_ = lambda z: 'n/a' if z is None else f'{100 * z:.2f}%'; nz = lambda z, k=1: 'n/a' if z is None else f'{z:,.{k}f}'
    L = [f'THE PAIRED CANDIDATE AND ITS CONTROLS THROUGH THE BANK, RESTRICTED EXPERIMENT: POST vs NO_ENTRY to END (pf_pipeline_paired_eval.py v2; {day}; c; local marks at END, not P&L; single-order episodes; decisions '
         f'fixed across the 19 scenarios); CONTROLS: YES (R0a-R0f)', f"  {len(O):,} original opportunities, {npath['both']:,} with a path; admissions: "
         + ', '.join(f'{k} {len(v):,}' for k, v in adm.items()) + '; orders with a pull step: ' + ', '.join(f'{r} {len(hit[r]):,}' for r in RULES), '',
         f"  {'scenario':<24} " + ' '.join(f'{p_:>9}' for p_ in pol if p_ != 'N0') + f" {'null A':>9}"]
    for nm in nul:
        L.append(f'  {nm:<24} ' + ' '.join(f'{val(p_, "both", nm):>+9,.0f}' for p_ in pol if p_ != 'N0') + f' {nul[nm]:>+9,.0f}')
    L += ['', '  AT B PER POLICY AND SIDE']
    for p_, x in M.items():
        for sd in ('both', 'bid', 'ask'):
            y = x[sd]
            L.append(f"    {p_:<6} {sd:<4} value {y['value']:>+10,.1f} ({nz(y['per_original_opportunity'], 4)} / opp; vs same-admission PULL-ALL {y['vs_same_admission_PULL_ALL']:+,.1f}; vs PI0 "
                     f"{y['vs_PI0']:+,.1f}; scenarios {y['scenario_range'][0]:+,.0f} .. {y['scenario_range'][1]:+,.0f}) | attempts {y['attempts']:,} accepted {y['accepted']:,} filled {y['filled']:,} "
                     f"| resting {pc_(y['resting'])} at best {pc_(y['at_best'])} | lifetime {y['lifetime_p10_p50_p90'] and '/'.join(f'{v:.0f}' for v in y['lifetime_p10_p50_p90'])} ms "
                     f"{y['causes']} | first review {pc_(y['first_review_of_live'])} of live, {pc_(y['first_review_of_pulls'])} of pulls | cancels {nz(y['cancels_per_hour'])}/h, "
                     f"{nz(y['cancels_per_100_filled_shares'], 2)} per 100 sh | loss P {pc_(y['p_loss'])} severe P {pc_(y['p_severe'])} p95/p99 {nz(y['loss_p95'], 2)}/{nz(y['loss_p99'], 2)}")
        L.append(f"    {p_:<6} pairs: either {pc_(x['pairs']['either'])}, both {pc_(x['pairs']['both'])}, at best both {pc_(x['pairs']['at_best_both'])}")
    L += ['', f"  REACHED-STATE SHIFT (A): {shift}", '', 'READINGS (declared before this module\'s outputs were read)'] + [f'  {k}: {v}' for k, v in R.items()]
    L += ['', 'LIMITS: consumed day (G = 1 per output); single-order episodes (no joint bid / ask replay, inventory or re-post); frozen decisions across scenarios; local marks, '
          'not P&L; queue rule M (C5); no order-level validation (C10); at-best time on the 50 ms grid; markouts at +100 ms / +1 s not computed.']
    out.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out / f'eval_{day}.csv', index=False); (out / f'eval_{day}.txt').write_text('\n'.join(L) + '\n')
    C.dump(out / f'eval_{day}.json', dict(asof=C.stamp(), day=day, readings=R, metrics=M, null_A=nul, shift=shift, admissions={k: len(v) for k, v in adm.items()},
                                          pulled={r: int(len(hit[r])) for r in RULES}, values={p_: {sd: {nm: val(p_, sd, nm) for nm in nul} for sd in ('both', 'bid', 'ask')} for p_ in pol},
                                          inputs=dict(paired=C.sha(paired / 'paired.json'), predictions=C.sha(pf_) if pf_.exists() else None, bench=C.sha(bench / 'bench.json'),
                                                      bench_predictions=C.sha(jf_) if jf_.exists() else None, export_receipt=C.sha(export / day / 'receipt.json'), days=rec),
                                          outputs={x: C.sha(out / x) for x in (f'eval_{day}.txt', f'eval_{day}.csv')},
                                          source_sha256=C.sha(Path(__file__).resolve()), provenance=KP.provenance(), seconds=time.time() - start))
    print('\n'.join(L), flush=True)
    return R, M, sides, nul


def selftest():  # noqa: C901
    """Synthetic tape only (see the module's SELFTEST)."""
    global LAST_PULL
    bad = []; src = KP.Syn(); days = list(KP.DAYS); d0 = days[0]
    v_ = lambda a, apa, pi0, pa=None: {'A': dict(both=a, bid=a, ask=a), 'A-PA': dict(both=apa, bid=apa, ask=apa), 'PI0': dict(both=pi0, bid=pi0, ask=pi0)}
    cases = [(v_(5., 3., 4.), 2., 'PASS', 'PASS'), (v_(5., 6., 4.), 2., 'FAIL', 'PASS'), (v_(5., 3., 6.), 2., 'FAIL', 'PASS'), (v_(-5., -9., -9.), -10., 'FAIL', 'PASS'),
             (v_(5., 3., 4.), 7., 'PASS', 'FAIL')]                                              # (values, A's null, economic gate, matched control)
    got = [(econ_readings(v, nb)['ECONOMIC GATE (this day, both sides)'], econ_readings(v, nb)['MATCHED CONTROL AT B: A above the matched null of its admissions']) for v, nb, _, _ in cases]
    pg = (practical_gate([True] * 4, 'PASS', True), practical_gate([True] * 4, 'FAIL', True), practical_gate([True, False, True, True], 'PASS', True), practical_gate([True] * 4, 'PASS', False))
    if got != [(e, m) for _, _, e, m in cases] or pg != ('PASS', 'FAIL', 'FAIL', 'INSUFFICIENT-SUPPORT'):
        bad.append(f'the gate readings on planted values: {got} / {pg}')
    Dm = pd.DataFrame({'side': [0, 0, 1, 1], 'endj': [50, 1, 50, 50], 'act': [True, False, True, False], 'T': [KP.J0, 99, 10, 99], 'MODEL-J': [-3., 0., 2., 4.],
                       'end_MODEL-J': [137., 80., 537., 5000.], 'cause_MODEL-J': [2, 1, 2, 3], 'sh_MODEL-J': [5., 0., 10., 0.]})   # one order dead before +100 ms
    rk = pd.DataFrame({'side': [0, 1], 'ab_MODEL-J': [True, True]})
    mm = metrics(Dm, np.ones(4, bool), rk, 58., dict(both=4, bid=2, ask=2), None, dict(both=5, bid=2, ask=3), 'MODEL-J')['both']
    want_m = dict(first_review_of_live=1 / 3, first_review_of_pulls=1 / 2, value=3., attempts=5, accepted=4, filled=2, resting=(79. + 22. + 479. + 4942.) / (4 * E.SLOT_MS),
                  cancels_per_100_filled_shares=100. * 2 / 15., p_loss=1 / 5)
    if any(abs(mm[k] - v) > 1e-12 for k, v in want_m.items()):
        bad.append(f'metrics() on a planted frame: { {k: mm[k] for k in want_m} } vs {want_m}')
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
        O = pd.read_parquet(tmp / 'kp' / d0 / 'orders.parquet'); el = O[O.status == 'eligible']; Opx = O[O.status.isin(('eligible', 'refused_post_only'))]
        endj = SC.A.endj_of(el.t_full.values, el.g_struct.values); rng = np.random.default_rng(9); rows = []; want_a, want_t = set(), {r: {} for r in RULES}
        for o, e in zip(el.itertuples(), endj):                                              # planted RETAIN forecasts: each rule pulls some orders at a chosen step
            js = np.arange(KP.J0, max(int(e), KP.J0)); oid = int(o.oid); vals = {}
            for i_, r in enumerate(RULES):
                k = js[(oid + i_) % len(js)] if (len(js) and (oid + i_) % 3 == 0) else None
                vals[r] = [(-1. if (k is not None and j >= k) else 1.) for j in js]
                if k is not None:
                    want_t[r][(d0, oid)] = int(k)
            for n_, j in enumerate(js):
                rows.append(dict(day=d0, oid=oid, kind='retain', j=int(j), side=int(o.side), status='eligible', pi0_V=vals['PI0'][n_], pi0_ok=True, NEU_V=vals['NEU'][n_], NEU_ok=True,
                                 A_V=vals['A'][n_], AF_V=vals['AF'][n_], AM_V=vals['AM'][n_], J_V=vals['E1'][n_], J_ok=True, pair_v=0., value_m50=0., o_depth=float(rng.integers(0, 2))))
        for o in Opx.itertuples():                                                           # planted POST forecasts: A posts every other opportunity
            oid = int(o.oid); a = 1. if oid % 2 == 0 else -1.
            rows.append(dict(day=d0, oid=oid, kind='post', j=0, side=int(o.side), status=str(o.status), pi0_V=1. if oid % 3 else -1., pi0_ok=True, NEU_V=-a, NEU_ok=True, A_V=a,
                             AF_V=1., AM_V=-1., J_V=a, J_ok=oid % 5 != 0, pair_v=0., value_m50=0., o_depth=0.))
            if a > 0:
                want_a.add((d0, oid))
        pred = pd.DataFrame(rows); O_, Fl, S, SF, Op, _ = V.load([d0], tmp / 'sc', tmp / 'kp'); Os, Fs = V.scen_inputs(SC.B0, Op, Fl, S, SF)
        D0 = E.per_order(O_.assign(day=O_.day.astype(str)), Os.assign(day=Os.day.astype(str)), Fs.assign(day=Fs.day.astype(str)), SC.B0, P.trigger_fn(triggers(pred[pred.kind == 'retain'], 'PI0')))
        pv = dict(zip(D0.oid.astype(int), D0['MODEL-J'].values)); pm = pred.kind == 'post'                       # the POST label = the order's value under pi0 (R0c)
        pred.loc[pm, 'pair_v'] = [pv.get(int(o), 0.) for o in pred.oid[pm]]
        exp = tmp / 'exp' / d0; exp.mkdir(parents=True); pred[['oid', 'kind', 'j', 'status', 'o_depth']].to_parquet(exp / 'rows_000.parquet', index=False)
        C.dump(exp / 'receipt.json', dict(outputs={'rows_000.parquet': C.sha(exp / 'rows_000.parquet')}))
        pq_ = tmp / 'pq'; pq_.mkdir(); C.dump(pq_ / 'paired.json', dict(outputs={}, readings={'PREDICTION GATE (A, per day and side)': {d0: {'0': 'PASS', '1': 'PASS'}}}))
        bj = tmp / 'bench'; bj.mkdir(); C.dump(bj / 'bench.json', dict(outputs={}))
        jp = pred[['day', 'oid', 'kind', 'j', 'J_V', 'J_ok']]; pp = pred.drop(columns=['J_V', 'J_ok', 'o_depth'])
        if not (admissions(pp[pp.kind == 'post'], 'A') == want_a and dict(triggers(pp[pp.kind == 'retain'], 'A').items()) == want_t['A']):
            bad.append('planted admissions / pull steps of A')
        def go(out_, **kw):
            with contextlib.redirect_stdout(io.StringIO()):
                return run(d0, out=out_, sc_out=tmp / 'sc', kp=tmp / 'kp', paired=pq_, bench=bj, export=tmp / 'exp', pred=kw.pop('pred', pp), jpred=kw.pop('jpred', jp))
        R = None
        try:
            R, M, sides, nul = go(tmp / 'e')
        except (SystemExit, ValueError) as e:
            bad.append(f'the synthetic run refused: {str(e)[:400]}')
        if R is not None:
            ref = Os.set_index('oid'); Mo = O_.set_index('oid'); en = SC.endj_n(ref.t_full.values, Mo.g_struct.reindex(ref.index).values, SC.B0['N']); endo = dict(zip(ref.index.astype(int), en))
            br = {k: 0. for k in ('A', 'A-PA', 'PI0')}
            for f in Fs.itertuples():                                                         # B by a per-fill loop
                o = int(f.oid)
                if bool(ref.refused.loc[o]):
                    continue
                v = float(f.q) * (1. if Mo.side.loc[o] == 0 else -1.) * (float(Mo.mark.loc[o]) - float(Mo.p.loc[o]))
                for k_, rule, pa in (('A', 'A', False), ('A-PA', 'A', True), ('PI0', 'PI0', False)):
                    admitted = (d0, o) in (want_a if rule == 'A' else admissions(pp[pp.kind == 'post'], 'PI0'))
                    T = KP.J0 if pa else want_t[rule].get((d0, o)); gone = T is not None and KP.J0 <= T < endo[o] and float(f.t) >= KP.STEP * T + SC.B0['L']
                    br[k_] += v if (admitted and not gone) else 0.
            vA = {}                                                                           # every eligible order's value under A's cancellation
            for f in Fs.itertuples():
                o = int(f.oid)
                if bool(ref.refused.loc[o]):
                    continue
                T = want_t['A'].get((d0, o)); gone = T is not None and KP.J0 <= T < endo[o] and float(f.t) >= KP.STEP * T + SC.B0['L']
                vA[o] = vA.get(o, 0.) + (0. if gone else float(f.q) * (1. if Mo.side.loc[o] == 0 else -1.) * (float(Mo.mark.loc[o]) - float(Mo.p.loc[o])))
            cells = Op.assign(oid=Op.oid.astype(int)).groupby(['m', 'side', 'phase']).oid.apply(list); null_ = 0.
            for os_ in cells:
                k = sum((d0, o) in want_a for o in os_); null_ += k / len(os_) * sum(vA.get(o, 0.) for o in os_)
            if not E.close(null_, nul[B_NAME]):
                bad.append(f'the matched null recomputed from its cells {null_:,.3f} vs the run {nul[B_NAME]:,.3f}')
            got = {k_: M[k_]['both']['value'] for k_ in br}
            if not all(E.close(br[k_], got[k_]) for k_ in br):
                bad.append(f'B by a per-fill loop {br} vs the run {got}')
            sh_ = json.loads((tmp / 'e' / f'eval_{d0}.json').read_text())['shift']; n_ = 0                      # v2: the shift over A's admitted orders, by a loop
            for o, g_ in pp[pp.kind == 'retain'].sort_values(['oid', 'j']).groupby('oid'):
                if (d0, int(o)) in want_a:
                    for v_ in g_.A_V.values:
                        n_ += 1
                        if v_ < 0:
                            break
            if sh_['reached_rows'] != n_ or not sh_['reached_rows_if_all_posted'] >= n_:
                bad.append(f"the reached-state shift: {sh_['reached_rows']:,} rows, a loop over A's admitted orders {n_:,}")
            cv = pd.read_csv(tmp / 'e' / f'eval_{d0}.csv'); g = lambda p_: cv[(cv.policy == p_) & (cv.side == 'both')].set_index('scenario').V
            again = {'A - N0 ACROSS TESTED SCENARIOS': V.above(float(g('A').min()), float(g('A').max()), 'N0'),
                     'A - PI0 ACROSS TESTED SCENARIOS': V.above(float((g('A') - g('PI0')).min()), float((g('A') - g('PI0')).max()), 'PI0')}
            B_ = lambda p_, sd='both': float(cv[(cv.policy == p_) & (cv.side == sd)].set_index('scenario').V.loc[B_NAME])
            nA = float(cv[(cv.policy == 'A') & (cv.side == 'both')].set_index('scenario').null_A.loc[B_NAME])
            again['MATCHED CONTROL AT B: A above the matched null of its admissions'] = 'PASS' if B_('A') > nA else 'FAIL'
            for sd in ('both', 'bid', 'ask'):
                again[f'ECONOMIC PREDICATES AT B ({sd}): A > N0 / A > A-PA / A > PI0'] = [B_('A', sd) > 0, B_('A', sd) > B_('A-PA', sd), B_('A', sd) > B_('PI0', sd)]
            a_, b_ = M['A'], M['A-PA']; lt = lambda z: z['lifetime_p10_p50_p90'][1]
            prac = [a_['pairs']['at_best_both'] > b_['pairs']['at_best_both'], lt(a_['both']) > lt(b_['both']), a_['both']['first_review_of_live'] < b_['both']['first_review_of_live'],
                    a_['both']['cancels_per_100_filled_shares'] is not None and b_['both']['cancels_per_100_filled_shares'] is not None
                    and a_['both']['cancels_per_100_filled_shares'] < b_['both']['cancels_per_100_filled_shares']]
            again['PRACTICAL PREDICATES AT B (A vs A-PA): at-best both sides / median lifetime / first-review reversals / cancels per filled share'] = prac
            econ = 'PASS' if all(again['ECONOMIC PREDICATES AT B (both): A > N0 / A > A-PA / A > PI0']) else 'FAIL'
            sup = any(M['A'][sd]['attempts'] < SUPPORT for sd in ('bid', 'ask'))
            again['PRACTICAL GATE (this day)'] = 'INSUFFICIENT-SUPPORT' if sup else ('PASS' if all(prac) and econ == 'PASS' else 'FAIL')
            liv = [o for (d_, o) in want_a if o in endo and not bool(ref.refused.loc[o]) and endo[o] > KP.J0]; fr = [o for o in liv if want_t['A'].get((d0, o)) == KP.J0]
            if any(R[k_] != v for k_, v in again.items()) or R['CONTROLS'] != 'YES' or R['ECONOMIC GATE (this day, both sides)'] != econ or not (
                    liv and abs(M['A']['both']['first_review_of_live'] - len(fr) / len(liv)) < 1e-12):
                bad.append(f'readings recomputed from the CSV / metrics {again} vs {R}; first review {M["A"]["both"]["first_review_of_live"]} vs {len(fr)} / {len(liv)}')
        def refused(name, msg, **kw):
            o_ = tmp / ('z' + str(abs(hash(name)) % 100000))
            try:
                go(o_, **kw); bad.append(f'the run accepted {name}')
            except (SystemExit, ValueError) as e:
                if msg not in str(e):
                    bad.append(f'{name} refused by another control: {str(e)[:300]}')
            if (o_ / f'eval_{d0}.json').exists():
                bad.append(f'a refused run wrote output ({name})')
        refused('a missing prediction', 'R0a', pred=pp.drop(pp.index[pp.kind == 'retain'][:1]), jpred=jp.drop(jp.index[jp.kind == 'retain'][:1]))
        lb = pp.copy(); i_ = lb.index[(lb.kind == 'post') & (lb.pair_v != 0)][0]; lb.loc[i_, 'pair_v'] += 1.
        refused('a POST label that is not the order\'s value under pi0', 'R0c', pred=lb)
        for mod, flag, val_, name, msg in ((E, 'LAND_SHIFT', 1., 'the per-opportunity landing moved by 1 ms', 'R0b'), (E, 'LAND_SHIFT', -float(SC.B0['L']), 'the landing at the decision', 'R0d'),
                                           (E, 'IGNORE_REFUSED', True, 'refused posts kept as activated', 'R0e'), (None, 'LAST_PULL', True, 'the last instead of the first pull step', 'R0f')):
            get_ = (lambda: globals()[flag]) if mod is None else (lambda: getattr(mod, flag))       # this module's own switch through globals() (the mutation
            set_ = (lambda v: globals().update({flag: v})) if mod is None else (lambda v: setattr(mod, flag, v))   # harness runs a copy under another name)
            old = get_(); set_(val_)
            try:
                refused(name, msg)
            finally:
                set_(old)
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print('pipeline paired eval selftests passed (synthetic only): the planted admissions and pull steps; every control passes; the B values of A, A-PA and PI0 '
          'recomputed by a per-fill loop; readings recomputed from the CSV; refusals by their own control with no output: a missing prediction (R0a), the '
          'per-opportunity landing moved (R0b), a POST label that is not the order\'s value under pi0 (R0c), the landing at the decision (R0d), refused posts kept '
          '(R0e), the last instead of the first pull step (R0f)', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'day':
        run(a[1])
    else:
        sys.exit('usage: pf_pipeline_paired_eval.py selftest | day <day>')
