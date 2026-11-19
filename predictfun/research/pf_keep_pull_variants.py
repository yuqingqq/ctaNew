"""BASELINE VERSUS CANDIDATE: THE FIVE DECLARED SCORE VARIANTS AGAINST A+0 THROUGH THE SCENARIO BANK (2026-11-19, session aea5bfb9).
USER: "Delete void outputs, can we run the one day test now"; asked what the test compares, the USER chose "Score variants vs A+0" (one day
first: Oct 23). Review a159116e's one next priority: "a predeclared baseline-versus-candidate research comparison through the same
consumed-day protocol and scenario bank. Preserve by-day values, fill populations, age/source splits and timing sensitivity."
Declared before any variant is valued in the scenario bank; consumed days only; nothing is fitted, chosen or promoted.
v2 (2026-11-19, review e96321df P2): the policy-free table and the CSV printed PULL-ALL's improvement over KEEP under a bare "PULL-ALL" beside
KEEP's absolute value; both now show PULL-ALL's improvement and its value (KEEP + improvement). The comparisons (score - PULL-ALL) were right.

POLICIES (pf_keep_pull.POLICIES, declared 2026-11-17 before the keep / pull run; multiplicity 6). Each decides at the order's live steps
 (every 50 ms from +100 ms) from the causal fair price f_j and pulls at its first trigger (at B the pull lands 37 ms later):
 A+0          BASELINE, the declared fair-edge rule: pull when the order's fair edge s * (f_j - price) falls below 0 c
 A+1 / A-1    the same edge rule with the threshold at +1 / -1 c (pulls earlier / later)
 B0.5 / B1 / B2   pull when the fair price has moved 0.5 / 1 / 2 c against the order since posting: s * (f_j - f_0) <= -theta
 The orders are the reference run's: every counted causal state (every 100th 50 ms decision step - every 5 s - of every window-side)
 posts one 30 sh JOIN order at the known best, live at +58 ms, held to END = +5 s (pf_keep_pull.py ORDERS).
METHOD. The scenario replay's day outputs (pf_keep_pull_scenarios v6: each committed state's keep branch re-simulated through the kernel in
 every scenario) do not depend on the policy: a pull only removes the keep branch's fills at or after its landing
 (pf_keep_pull_scenarios.outcome). Every policy is valued from the same committed day outputs with outcome(..., policy=p): no kernel
 re-simulation. All six are reported; none is selected.
VALUES per policy p and scenario s (the bank's 19; c summed over the days): E_p (the improvement over p's own exact STATES null), score_p
 (the improvement over KEEP), V_p = KEEP + score_p (the rule's own local value at END, not P&L), pulls; C1 (E by the order's age at the
 fill, without the < 100 ms fills, B1a-weighted), C3 (fills landing while p's pull is in flight), C12 (E at B by trade-anchor source).
 KEEP and PULL-ALL do not depend on the policy.
COMPARISON per candidate c against A+0, per scenario: dV = V_c - V_A+0 (= score_c - score_A+0), dE = E_c - E_A+0, dV-NY = dV without the
 fills on orders < 100 ms old.
CONTROLS (any failure -> no output):
 V0a A+0 reproduces the committed scenario report of the same days (pf_keep_pull_scenarios' scenarios_<days>.json): every value of every
     scenario
 V0b at B every policy's pulls and improvement over KEEP equal the committed keep / pull run's (keep_pull.json, M50 / 37 ms, by day)
 V0c at B every policy's E, improvement and pulls equal the clock audit's per-fill values (pf_keep_pull_clock_audit.per_fill) on the
     committed tables - a second code path
 V0d KEEP, PULL-ALL, the fills and the eligible orders are the same for every policy in every scenario; KEEP is the sum of its fills' values
 V0e the scenario and keep / pull day outputs match their receipts; the scenario days were built on those keep / pull outputs; the
     settlement-timing model is the current one
READINGS (computed; declared here before any variant was valued in the scenario bank)
 per policy (A+0 included)
  P1 E ACROSS TESTED SCENARIOS, P3 V ACROSS TESTED SCENARIOS, P1-NY / P3-NY the same without fills on orders < 100 ms old:
     STABLE-POSITIVE- / STABLE-NEGATIVE- / NOT-STABLE-IN-TESTED-SCENARIOS (as the scenario replay's Z1 / Z3)
  P4 V - KEEP and P5 V - V(PULL-ALL) ACROSS TESTED SCENARIOS: ABOVE-<ref>- / BELOW-<ref>-IN-TESTED-SCENARIOS if the difference is > 0 / < 0
     in every scenario, else NOT-STABLE-IN-TESTED-SCENARIOS
  C2 E AT M250 vs M50, C2 V AT M250 vs M50 (HOLDS- / SIGN-DIFFERS-AT-M250); C1 B1a-WEIGHTED E AT B (HOLDS-SIGN / SIGN-DIFFERS)
 per candidate
  D1 V - V(A+0), D2 E - E(A+0) ACROSS TESTED SCENARIOS, D1-NY V - V(A+0) WITHOUT FILLS ON ORDERS < 100 MS OLD: ABOVE-BASELINE- /
     BELOW-BASELINE- / NOT-STABLE-IN-TESTED-SCENARIOS; D1 AT M250 vs M50 (HOLDS- / SIGN-DIFFERS-AT-M250)
 summary (the policies whose reading says so, or none)
  SURVIVES-TIMING: P1 reads STABLE-POSITIVE-IN-TESTED-SCENARIOS; ABOVE THE BASELINE IN EVERY TESTED SCENARIO: D1 reads ABOVE-BASELINE-...
LIMITS: one consumed day first (Oct 23; G = 1: point values, no interval); the candidates' M50 / 37 ms values were seen in the keep / pull
 run, their values in the rest of the bank were not; multiplicity 6: a reading above the baseline is a candidate, never a validation (C8 /
 C16); local marks at END, not P&L; queue rule M only (C5); one order at a time, no re-post after a pull; the bank's assumption ranges, one
 factor at a time.
SELFTEST (synthetic tape, no data): the production path end to end on the keep / pull synthetic tape (queue place, the keep / pull run and
 its report, one scenario day and its report) with every control passing and the JSON / CSV / text outputs agreeing; the readings on a
 planted result tree; refusals, each by its own control with no output: a changed value in the committed scenario report (V0a), a changed
 per-day value of a candidate in the committed keep / pull report (V0b), a changed scenario fill under a matching receipt (V0a), a changed
 scenario fill and a changed keep / pull fill without their receipts (V0e), a changed keep / pull fill under a matching receipt (V0e), a
 perturbed E (V0c) and KEEP (V0d) in a result tree.
 Mutations: pf_keep_pull_variants_mutants.py.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_variants.py selftest | day <day> | all
"""
import contextlib
import copy
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_keep_pull as KP                                 # noqa: E402
import pf_keep_pull_clock_audit as A                      # noqa: E402
import pf_keep_pull_exec_windows as X                     # noqa: E402
import pf_keep_pull_scenarios as SC                       # noqa: E402
import pf_queue_place_test as QP                          # noqa: E402
import pf_queue_place_ledger as QL                        # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'keep_pull_variants_v2_20261119'     # v2 (review e96321df P2): PULL-ALL as both its improvement and its value; v1 stays as provenance
BASE = KP.DECLARED                                        # the baseline: the declared fair-edge rule A+0
CANDIDATES = tuple(p for p in KP.POLICIES if p != BASE)   # A+1, A-1, B0.5, B1, B2
KP_CELL = 'M50_L37'                                       # the committed keep / pull run's cell of the bank's baseline B
B_NAME, M250_NAME = SC.name(SC.B0), SC.name(dict(SC.B0, tl='M250'))
MARK = 'KEEP-PULL VARIANTS'
P_KEYS = ('P1 E ACROSS TESTED SCENARIOS', 'P3 V ACROSS TESTED SCENARIOS', 'P1-NY E WITHOUT FILLS ON ORDERS < 100 MS OLD ACROSS TESTED SCENARIOS',
          'P3-NY V WITHOUT FILLS ON ORDERS < 100 MS OLD ACROSS TESTED SCENARIOS', 'P4 V - KEEP ACROSS TESTED SCENARIOS',
          'P5 V - V(PULL-ALL) ACROSS TESTED SCENARIOS', 'C2 E AT M250 vs M50', 'C2 V AT M250 vs M50', 'C1 B1a-WEIGHTED E AT B')
D_KEYS = ('D1 V - V(A+0) ACROSS TESTED SCENARIOS', 'D2 E - E(A+0) ACROSS TESTED SCENARIOS',
          'D1-NY V - V(A+0) WITHOUT FILLS ON ORDERS < 100 MS OLD ACROSS TESTED SCENARIOS', 'D1 AT M250 vs M50')
S_KEYS = ('SURVIVES-TIMING (P1 STABLE-POSITIVE-IN-TESTED-SCENARIOS)', 'ABOVE THE BASELINE IN EVERY TESTED SCENARIO (D1)')


def above(lo, hi, ref):
    """A difference against a reference over the tested scenarios: above / below it in every one, else not stable."""
    return f'ABOVE-{ref}-IN-TESTED-SCENARIOS' if lo > 0 else f'BELOW-{ref}-IN-TESTED-SCENARIOS' if hi < 0 else 'NOT-STABLE-IN-TESTED-SCENARIOS'


def numbers(v):
    """A policy's numbers in one scenario that the readings use (V = KEEP + the improvement; _ny: without fills on orders < 100 ms old)."""
    ny = v['no_young']
    return dict(E=v['E'], V=v['KEEP'] + v['score'], score=v['score'], vs_pullall=v['score'] - v['PULLALL'], E_ny=ny['E'], V_ny=ny['fills_value'] + ny['score'])


def diffs(res, c):
    """Candidate c against the baseline per scenario: dV = V_c - V_A+0 (KEEP is the same for both), dE, dV without the < 100 ms fills."""
    out = {}
    for nm in res[BASE]:
        a, b = numbers(res[c][nm]), numbers(res[BASE][nm])
        out[nm] = dict(dV=a['V'] - b['V'], dE=a['E'] - b['E'], dV_ny=a['V_ny'] - b['V_ny'])
    return out


def readings(res):
    """The declared readings of a result tree {policy: {scenario: outcome}} whose scenarios include B and M250."""
    R = {}
    for p in KP.POLICIES:
        x = {nm: numbers(v) for nm, v in res[p].items()}; lo = {k: min(y[k] for y in x.values()) for k in x[B_NAME]}; hi = {k: max(y[k] for y in x.values()) for k in x[B_NAME]}
        R[p] = dict(zip(P_KEYS, (SC.stable(lo['E'], hi['E']), SC.stable(lo['V'], hi['V']), SC.stable(lo['E_ny'], hi['E_ny']), SC.stable(lo['V_ny'], hi['V_ny']),
                                 above(lo['score'], hi['score'], 'KEEP'), above(lo['vs_pullall'], hi['vs_pullall'], 'PULL-ALL'),
                                 SC.at_m250(x[B_NAME]['E'], x[M250_NAME]['E']), SC.at_m250(x[B_NAME]['V'], x[M250_NAME]['V']),
                                 SC.b1a_sign(res[p][B_NAME]['E'], res[p][B_NAME]['b1a_E']))))
    for c in CANDIDATES:
        d = diffs(res, c); lo = {k: min(y[k] for y in d.values()) for k in d[B_NAME]}; hi = {k: max(y[k] for y in d.values()) for k in d[B_NAME]}
        R[c].update(zip(D_KEYS, (above(lo['dV'], hi['dV'], 'BASELINE'), above(lo['dE'], hi['dE'], 'BASELINE'), above(lo['dV_ny'], hi['dV_ny'], 'BASELINE'),
                                 SC.at_m250(d[B_NAME]['dV'], d[M250_NAME]['dV']))))
    R['SUMMARY'] = dict(zip(S_KEYS, ([p for p in KP.POLICIES if R[p][P_KEYS[0]] == 'STABLE-POSITIVE-IN-TESTED-SCENARIOS'] or 'none',
                                     [c for c in CANDIDATES if R[c][D_KEYS[0]] == 'ABOVE-BASELINE-IN-TESTED-SCENARIOS'] or 'none')))
    return R


def differ(a, b, at=''):
    """The key paths where two result trees differ: numbers within 1e-9 relative (NaN equals NaN), everything else exactly."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or sorted(map(str, a)) != sorted(map(str, b)):
            return [at + ' (keys)']
        bb = {str(k): v for k, v in b.items()}
        return [x for k, v in a.items() for x in differ(v, bb[str(k)], f'{at}.{k}')]
    num = lambda z: isinstance(z, (int, float, np.integer, np.floating)) and not isinstance(z, (bool, np.bool_))
    if num(a) and num(b):
        fa, fb = float(a), float(b)
        return [] if (np.isnan(fa) and np.isnan(fb)) or abs(fa - fb) <= 1e-9 * max(1., abs(fa)) else [at]
    return [] if a == b else [at]


def load(days, sc_out, kp):
    """The committed scenario day outputs and the keep / pull tables of `days`, with the receipt checks (V0e) -> O, Fl, S, SF, Op, receipts."""
    O, Fl, S, SF, rec = [], [], [], [], {}
    for d in days:
        r = json.loads((sc_out / d / 'receipt.json').read_text()); rk = json.loads((kp / d / 'receipt.json').read_text())
        C.require(r.get('settlement_model') == X.SETTLEMENT_MODEL, f'V0e {d}: legacy settlement timing receipt')
        C.require(all(C.sha(sc_out / d / x) == h for x, h in r['outputs'].items()), f'V0e {d}: scenario outputs changed since their receipt')
        C.require(all(C.sha(kp / d / x) == h for x, h in rk['outputs'].items()), f'V0e {d}: keep-pull outputs changed since their receipt')
        C.require(r['keep_pull_outputs'] == rk['outputs'], f'V0e {d}: scenario day built on other keep-pull outputs')
        O.append(pd.read_parquet(kp / d / 'orders.parquet')); f = pd.read_parquet(kp / d / 'fills.parquet'); f['day'] = d; Fl.append(f)
        s_ = pd.read_parquet(sc_out / d / 'orders.parquet'); s_['day'] = d; S.append(s_); f_ = pd.read_parquet(sc_out / d / 'fills.parquet'); f_['day'] = d; SF.append(f_)
        rec[d] = dict(scenario_receipt=C.sha(sc_out / d / 'receipt.json'), scenario_asof=r['asof'], keep_pull_receipt=C.sha(kp / d / 'receipt.json'), stat=r['stat'])
    O, Fl, S, SF = (pd.concat(x, ignore_index=True) for x in (O, Fl, S, SF))
    return O, Fl, S, SF, O[O.status.isin(('eligible', 'refused_post_only'))], rec


def scen_inputs(sc, Op, Fl, S, SF):
    """One scenario's per-order rows and fills as the scenario report builds them (M0 / M250: the committed run's uniform shifts)."""
    nm = SC.name(sc)
    if sc['tl'] in ('M0', 'M250'):
        tag = KP.CTAG[float(sc['tl'][1:])]
        Os = pd.DataFrame(dict(day=Op.day.values, oid=Op.oid.values, refused=(Op.status == 'refused_post_only').values, v=Op['v' + tag].values, q=Op['q' + tag].values,
                               t_full=Op['t_full' + tag].values, land=Op.land_struct.values))
        Fs = Fl[Fl.clock == float(sc['tl'][1:])][['day', 'oid', 't', 'q', 'src']]
    else:
        key = SC.name(dict(sc, N=KP.NOTICE)); Os = S[S.scen == key][['day', 'oid', 'refused', 'v', 'q', 't_full', 'land']]
        Fs = SF[SF.scen == key][['day', 'oid', 't', 'q', 'held', 'src']]
    C.require(len(Os) == len(Op), f'{nm}: {len(Os)} order rows for {len(Op)} committed states with a path')
    return Os, Fs


def controls(res, days, O, Fl, ref, kp_json):
    """V0a-V0d on a result tree -> the failures (none: pass)."""
    bad = []; J = json.loads(Path(ref).read_text())
    if J.get('days') != list(days) or J.get('settlement_model') != X.SETTLEMENT_MODEL:                       # V0a: A+0 = the committed scenario report
        bad.append(f"V0a: the committed scenario report covers {J.get('days')} under {J.get('settlement_model')}")
    else:
        for nm, v in res[BASE].items():
            v_ = {k: x for k, x in v.items() if not k.startswith('_')}; r_ = J['scenarios'].get(nm, {})
            dd = differ(v_, {k: r_[k] for k in v_}) if all(k in r_ for k in v_) else ['missing values']
            if dd:
                bad.append(f'V0a {nm}: A+0 differs from the committed scenario report at {dd[:4]}')
    D = json.loads(Path(kp_json).read_text())['D'][KP_CELL]; ix = [KP.DAYS.index(d) for d in days]           # V0b: the committed keep / pull run
    for p in KP.POLICIES:
        n = sum(D[p]['policy_by_day']['n'][i] for i in ix); imp = sum(D[p]['policy_by_day']['imp'][i] for i in ix); b = res[p][B_NAME]
        if not (int(round(n)) == b['pulls'] and abs(imp - b['score']) <= 1e-6 * max(1., abs(imp))):
            bad.append(f"V0b {p}: pulls {b['pulls']:,} / improvement {b['score']:,.3f} at B, the committed keep / pull run {n:,.0f} / {imp:,.3f}")
    Ee = O[O.status == 'eligible'].reset_index(drop=True); cell = Ee.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values   # V0c: the clock audit
    T = KP.score_steps(Ee[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, Ee.side.values, Ee.p.values)
    endj, fl = A.endj_of(Ee.t_full.values, Ee.g_struct.values), A.fills_of(Ee, Fl, SC.SHIFT)
    for p in KP.POLICIES:
        pf = A.per_fill(T[p], cell, endj, fl); b = res[p][B_NAME]; e_, s_ = float(pf['score'].sum() - pf['null'].sum()), float(pf['score'].sum())
        if not (abs(b['E'] - e_) < 1e-6 and abs(b['score'] - s_) < 1e-6 and pf['pulls'] == b['pulls']):
            bad.append(f"V0c {p}: E {b['E']:,.3f} / improvement {b['score']:,.3f} / pulls {b['pulls']:,} at B, the clock audit {e_:,.3f} / {s_:,.3f} / {pf['pulls']:,}")
    for nm in res[BASE]:                                                                                      # V0d: what does not depend on the policy
        if len({(res[p][nm]['KEEP'], res[p][nm]['PULLALL'], res[p][nm]['fills'], res[p][nm]['fills_value'], res[p][nm]['eligible']) for p in KP.POLICIES}) != 1:
            bad.append(f'V0d {nm}: KEEP / PULL-ALL / fills / eligible orders differ across the policies')
    for p in KP.POLICIES:
        k = SC.keep_not_fills(res[p])
        if k:
            bad.append(f'V0d {p}: KEEP is not the sum of its fills\' values in {k}')
    return bad


def tables(res, R, days, Op, rec):
    """The printed report and the two CSV tables (one row per policy and scenario; one per candidate and scenario)."""
    nms = list(res[BASE]); fam = {SC.name(sc): SC.family(sc)[0] for sc in SC.SCEN}; dd = {c: diffs(res, c) for c in CANDIDATES}
    rows = []
    for p in KP.POLICIES:
        for nm, v in res[p].items():
            n_, a_, cp = numbers(v), v['by_age'], v['cancel_pending']
            rows.append(dict(policy=p, role='baseline' if p == BASE else 'candidate', scenario=nm, family=fam[nm], E=v['E'], score=v['score'], null=v['null'], V=n_['V'],
                             KEEP=v['KEEP'], PULLALL_improvement=v['PULLALL'], PULLALL_value=v['KEEP'] + v['PULLALL'], pulls=v['pulls'], eligible=v['eligible'], fills=v['fills'], shares=v['shares'],
                             E_lt100=a_['lt100']['E'], E_100to500=a_['100to500']['E'], E_ge500=a_['ge500']['E'], E_noyoung=n_['E_ny'], V_noyoung=n_['V_ny'],
                             E_b1a_M0=v['b1a_E']['M0'], E_b1a_M50=v['b1a_E']['M50'], pending_fills=cp['fills'], pending_value=cp['value'],
                             pending_share_of_gross=cp['share_of_gross']))
    P = pd.DataFrame(rows); Q = pd.DataFrame([dict(candidate=c, scenario=nm, family=fam[nm], **x) for c in CANDIDATES for nm, x in dd[c].items()])
    col = lambda f: ' '.join(f'{f(p):>11}' for p in KP.POLICIES); hdr = f"  {'scenario':<24} " + col(lambda p: p + (' (base)' if p == BASE else ''))
    tab = lambda title, g: ['', title, hdr] + [f'  {nm:<24} ' + col(lambda p: f'{g(res[p][nm]):+,.0f}') for nm in nms]
    chdr = f"  {'scenario':<24} " + ' '.join(f'{c:>11}' for c in CANDIDATES)
    ctab = lambda title, k: ['', title, chdr] + [f'  {nm:<24} ' + ' '.join(f'{dd[c][nm][k]:>+11,.0f}' for c in CANDIDATES) for nm in nms]
    L = [f'BASELINE VERSUS CANDIDATE: THE SIX DECLARED SCORE POLICIES THROUGH THE SCENARIO BANK (pf_keep_pull_variants.py; days {", ".join(days)}; c summed; baseline '
         f'{BASE}, candidates {", ".join(CANDIDATES)}; one 30 sh JOIN order per counted state, pulls decided every 50 ms from +100 ms; local marks at END, not P&L)',
         f"  {len(Op):,} states with a path ({int((Op.status == 'eligible').sum()):,} eligible at B); scenario day outputs as of "
         + ', '.join(f"{d} {rec[d]['scenario_asof']}" for d in days),
         f"  CONTROLS: {R['CONTROLS']} (V0a A+0 = the committed scenario report in every value of every scenario; V0b pulls and improvement at B = the committed "
         'keep / pull run, M50 / 37 ms, for every policy; V0c E, improvement and pulls at B = the clock audit\'s per-fill values for every policy; V0d KEEP / '
         'PULL-ALL / fills / eligible the same for every policy, KEEP = its fills; V0e receipts)', '',
         f"  policy-free: {'scenario':<24} {'KEEP value':>12} {'PULL-ALL improvement':>21} {'PULL-ALL value':>15} {'fills':>8} {'eligible':>9}   (PULL-ALL value = KEEP + its improvement)"]
    L += [f"               {nm:<24} {res[BASE][nm]['KEEP']:>+12,.0f} {res[BASE][nm]['PULLALL']:>+21,.0f} {res[BASE][nm]['KEEP'] + res[BASE][nm]['PULLALL']:>+15,.0f} "
          f"{res[BASE][nm]['fills']:>8,} {res[BASE][nm]['eligible']:>9,}" for nm in nms]
    L += tab('  E: each policy\'s improvement over its own matched STATES null (c)', lambda v: v['E'])
    L += tab('  V = KEEP + the policy\'s improvement over KEEP (the rule\'s own local value, c)', lambda v: v['KEEP'] + v['score'])
    L += tab('  pulls executed', lambda v: v['pulls'])
    L += tab('  E without fills on orders < 100 ms old (c)', lambda v: v['no_young']['E'])
    L += tab('  V without fills on orders < 100 ms old (c)', lambda v: v['no_young']['fills_value'] + v['no_young']['score'])
    L += ctab('  dV = V(candidate) - V(A+0) (c)', 'dV') + ctab('  dE = E(candidate) - E(A+0) (c)', 'dE') + ctab('  dV without fills on orders < 100 ms old (c)', 'dV_ny')
    L += ['', '  C1 / C3 at B per policy: E by the order\'s age at the fill (< 100 / 100-500 / >= 500 ms after activation), B1a-weighted (M 0 / M 50 weights), and '
          'the fills landing while the pull is in flight (count / value c / share of the fills\' gross) at B, L100 and the slow mixture:']
    for p in KP.POLICIES:
        b = res[p][B_NAME]; a_ = b['by_age']; cps = [res[p][nm]['cancel_pending'] for nm in (B_NAME, SC.name(dict(SC.B0, L=100.)), SC.name(dict(SC.B0, X=SC.CANCEL_MIX)))]
        L.append(f"  {p:<5} E {a_['lt100']['E']:+,.0f} / {a_['100to500']['E']:+,.0f} / {a_['ge500']['E']:+,.0f}; B1a {b['b1a_E']['M0']:+,.0f} / {b['b1a_E']['M50']:+,.0f}; pending "
                 + ' | '.join(f"{x['fills']:,} / {x['value']:+,.0f} / {100 * x['share_of_gross']:.1f}%" for x in cps))
    L += ['', f"  C12 at B per policy: E by trade-anchor source ({' / '.join(SC.SRC_LABELS)}; E2 anchors are uncertain by ~+-250 ms), c (fills):"]
    for p in KP.POLICIES:
        bs = res[p][B_NAME]['by_age_src']
        L.append(f'  {p:<5} ' + ' | '.join(f"{sl} {sum(bs[lab][sl]['E'] for lab in SC.AGE_LABELS):+,.0f} ({sum(bs[lab][sl]['fills'] for lab in SC.AGE_LABELS):,})" for sl in SC.SRC_LABELS))
    L += ['', 'READINGS (declared before any variant was valued in the scenario bank)']
    for p in KP.POLICIES:
        L += [f"  {p}{' (baseline)' if p == BASE else ''}:"] + [f'    {k}: {v}' for k, v in R[p].items()]
    L += ['  SUMMARY:'] + [f'    {k}: {v}' for k, v in R['SUMMARY'].items()]
    L += ['', f'LIMITS: consumed day(s) {", ".join(days)} (G = {len(days)}: point values, no interval); multiplicity {len(KP.POLICIES)} - a reading above the baseline is a '
          'candidate, never a validation (C8 / C16); the candidates\' M50 / 37 ms values were seen in the keep / pull run; local marks at END, not P&L; queue rule M '
          'only (C5); one order at a time, no re-post after a pull; the bank\'s assumption ranges, one factor at a time.']
    return L, P, Q


def run(days, out=None, sc_out=None, kp=None):
    """Every declared policy in every scenario of the bank on `days` -> variants_<days>.json / .txt / _policies.csv / _compare.csv."""
    start = time.time(); days = list(days)
    C.require(len(days) > 0 and len(set(days)) == len(days) and all(d in KP.DAYS for d in days), 'undeclared day')
    out = Path(out or OUT); sc_out = Path(sc_out or SC.OUT); kp = Path(kp or KP.OUT); allday = days == list(KP.DAYS)
    stem = 'variants' if allday else 'variants_' + '_'.join(days); ref = sc_out / (('scenarios' if allday else 'scenarios_' + '_'.join(days)) + '.json')
    C.require(not (out / f'{stem}.json').exists(), f'{stem}: output exists')
    O, Fl, S, SF, Op, rec = load(days, sc_out, kp); res = {p: {} for p in KP.POLICIES}
    for sc in SC.SCEN:
        Os, Fs = scen_inputs(sc, Op, Fl, S, SF)
        for p in KP.POLICIES:
            res[p][SC.name(sc)] = SC.outcome(O, Os, Fs, sc, policy=p)
        C.guard('keep-pull variants')
    bad = controls(res, days, O, Fl, ref, kp / 'keep_pull.json')
    if bad:
        sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(bad))
    R = readings(res); R['CONTROLS'] = 'YES' if not bad else 'NO'
    L, P, Q = tables(res, R, days, Op, rec)
    vals = {p: {nm: {k: x for k, x in v.items() if not k.startswith('_')} for nm, v in res[p].items()} for p in KP.POLICIES}
    out.mkdir(parents=True, exist_ok=True); P.to_csv(out / f'{stem}_policies.csv', index=False); Q.to_csv(out / f'{stem}_compare.csv', index=False)
    (out / f'{stem}.txt').write_text('\n'.join(L) + '\n')
    C.dump(out / f'{stem}.json', dict(asof=C.stamp(), days=days, readings=R, values=vals, diffs={c: diffs(res, c) for c in CANDIDATES},
                                      policies=dict(baseline=BASE, candidates=list(CANDIDATES), multiplicity=len(KP.POLICIES), keep_pull_cell=KP_CELL),
                                      families={SC.name(sc): SC.family(sc) for sc in SC.SCEN},
                                      inputs=dict(scenario_report=str(ref), scenario_report_sha256=C.sha(ref), keep_pull_report=str(kp / 'keep_pull.json'),
                                                  keep_pull_report_sha256=C.sha(kp / 'keep_pull.json'), days=rec),
                                      outputs={x: C.sha(out / x) for x in (f'{stem}.txt', f'{stem}_policies.csv', f'{stem}_compare.csv')},
                                      provenance=KP.provenance(), source_sha256=C.sha(Path(__file__).resolve()),
                                      dependencies={Path(m.__file__).name: C.sha(Path(m.__file__)) for m in (SC, KP, A, X)}, seconds=time.time() - start))
    print('\n'.join(L), flush=True)
    return res, R


PLANT = {  # per policy: (E, improvement, E without young fills, improvement without young fills) at B / M250 / M0 / S1lo, and B's B1a values
    'A+0': (((10., 40., 8., 35.), (-5., 20., -6., 15.), (12., 45., 10., 40.), (3., 30., 2., 25.)), (4., -1.)),
    'A+1': (((12., 50., 9., 30.), (2., 30., 1., 25.), (14., 55., 11., 35.), (5., 45., 4., 20.)), (3., 6.)),
    'A-1': (((-3., 10., -2., 100.), (-4., 5., -3., 2.), (-2., 12., -1., 3.), (-1., 8., -1., 4.)), (-1., -2.)),
    'B0.5': (((10., 40., 8., 35.), (-5., 20., -6., 15.), (12., 45., 10., 40.), (3., 30., 2., 25.)), (4., -1.)),
    'B1': (((8., 160., 15., 150.), (6., 140., 5., 130.), (10., 165., 12., 155.), (9., 155., 7., 150.)), (5., 5.)),
    'B2': (((1., 35., 1., 40.), (-1., 25., 1., 20.), (2., 38., 2., 42.), (2., 30., 2., 30.)), (1., 1.))}
WANT = {  # the planted tree's readings: P1 P3 P1-NY P3-NY P4 P5 C2E C2V C1 | D1 D2 D1-NY D1@M250 (SP / SN / NS stable, A / B above / below, H / D M250, HS / SD B1a)
    'A+0': 'NS SN NS SN A B D H SD', 'A+1': 'SP SN SP SN A NS H H HS | A A NS H', 'A-1': 'SN SN SN NS A B H H HS | B NS NS H',
    'B0.5': 'NS SN NS SN A B D H SD | NS NS NS D', 'B1': 'SP SP SP SP A A H H HS | A NS A H', 'B2': 'NS SN SP SN A B D H HS | NS NS A D'}


def planted():
    """A result tree with chosen values: KEEP -100 and PULL-ALL +50 in each of four scenarios (B, M250, M0, S1lo); the young fills' value -90 removed."""
    sc4 = (B_NAME, M250_NAME, SC.name(dict(SC.B0, tl='M0')), SC.name(dict(SC.B0, tl='S1lo')))
    mk = lambda e, s, ey, sy, w: dict(E=e, score=s, KEEP=-100., PULLALL=50., no_young=dict(E=ey, score=sy, fills_value=-90.), b1a_E=dict(M0=w[0], M50=w[1]))
    return {p: {nm: mk(*x, w) for nm, x in zip(sc4, rows)} for p, (rows, w) in PLANT.items()}


def want_readings():
    """WANT spelled out in the readings' words."""
    st = dict(SP='STABLE-POSITIVE-IN-TESTED-SCENARIOS', SN='STABLE-NEGATIVE-IN-TESTED-SCENARIOS', NS='NOT-STABLE-IN-TESTED-SCENARIOS')
    m = dict(H='HOLDS-AT-M250', D='SIGN-DIFFERS-AT-M250'); b = dict(HS='HOLDS-SIGN', SD='SIGN-DIFFERS'); out = {}
    for p, w in WANT.items():
        a, *d = w.split(' | '); a = a.split()
        ab = lambda x, ref: st['NS'] if x == 'NS' else f"{'ABOVE' if x == 'A' else 'BELOW'}-{ref}-IN-TESTED-SCENARIOS"
        out[p] = dict(zip(P_KEYS, [st[x] for x in a[:4]] + [ab(a[4], 'KEEP'), ab(a[5], 'PULL-ALL'), m[a[6]], m[a[7]], b[a[8]]]))
        if d:
            d = d[0].split(); out[p].update(zip(D_KEYS, [ab(x, 'BASELINE') for x in d[:3]] + [m[d[3]]]))
    out['SUMMARY'] = dict(zip(S_KEYS, (['A+1', 'B1'], ['A+1', 'B1'])))
    return out


def selftest():
    """Synthetic tape only: the planted readings; the production path end to end; each refusal by its own control with no output."""
    bad = []; src = KP.Syn(); days = list(KP.DAYS); d0 = days[0]
    env = lambda d_: dict(h=600., q01=-900., q99=1050., inputs={})
    syn_set = lambda d_: (pd.DataFrame(dict(t_set_ms=np.zeros(0), set_status=np.zeros(0, int))), {})
    if (above(1., 2., 'X'), above(-2., -1., 'X'), above(-1., 1., 'X'), above(0., 1., 'X'), above(-1., 0., 'X')) != (
            'ABOVE-X-IN-TESTED-SCENARIOS', 'BELOW-X-IN-TESTED-SCENARIOS', 'NOT-STABLE-IN-TESTED-SCENARIOS', 'NOT-STABLE-IN-TESTED-SCENARIOS', 'NOT-STABLE-IN-TESTED-SCENARIOS'):
        bad.append('the above / below predicate')
    if not (differ({'a': 1., 'b': {'c': 2}}, {'a': 1., 'b': {'c': 2}}) == [] and differ({'a': 1.}, {'a': 1.1}) == ['.a'] and differ({'a': 1}, {'b': 1}) == [' (keys)']
            and differ({'a': float('nan')}, {'a': float('nan')}) == [] and differ({'a': {'b': 1.}}, {'a': 1.}) == ['.a (keys)'] and differ({'a': 'x'}, {'a': 'y'}) == ['.a']):
        bad.append('the result-tree comparison')
    try:
        got = readings(planted()); want = want_readings()
        for p in want:
            for k, v in want[p].items():
                if got.get(p, {}).get(k) != v:
                    bad.append(f'planted reading {p} {k}: {got.get(p, {}).get(k)} != {v}')
        if set(got) != set(want) | {'SUMMARY'} or any(set(got[p]) != set(want[p]) for p in want):
            bad.append(f'planted readings: keys {sorted(got)}')
        dv = diffs(planted(), 'A+1')
        if not (dv[B_NAME] == dict(dV=10., dE=2., dV_ny=-5.) and dv[M250_NAME] == dict(dV=10., dE=7., dV_ny=10.)):
            bad.append(f'planted differences: {dv[B_NAME]} {dv[M250_NAME]}')
    except (KeyError, TypeError, ValueError) as e:
        bad.append(f'the planted readings failed: {e!r}')
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
        res = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res, R = run([d0], out=tmp / 'var', sc_out=tmp / 'sc', kp=tmp / 'kp')
        except (SystemExit, ValueError) as e:
            bad.append(f'the synthetic production run refused: {e}')
        stem = f'variants_{d0}'
        if res is not None:
            J = json.loads((tmp / 'var' / f'{stem}.json').read_text()); P = pd.read_csv(tmp / 'var' / f'{stem}_policies.csv'); Q = pd.read_csv(tmp / 'var' / f'{stem}_compare.csv')
            txt = (tmp / 'var' / f'{stem}.txt').read_text(); nms = [SC.name(sc) for sc in SC.SCEN]
            back = readings(J['values'])                                                  # the predicates re-evaluated from the saved numbers
            if not (all(back[p] == J['readings'][p] for p in back) and J['readings']['CONTROLS'] == 'YES'):
                bad.append('the saved readings differ from the readings of the saved values')
            if not (len(P) == len(KP.POLICIES) * len(nms) and len(Q) == len(CANDIDATES) * len(nms)
                    and all(abs(P[(P.policy == p) & (P.scenario == nm)].E.iloc[0] - J['values'][p][nm]['E']) < 1e-6 for p in KP.POLICIES for nm in nms)
                    and all(abs(Q[(Q.candidate == c) & (Q.scenario == nm)].dV.iloc[0] - (J['values'][c][nm]['score'] - J['values'][BASE][nm]['score'])) < 1e-6
                            for c in CANDIDATES for nm in nms)):
                bad.append('the CSV tables differ from the saved values')
            if not all(f'{k}: {v}' in txt for p in KP.POLICIES for k, v in J['readings'][p].items()):
                bad.append('the text report lacks a reading')
            if not (np.allclose(P.PULLALL_value, P.KEEP + P.PULLALL_improvement, rtol=0, atol=1e-6) and 'PULL-ALL value' in txt
                    and np.allclose(P[P.policy == BASE].PULLALL_value.values, [J['values'][BASE][nm]['KEEP'] + J['values'][BASE][nm]['PULLALL'] for nm in nms], rtol=0, atol=1e-6)):
                bad.append('PULL-ALL: its value is not KEEP + its improvement in the CSV / text (review e96321df P2)')
            if len({res[p][B_NAME]['pulls'] for p in KP.POLICIES}) < 2:
                bad.append('the synthetic policies do not differ at B (the comparison would be vacuous)')
            O, Fl, _, _, _, _ = load([d0], tmp / 'sc', tmp / 'kp'); ref = tmp / 'sc' / f'scenarios_{d0}.json'
            if controls(res, [d0], O, Fl, ref, tmp / 'kp' / 'keep_pull.json'):
                bad.append('the controls fail on the unchanged result tree')
            for nm_, key, need, change in (('a perturbed E of a candidate', 'V0c', ('V0c',), lambda r: r[CANDIDATES[0]][B_NAME].update(E=r[CANDIDATES[0]][B_NAME]['E'] + 1.)),
                                           ('a perturbed KEEP of a candidate', 'V0d', ('differ across the policies', 'sum of its fills'),
                                            lambda r: r[CANDIDATES[-1]][M250_NAME].update(KEEP=r[CANDIDATES[-1]][M250_NAME]['KEEP'] + 1.))):
                r2 = copy.deepcopy(res); change(r2); got_ = controls(r2, [d0], O, Fl, ref, tmp / 'kp' / 'keep_pull.json')
                if not (got_ and all(x.startswith(key) for x in got_) and all(any(n in x for x in got_) for n in need)):
                    bad.append(f'{nm_}: controls {got_}, want only {key} with {need}')
        # refusals on the files, each by its own control, with no output
        def plant_sc(name_, receipt=True):
            d_ = tmp / name_; shutil.copytree(tmp / 'sc', d_); f_ = pd.read_parquet(d_ / d0 / 'fills.parquet'); j_ = int(np.flatnonzero(f_.scen.values != B_NAME)[0])
            f_.loc[f_.index[j_], 'q'] = f_.q.iloc[j_] - 1.; f_.to_parquet(d_ / d0 / 'fills.parquet', index=False)
            if receipt:
                r_ = json.loads((d_ / d0 / 'receipt.json').read_text()); r_['outputs']['fills.parquet'] = C.sha(d_ / d0 / 'fills.parquet'); (d_ / d0 / 'receipt.json').write_text(json.dumps(r_))
            return d_
        sca = tmp / 'sca'; shutil.copytree(tmp / 'sc', sca); Ja = json.loads((sca / f'scenarios_{d0}.json').read_text())
        Ja['scenarios'][B_NAME]['E'] += 1.; (sca / f'scenarios_{d0}.json').write_text(json.dumps(Ja))
        kpb = tmp / 'kpb'; shutil.copytree(tmp / 'kp', kpb); Jb = json.loads((kpb / 'keep_pull.json').read_text())
        Jb['D'][KP_CELL][CANDIDATES[-1]]['policy_by_day']['imp'][0] += 1.; (kpb / 'keep_pull.json').write_text(json.dumps(Jb))
        kpe = tmp / 'kpe'; shutil.copytree(tmp / 'kp', kpe); fk = pd.read_parquet(kpe / d0 / 'fills.parquet'); fk.loc[fk.index[0], 'q'] = fk.q.iloc[0] - 1.
        fk.to_parquet(kpe / d0 / 'fills.parquet', index=False)
        kpf = tmp / 'kpf'; shutil.copytree(kpe, kpf); rf = json.loads((kpf / d0 / 'receipt.json').read_text())       # the same change under a matching receipt
        rf['outputs']['fills.parquet'] = C.sha(kpf / d0 / 'fills.parquet'); (kpf / d0 / 'receipt.json').write_text(json.dumps(rf))
        for nm_, kw, msg in (('a changed value in the committed scenario report', dict(sc_out=sca, kp=tmp / 'kp'), 'V0a'),
                             ("a changed per-day value of a candidate in the committed keep / pull report", dict(sc_out=tmp / 'sc', kp=kpb), f'V0b {CANDIDATES[-1]}'),
                             ('a changed scenario fill under a matching receipt', dict(sc_out=plant_sc('scc'), kp=tmp / 'kp'), 'V0a'),
                             ('a changed scenario fill without its receipt', dict(sc_out=plant_sc('scd', receipt=False), kp=tmp / 'kp'), 'scenario outputs changed'),
                             ('a changed keep / pull fill without its receipt', dict(sc_out=tmp / 'sc', kp=kpe), 'keep-pull outputs changed'),
                             ('a changed keep / pull fill under a matching receipt', dict(sc_out=tmp / 'sc', kp=kpf), 'built on other keep-pull outputs')):
            o_ = tmp / ('z' + str(abs(hash(nm_)) % 100000))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    run([d0], out=o_, **kw)
                bad.append(f'the variants run accepted {nm_}')
            except (SystemExit, ValueError) as e:
                if msg not in str(e):
                    bad.append(f'{nm_} refused by another control: {str(e)[:300]}')
            if (o_ / f'{stem}.json').exists() or (o_ / f'{stem}.txt').exists():
                bad.append(f'a refused variants run wrote output ({nm_})')
        try:
            run(['20990101'], out=tmp / 'zz', sc_out=tmp / 'sc', kp=tmp / 'kp'); bad.append('the variants run accepted an undeclared day')
        except ValueError as e:
            if 'undeclared day' not in str(e):
                bad.append(f'the undeclared day refused by another check: {e}')
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print('keep-pull variants selftests passed (synthetic only): the above / below predicate and the result-tree comparison; every reading on a planted result '
          'tree; the production path end to end on the keep / pull synthetic tape (every control passing, the saved readings re-evaluated from the saved values, the '
          'CSV and text outputs agreeing, the policies differing at B); a perturbed E refused by V0c and a perturbed KEEP by V0d; a changed committed scenario value, '
          'a changed committed keep / pull value of a candidate, a changed scenario fill under a matching receipt, changed scenario and keep / pull fills without '
          'their receipts, a changed keep / pull fill under a matching receipt and an undeclared day each refused by its own control with no output', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'day':
        run([a[1]])
    elif a == ['all']:
        run(KP.DAYS)
    else:
        sys.exit('usage: pf_keep_pull_variants.py selftest | day <day> | all')
