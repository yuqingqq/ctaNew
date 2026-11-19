"""THE LEARNED KEEP / PULL RULE THROUGH THE SCENARIO BANK - THE ONE-DAY TEST OF THE WHOLE PIPELINE (2026-11-19, session aea5bfb9). USER: "We
run the whole pipeline for one day test first". The pipeline: pf_pipeline_export (causal inputs and labels on the simulator's JOIN orders) ->
pf_pipeline_bench (the joint two-part prediction model J, trained Oct 23-25, and the contract's value mapping) -> this module: the rule
declared in pf_pipeline_bench - "pull at the first live step whose predicted RETAIN value (J) is below 0", no threshold fitted - valued
through the simulator's 19-scenario bank (pf_keep_pull_scenarios.outcome with external triggers) on the evaluation day Oct 27, against the
declared fair-edge rule A+0, KEEP and PULL-ALL. Declared before the benchmark's predictions are read; one consumed day; nothing fitted here.

RULE (MODEL-J): per eligible order, T = the first live RETAIN step j (J0..98) whose predicted value V_J < 0, else never (NSTEP + 1). Every
 exported RETAIN row with status eligible (live at B) has a prediction; a step without one (off_grid / feature_unavailable) keeps - the
 contract's fallback (no forecast -> keep). The same decisions are evaluated in every scenario: information never moves with an execution
 time. LIMIT: the model's inputs that depend on the order's own fills (known remaining) are B's in every scenario.
v2 (review 89caa74c P2, declared before v2 ran): the rule acts only on contract-coherent forecasts (the benchmark's J_ok, the contract's own
 two-atom check); a refused forecast keeps (the contract's fallback) and is counted; P0f samples coherent AND incoherent rows and requires the
 contract's verdict = J_ok and its action (or fallback) = the rule's; P0g recomputes the executed pull steps by a second code path. Days: Oct 26
 (the benchmark's economic gate of record) and Oct 27 (post hoc). This is a prediction -> keep / pull test with decisions FIXED across the
 scenarios (an execution sensitivity), not a replay of a state-dependent controller and not POST / WAIT.
CONTRACT PARITY: on sampled rows the rule's action equals pf_pipeline_contract.decide_one on the equivalent two-atom forecast (P at
 (E[q | fill], E[qm] / E[q] per share), 1 - P at no fill; client state live, no economic terms) -> pull iff V_J < 0. Rows whose atom the
 contract refuses (an endpoint mid outside [0, 100]) or whose |V_J| < 1e-6 (a sign set by rounding) are counted, not compared.
VALUES per scenario (as pf_keep_pull_variants): E (the improvement over the rule's own exact STATES null), V = KEEP + the improvement, pulls,
 C1 (by age, without < 100 ms fills, B1a-weighted), C3 (cancel-pending), C12 (by trade-anchor source); differences against A+0.
CONTROLS (any failure -> no output):
 P0a A+0's E in every scenario equals the committed five-day scenario report's value for the day (keep_pull_scenarios_v6 scenarios.json by_day)
 P0b at B A+0's pulls and improvement equal the committed keep / pull run's for the day (keep_pull.json, M50 / 37 ms)
 P0c at B MODEL-J's E, improvement and pulls equal the clock audit's per-fill values for its trigger steps (a second code path)
 P0d KEEP, PULL-ALL, fills and eligible orders agree between the rules in every scenario; KEEP is the sum of its fills
 P0e provenance: the prediction file matches the benchmark's recorded hash and covers exactly the export's eligible RETAIN rows of the day
 P0f contract parity (above) on the sampled rows, with at least one comparable row
READINGS (computed): MODEL-J P1 E / P3 V / P1-NY / P3-NY ACROSS TESTED SCENARIOS (STABLE-POSITIVE- / STABLE-NEGATIVE- / NOT-STABLE-), P4 V - KEEP,
 P5 V - V(PULL-ALL) (ABOVE- / BELOW-<ref>- / NOT-STABLE-IN-TESTED-SCENARIOS), C2 E / V AT M250 vs M50, C1 B1a-WEIGHTED E AT B; against A+0: D1 V,
 D2 E, D1-NY (ABOVE- / BELOW-BASELINE- / NOT-STABLE-), D1 AT M250; carried: the benchmark's RETAIN mapping gate (a FAIL means the values are
 not attributable to a calibrated mapping).
LIMITS: one consumed day (G = 1, point values); the model was chosen by nothing here but is one of the benchmark's 8 candidates; local marks
 at END, not P&L; queue rule M only (C5); replay fills not validated on live orders (C10); one order at a time, no re-post after a pull.
SELFTEST (synthetic tape, no data): planted predictions give the planted trigger steps; the bank values them; every control passes; refusals
 by their own control with no output: a prediction for a row the export does not have and a missing prediction (P0e), a changed committed
 report value (P0a) and keep / pull value (P0b), the model rule valued as A+0 (P0c), a KEEP drift (P0d), the rule's sign reversed (P0f). Mutations: pf_pipeline_policy_mutants.py.
Usage (run_light, PF_SLOT=claude): pf_pipeline_policy.py selftest | day <day>
"""
import contextlib
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
import pf_keep_pull_scenarios as SC                       # noqa: E402
import pf_keep_pull_variants as V                         # noqa: E402
import pf_pipeline_contract as PC                         # noqa: E402  the shared contract (imported, never edited)
import pf_queue_place_test as QP                          # noqa: E402
import pf_queue_place_ledger as QL                        # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'pipeline_policy_v2_20261119'       # v2 (review 89caa74c): one accepted-forecast path for the rule and the parity
BENCH = Path(I.K.IO) / 'pipeline_bench_v2_20261119'
EXPORT = Path(I.K.IO) / 'pipeline_export_v2_20261119'
MARK = 'PIPELINE POLICY'
RULE = 'MODEL-J'
B_NAME, M250_NAME = V.B_NAME, V.M250_NAME
FLIP = False                                              # planted-error switches (selftest only): pull iff the predicted value is above 0;
MISS_TRIGGERS = False                                     # the model rule valued as A+0;
RAW_TRIGGERS = False                                      # the rule acting on forecasts the contract refuses (v1's path);
KEEP_SHIFT = 0.                                           # a drift in the model rule's KEEP at B


def triggers_of(pred):
    """MODEL-J's first pull step per order: the first contract-coherent predicted RETAIN value below 0 (pull iff J_ok and V_J < 0), else never;
    a forecast the contract refuses keeps (its fallback, review 89caa74c P2)."""
    p = pred[pred.kind == 'retain'][['day', 'oid', 'j', 'J_V', 'J_ok']].sort_values(['day', 'oid', 'j'], kind='stable')
    acts = (p.J_V > 0) if FLIP else (p.J_V < 0)
    hit = p[acts if RAW_TRIGGERS else (acts & p.J_ok.astype(bool))].groupby(['day', 'oid']).j.min()   # v2: only contract-coherent forecasts act
    return hit


def trigger_fn(hit):
    """outcome()'s callable: each eligible order's step (never when no predicted value is below 0)."""
    def f(E):
        ix = pd.MultiIndex.from_arrays([E.day.astype(str).values, E.oid.values.astype(int)])
        return hit.reindex(ix).fillna(KP.NSTEP + 1).astype(int).values
    return f


def contract_parity(pred, trained, n=2000, seed=11):
    """P0f: on RETAIN rows sampled from both the coherent and the incoherent forecasts, the contract's verdict equals the coherence flag and its
    action (decide_one, or the fallback keep when it refuses) equals the rule's -> (compared, refused atoms, agreements)."""
    rng = np.random.default_rng(seed); p = pred[pred.kind == 'retain']; parts = []
    for flag in (True, False):
        g = p[p.J_ok.astype(bool) == flag]; parts.append(g.iloc[rng.choice(len(g), min(n // 2, len(g)), replace=False)] if len(g) else g)
    p = pd.concat(parts); comp = ref = agree = 0
    for r in p.itertuples():
        if abs(r.J_V) < 1e-6:
            continue
        side = 'bid' if r.side == 0 else 'ask'; sg = 1. if r.side == 0 else -1.; rem = float(r.fp_remaining); t = float(r.t); d = float(r.decision_ms)
        tg = PC.Target(f'{r.day}:{int(r.oid)}', side, 'retain', d, t + float(KP.landing(int(r.j))), t + KP.H, float(r.pf_mid), float(r.p), rem,
                       B_NAME, 'no_repost_to_common_END;zero_continuation_after_END', 'JOIN30/END=birth+5000/queueM/structuralKeeper/scenarios-v6')
        try:
            qf = float(r.J_Eq / r.J_P) if r.J_P > 0 else 0.; mv = float(r.J_Eqm / r.J_Eq) if r.J_Eq > 0 else 0.
            atoms = (PC.Atom(float(r.J_P), qf, mv), PC.Atom(1. - float(r.J_P), 0., 0.)) if 0. < r.J_P < 1. else (PC.Atom(1., qf if r.J_P >= 1 else 0., mv),)
            fc = PC.Forecast(tg, atoms, d, trained, 'learned', 'pf_pipeline_bench:J')
            st = PC.ClientState(tg.order_id, side, d, 'live', rem, 0., rem if side == 'bid' else 0., rem if side == 'ask' else 0., 1e9)
            dec = PC.decide_one(tg, fc, st, PC.Economics(tg, 0., 0., 0., 0.)); accepted, action = True, dec.action
        except ValueError:
            ref += 1; accepted, action = False, 'keep'                                          # the contract's fallback: no usable forecast -> keep
        rule = 'pull' if (bool(r.J_ok) and bool((r.J_V > 0) if FLIP else (r.J_V < 0))) else 'keep'
        comp += 1; agree += int(accepted == bool(r.J_ok) and action == rule)
    return comp, ref, agree


def rule_readings(res, name):
    """The declared readings of one rule (res[name][scenario] = outcome) and against A+0 (res['A+0'])."""
    x = {nm: V.numbers(v) for nm, v in res[name].items()}; lo = {k: min(y[k] for y in x.values()) for k in x[B_NAME]}; hi = {k: max(y[k] for y in x.values()) for k in x[B_NAME]}
    R = dict(zip(V.P_KEYS, (SC.stable(lo['E'], hi['E']), SC.stable(lo['V'], hi['V']), SC.stable(lo['E_ny'], hi['E_ny']), SC.stable(lo['V_ny'], hi['V_ny']),
                            V.above(lo['score'], hi['score'], 'KEEP'), V.above(lo['vs_pullall'], hi['vs_pullall'], 'PULL-ALL'),
                            SC.at_m250(x[B_NAME]['E'], x[M250_NAME]['E']), SC.at_m250(x[B_NAME]['V'], x[M250_NAME]['V']),
                            SC.b1a_sign(res[name][B_NAME]['E'], res[name][B_NAME]['b1a_E']))))
    d = {}
    for nm in res[name]:
        a, b = V.numbers(res[name][nm]), V.numbers(res['A+0'][nm]); d[nm] = dict(dV=a['V'] - b['V'], dE=a['E'] - b['E'], dV_ny=a['V_ny'] - b['V_ny'])
    lo = {k: min(y[k] for y in d.values()) for k in d[B_NAME]}; hi = {k: max(y[k] for y in d.values()) for k in d[B_NAME]}
    R.update(zip(V.D_KEYS, (V.above(lo['dV'], hi['dV'], 'BASELINE'), V.above(lo['dE'], hi['dE'], 'BASELINE'), V.above(lo['dV_ny'], hi['dV_ny'], 'BASELINE'),
                            SC.at_m250(d[B_NAME]['dV'], d[M250_NAME]['dV']))))
    return R, d


def controls(res, day, O, Fl, hit, report, kp_json, pred, exp_rows, parity):
    """P0a-P0f -> the failures (none: pass)."""
    bad = []; J = json.loads(Path(report).read_text())
    for nm, v in res['A+0'].items():                                                        # P0a
        ref = J['scenarios'].get(nm, {}).get('by_day', {}).get(day)
        if ref is None or abs(v['E'] - float(ref)) > 1e-6 * max(1., abs(float(ref))):
            bad.append(f'P0a {nm}: A+0 E {v["E"]:,.3f} vs the committed report {ref}')
    D = json.loads(Path(kp_json).read_text())['D']['M50_L37']['A+0']['policy_by_day']; i = KP.DAYS.index(day); b = res['A+0'][B_NAME]   # P0b
    if not (int(round(D['n'][i])) == b['pulls'] and abs(D['imp'][i] - b['score']) <= 1e-6 * max(1., abs(D['imp'][i]))):
        bad.append(f"P0b: A+0 at B {b['pulls']:,} pulls / {b['score']:,.3f} vs the committed keep / pull run {D['n'][i]:,.0f} / {D['imp'][i]:,.3f}")
    Ee = O[O.status == 'eligible'].reset_index(drop=True); cell = Ee.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values   # P0c
    T = trigger_fn(hit)(Ee); pf = A.per_fill(T, cell, A.endj_of(Ee.t_full.values, Ee.g_struct.values), A.fills_of(Ee, Fl, SC.SHIFT)); m = res[RULE][B_NAME]
    e_, s_ = float(pf['score'].sum() - pf['null'].sum()), float(pf['score'].sum())
    if not (abs(m['E'] - e_) < 1e-6 and abs(m['score'] - s_) < 1e-6 and pf['pulls'] == m['pulls']):
        bad.append(f"P0c: {RULE} at B E {m['E']:,.3f} / improvement {m['score']:,.3f} / pulls {m['pulls']:,}, the clock audit {e_:,.3f} / {s_:,.3f} / {pf['pulls']:,}")
    for nm in res['A+0']:                                                                   # P0d
        if len({(res[r][nm]['KEEP'], res[r][nm]['PULLALL'], res[r][nm]['fills'], res[r][nm]['eligible']) for r in res}) != 1:
            bad.append(f'P0d {nm}: KEEP / PULL-ALL / fills / eligible differ between the rules')
    bad += [f'P0d {r}: KEEP is not the sum of its fills in {k}' for r in res for k in [SC.keep_not_fills(res[r])] if k]
    pr = pred[pred.kind == 'retain']; got = set(zip(pr.oid.astype(int), pr.j.astype(int)))   # P0e
    if got != exp_rows:
        bad.append(f'P0e: {len(got - exp_rows):,} predictions for rows the export does not have, {len(exp_rows - got):,} exported eligible RETAIN rows without one')
    want = {}                                                                               # P0g: the executed triggers, recomputed by a loop
    for (dy, oid), g in pred[pred.kind == 'retain'].groupby(['day', 'oid'], sort=False):
        for j, v, ok in sorted(zip(g.j.values.astype(int), g.J_V.values, g.J_ok.values.astype(bool))):
            if ok and ((v > 0) if FLIP else (v < 0)):
                want[(str(dy), int(oid))] = j; break
    hitd = {(str(k[0]), int(k[1])): int(v) for k, v in hit.items()}; off = {k for k in set(want) | set(hitd) if want.get(k) != hitd.get(k)}
    if off:
        bad.append(f'P0g: the executed pull steps differ from the first coherent predicted loss on {len(off):,} orders')
    comp, refd, agree = parity                                                              # P0f
    if not (comp > refd and agree == comp):
        bad.append(f'P0f: the rule agrees with the contract\'s decide_one on {agree:,} of {comp:,} comparable rows ({refd:,} atoms refused)')
    return bad


def run(day, out=None, sc_out=None, kp=None, bench=None, export=None, report=None, pred=None):  # noqa: C901
    """MODEL-J and A+0 in every scenario of the bank on one day -> policy_<day>.json / .txt / .csv."""
    start = time.time(); C.require(day in KP.DAYS, 'undeclared day'); out = Path(out or OUT); sc_out = Path(sc_out or SC.OUT); kp = Path(kp or KP.OUT)
    bench = Path(bench or BENCH); export = Path(export or EXPORT); report = Path(report or sc_out / 'scenarios.json'); C.require(not (out / f'policy_{day}.json').exists(), 'output exists')
    bj = json.loads((bench / 'bench.json').read_text()); pf_ = bench / f'pred_{day}.parquet'
    C.require(pred is not None or bj['outputs'].get(pf_.name) == C.sha(pf_), f'P0e: the prediction file {pf_.name} does not match the benchmark receipt')
    pred = pd.read_parquet(pf_) if pred is None else pred; pred = pred[pred.day.astype(str) == day]
    er = json.loads((export / day / 'receipt.json').read_text()); exp_rows = set()
    for x, h in er['outputs'].items():
        C.require(C.sha(export / day / x) == h, f'P0e: export part {x} changed since its receipt')
        g = pd.read_parquet(export / day / x, columns=['oid', 'kind', 'j', 'status']); g = g[(g.kind == 'retain') & (g.status == 'eligible')]
        exp_rows |= set(zip(g.oid.astype(int), g.j.astype(int)))
    O, Fl, S, SF, Op, rec = V.load([day], sc_out, kp); hit = triggers_of(pred); fn = trigger_fn(hit); res = {'A+0': {}, RULE: {}}
    for sc in SC.SCEN:
        Os, Fs = V.scen_inputs(sc, Op, Fl, S, SF); nm = SC.name(sc)
        res['A+0'][nm] = SC.outcome(O, Os, Fs, sc); res[RULE][nm] = SC.outcome(O, Os, Fs, sc) if MISS_TRIGGERS else SC.outcome(O, Os, Fs, sc, triggers=fn)
    res[RULE][B_NAME]['KEEP'] += KEEP_SHIFT
    tr = bj.get('train'); C.require(bool(tr), 'P0e: the benchmark records no training days')
    trained = float(I._ms(f'{max(tr)[:4]}-{max(tr)[4:6]}-{max(tr)[6:]}') + 86400000)             # the end of the last training day
    C.guard('pipeline policy'); parity = contract_parity(pred, trained)
    bad = controls(res, day, O, Fl, hit, report, kp / 'keep_pull.json', pred, exp_rows, parity)
    if bad:
        sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(bad[:12]))
    R, d = rule_readings(res, RULE); gate = bj.get('gate', {}).get('retain', 'MISSING'); R['BENCH RETAIN ECONOMIC MAPPING GATE'] = gate
    R['BENCH RETAIN RANK / QUANTITY DIAGNOSTIC'] = bj.get('diagnostic', {}).get('retain', 'MISSING'); R['CONTROLS'] = 'YES'
    R['INCOHERENT RETAIN FORECASTS (KEPT, THE CONTRACT\'S FALLBACK)'] = int((~pred[pred.kind == 'retain'].J_ok.astype(bool)).sum())
    nms = list(res['A+0']); P = pd.DataFrame([dict(rule=r, scenario=nm, E=res[r][nm]['E'], score=res[r][nm]['score'], V=res[r][nm]['KEEP'] + res[r][nm]['score'],
                                                    KEEP=res[r][nm]['KEEP'], PULLALL_improvement=res[r][nm]['PULLALL'], PULLALL_value=res[r][nm]['KEEP'] + res[r][nm]['PULLALL'],
                                                    pulls=res[r][nm]['pulls'], E_noyoung=res[r][nm]['no_young']['E'],
                                                    V_noyoung=res[r][nm]['no_young']['fills_value'] + res[r][nm]['no_young']['score'],
                                                    E_b1a_M50=res[r][nm]['b1a_E']['M50'], pending_share_of_gross=res[r][nm]['cancel_pending']['share_of_gross'])
                                               for r in res for nm in nms])
    L = [f'THE LEARNED KEEP / PULL RULE {RULE} THROUGH THE SCENARIO BANK, DECISIONS FIXED ACROSS SCENARIOS (pf_pipeline_policy.py v2; {day}; c summed; pull at the first '
         f'live step whose contract-coherent predicted RETAIN value is below 0; the benchmark\'s RETAIN economic gate: {gate}; local marks at END, not P&L)',
         f"  {len(Op):,} states with a path ({int((Op.status == 'eligible').sum()):,} eligible); MODEL-J pulls {int(len(hit)):,} orders at a predicted loss; contract parity "
         f'{parity[2]:,} / {parity[0]:,} ({parity[1]:,} atoms refused); CONTROLS: YES (P0a-P0f)', '',
         f"  {'scenario':<24} {'E A+0':>11} {'E MODEL-J':>11} {'V A+0':>12} {'V MODEL-J':>12} {'dV':>11} {'pulls A+0':>10} {'pulls J':>9} {'V KEEP':>12} {'V PULL-ALL':>11}   (V = value; PULL-ALL value = KEEP + its improvement)"]
    for nm in nms:
        a, m = res['A+0'][nm], res[RULE][nm]
        L.append(f"  {nm:<24} {a['E']:>+11,.0f} {m['E']:>+11,.0f} {a['KEEP'] + a['score']:>+12,.0f} {m['KEEP'] + m['score']:>+12,.0f} {d[nm]['dV']:>+11,.0f} "
                 f"{a['pulls']:>10,} {m['pulls']:>9,} {a['KEEP']:>+12,.0f} {a['KEEP'] + a['PULLALL']:>+11,.0f}")
    mb = res[RULE][B_NAME]; ab = mb['by_age']
    L += ['', f"  C1 / C3 / C12 at B ({RULE}): E by age {ab['lt100']['E']:+,.0f} / {ab['100to500']['E']:+,.0f} / {ab['ge500']['E']:+,.0f}; B1a {mb['b1a_E']['M0']:+,.0f} / "
          f"{mb['b1a_E']['M50']:+,.0f}; cancel-pending {mb['cancel_pending']['fills']:,} fills / {mb['cancel_pending']['value']:+,.0f} c / {100 * mb['cancel_pending']['share_of_gross']:.1f}%; "
          'by source ' + ' | '.join(f"{sl} {sum(mb['by_age_src'][lab][sl]['E'] for lab in SC.AGE_LABELS):+,.0f}" for sl in SC.SRC_LABELS), '',
          'READINGS (declared before the predictions were read)'] + [f'  {k}: {v}' for k, v in R.items()]
    L += ['', 'LIMITS: one consumed day (G = 1); the model is one of the benchmark\'s 8 candidates; the model\'s own-fill inputs are B\'s in every scenario; local marks, not '
          'P&L; queue rule M only (C5); no order-level validation (C10).']
    out.mkdir(parents=True, exist_ok=True); P.to_csv(out / f'policy_{day}.csv', index=False); (out / f'policy_{day}.txt').write_text('\n'.join(L) + '\n')
    C.dump(out / f'policy_{day}.json', dict(asof=C.stamp(), day=day, readings=R, diffs=d, values={r: {nm: {k: x for k, x in v.items() if not k.startswith('_')} for nm, v in res[r].items()} for r in res},
                                            pulled_orders=int(len(hit)), parity=dict(compared=parity[0], refused=parity[1], agree=parity[2]),
                                            inputs=dict(bench=C.sha(bench / 'bench.json'), predictions=C.sha(pf_) if pf_.exists() else None, export_receipt=C.sha(export / day / 'receipt.json'),
                                                        report=C.sha(report), keep_pull=C.sha(kp / 'keep_pull.json'), days=rec),
                                            outputs={x: C.sha(out / x) for x in (f'policy_{day}.txt', f'policy_{day}.csv')},
                                            source_sha256=C.sha(Path(__file__).resolve()), provenance=KP.provenance(), seconds=time.time() - start))
    print('\n'.join(L), flush=True)
    return res, R


def selftest():
    """Synthetic tape only (see the module's SELFTEST)."""
    global FLIP, MISS_TRIGGERS, KEEP_SHIFT, RAW_TRIGGERS
    bad = []; src = KP.Syn(); days = list(KP.DAYS); d0 = days[0]
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
        O = pd.read_parquet(tmp / 'kp' / d0 / 'orders.parquet'); el = O[O.status == 'eligible']
        endj = A.endj_of(el.t_full.values, el.g_struct.values); rows = []; want = {}
        for o, e in zip(el.itertuples(), endj):                                              # planted predictions: negative from a chosen step
            js = np.arange(KP.J0, max(int(e), KP.J0))
            if not len(js):
                continue
            k = js[len(js) // 2] if int(o.oid) % 3 == 0 else None
            k2 = js[len(js) // 3] if (int(o.oid) % 5 == 0 and k is None) else None                 # v2: a forecast the contract refuses (endpoint mid
            for j in js:                                                                             # far below 0) predicting a loss - it must keep
                bad_ = k2 is not None and j == k2; qm_ = -1e4 if bad_ else (-6. if k is not None and j >= k else 6.)
                rows.append(dict(day=d0, oid=int(o.oid), kind='retain', j=int(j), side=int(o.side), t=float(o.t), decision_ms=float(o.t) + KP.STEP * j, p=float(o.p),
                                 pf_mid=float(o.p) + (.5 if o.side == 0 else -.5), fp_remaining=30., J_P=.4, J_Eq=4., J_Eqm=qm_, J_V=.5 * 4. + qm_, J_ok=not bad_))
            if k is not None:
                want[(d0, int(o.oid))] = int(k)
        pred = pd.DataFrame(rows); hit = triggers_of(pred)
        if not (dict(hit.items()) == want and len(want) > 0):
            bad.append(f'planted triggers: {len(hit)} found, {len(want)} planted')
        exp = tmp / 'exp' / d0; exp.mkdir(parents=True); pred[['oid', 'kind', 'j']].assign(status='eligible').to_parquet(exp / 'rows_000.parquet', index=False)
        C.dump(exp / 'receipt.json', dict(outputs={'rows_000.parquet': C.sha(exp / 'rows_000.parquet')})); bj = tmp / 'bench'; bj.mkdir(); C.dump(bj / 'bench.json', dict(outputs={}, gate=dict(retain='PASS'), diagnostic=dict(retain='PASS'), train=['20231206']))   # the synthetic tape's clock starts 2023-12-07
        rep = tmp / 'sc' / f'scenarios_{d0}.json'
        def go(out_, **kw):
            with contextlib.redirect_stdout(io.StringIO()):
                return run(d0, out=out_, sc_out=tmp / 'sc', kp=kw.pop('kp', tmp / 'kp'), bench=bj, export=tmp / 'exp', report=kw.pop('report', rep), pred=kw.pop('pred', pred))
        R = None
        try:
            res, R = go(tmp / 'p')
        except (SystemExit, ValueError) as e:
            bad.append(f'the synthetic policy run refused: {str(e)[:400]}')
        if R is not None:
            Jp = json.loads((tmp / 'p' / f'policy_{d0}.json').read_text()); Pc = pd.read_csv(tmp / 'p' / f'policy_{d0}.csv')
            if not np.allclose(Pc.PULLALL_value, Pc.KEEP + Pc.PULLALL_improvement, rtol=0, atol=1e-6):
                bad.append('PULL-ALL: its value is not KEEP + its improvement in the CSV (review e96321df P2)')
            if not (R['CONTROLS'] == 'YES' and R['BENCH RETAIN ECONOMIC MAPPING GATE'] == 'PASS' and Jp['pulled_orders'] == len(want) and Jp['parity']['compared'] > Jp['parity']['refused'] > 0
                    and R["INCOHERENT RETAIN FORECASTS (KEPT, THE CONTRACT'S FALLBACK)"] > 0 and not (pred.J_ok == False).sum() == 0
                    and all(k in R for k in V.P_KEYS + V.D_KEYS) and res[RULE][B_NAME]['pulls'] == len(want) and res[RULE][B_NAME]['pulls'] != res['A+0'][B_NAME]['pulls']
                    and all(abs(Jp['diffs'][nm]['dV'] - ((res[RULE][nm]['KEEP'] + res[RULE][nm]['score']) - (res['A+0'][nm]['KEEP'] + res['A+0'][nm]['score']))) < 1e-6 for nm in res[RULE])):
                bad.append(f"synthetic policy run: {R}, pulled {Jp['pulled_orders']} of {len(want)}")
        def refused(name, msg, **kw):
            o_ = tmp / ('z' + str(abs(hash(name)) % 100000))
            try:
                go(o_, **kw); bad.append(f'the policy run accepted {name}')
            except (SystemExit, ValueError) as e:
                if not all(x in str(e) for x in ((msg,) if isinstance(msg, str) else msg)):          # every named control must refuse
                    bad.append(f'{name} refused by another control: {str(e)[:300]}')
            if (o_ / f'policy_{d0}.json').exists():
                bad.append(f'a refused policy run wrote output ({name})')
        extra = pd.concat([pred, pred.iloc[:1].assign(j=KP.NSTEP + 5)], ignore_index=True)
        refused('a prediction for a row the export does not have', 'P0e', pred=extra)
        refused('an exported row without a prediction', 'P0e', pred=pred.iloc[1:])
        flip = pred.copy(); i_ = flip.index[flip.J_ok.astype(bool) & (flip.J_V > 0)][0]; flip.loc[i_, 'J_ok'] = False   # the contract accepts it, the flag
        refused('a coherence flag that disagrees with the contract (the action alike)', 'P0f', pred=flip)                # says refused: only the verdict differs
        J0 = json.loads(rep.read_text()); J0['scenarios'][B_NAME]['by_day'][d0] += 1.; bad_rep = tmp / 'rep_bad.json'; bad_rep.write_text(json.dumps(J0))
        refused('a changed committed report value', 'P0a', report=bad_rep)
        kpb = tmp / 'kpb'; shutil.copytree(tmp / 'kp', kpb); Jk = json.loads((kpb / 'keep_pull.json').read_text())
        Jk['D']['M50_L37']['A+0']['policy_by_day']['imp'][KP.DAYS.index(d0)] += 1.; (kpb / 'keep_pull.json').write_text(json.dumps(Jk))
        refused('a changed committed keep / pull value', 'P0b', kp=kpb)
        for flag, val, name, msg in (('FLIP', True, 'the rule with its sign reversed', 'P0f'), ('MISS_TRIGGERS', True, 'the model rule valued as A+0', 'P0c'),
                                     ('RAW_TRIGGERS', True, 'the rule acting on forecasts the contract refuses', 'P0g'),
                                     ('KEEP_SHIFT', 1., 'a KEEP drift in the model rule', ('differ between the rules', 'KEEP is not the sum of its fills'))):
            globals()[flag] = val
            try:
                refused(name, msg)
            finally:
                FLIP, MISS_TRIGGERS, KEEP_SHIFT, RAW_TRIGGERS = False, False, 0., False
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print('pipeline policy selftests passed (synthetic only): planted predictions give the planted trigger steps; MODEL-J and A+0 valued through the bank '
          'with every control passing (A+0 = the committed report and keep / pull run, MODEL-J = the clock audit, contract parity); refusals by their own '
          'control with no output: a prediction for a row the export lacks and a missing prediction (P0e), a changed committed report value (P0a) and '
          'keep / pull value (P0b), the model rule valued as A+0 (P0c), a KEEP drift (P0d), the rule with its sign reversed (P0f), a coherence flag that '
          'disagrees with the contract (P0f), the rule acting on refused forecasts (P0g)', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif len(a) == 2 and a[0] == 'day':
        run(a[1])
    else:
        sys.exit('usage: pf_pipeline_policy.py selftest | day <day>')
