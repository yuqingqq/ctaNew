"""MUTATION CHECK of pf_movement_audit (PI0-MOVEMENT-CALIBRATION-AUDIT-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_movement_audit_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_movement_audit.py'
SELFTEST = 'pf_movement_audit_selftest'
FAIL = 'MOVEMENT AUDIT SELFTEST FAILED'
MUTANTS = [
    # accounting
    ('every opportunity a fill', "z = 1. if q > 1e-12 else 0.; M = (float(tg['qm'])", "z = 1. if q >= 0 else 0.; M = (float(tg['qm'])"),
    ('the probability term without z', "mov_prob=(P - z) * mh,", "mov_prob=P * mh,"),
    ('the conditional term on every opportunity', "mov_cond=z * (mh - M),", "mov_cond=(mh - M),"),
    ('the identity residual unchecked', "    if out['identity_residual'] > TOL:\n", "    if out['identity_residual'] > 1e9:\n"),
    ('zero_P reported ok', "        return dict(base, acc_status='zero_P')", "        return dict(base, acc_status='ok')"),
    ('a missing qm as zero', "M = (float(tg['qm']) if tg.get('qm') is not None else np.nan) if z else 0.", "M = (float(tg['qm']) if tg.get('qm') is not None else 0.) if z else 0."),
    ('the mechanical baseline on the realized q', "c_q=(qh - q) if z else np.nan, mech=edge * Eq)", "c_q=(qh - q) if z else np.nan, mech=edge * q)"),
    # fills
    ('fills before the effect counted', "        if t < eff or int(f[1]) != int(r['held']['gid']):\n", "        if int(f[1]) != int(r['held']['gid']):\n"),
    ('other gids counted', "        if t < eff or int(f[1]) != int(r['held']['gid']):\n", "        if t < eff:\n"),
    ('the entry from the review effect', "    return float(r['held']['act']) if nm == 'entry' else float(x['effect'])", "    return float(x['effect'])"),
    ('the pre-fill sign reversed', "pre=np.nan if fm is None else sg * q * (fm - dmid)", "pre=np.nan if fm is None else sg * q * (dmid - fm)"),
    ('the ask side as the bid', "    s = int(r['side']); sg = 1. if s == 0 else -1.; mark = float(r['held']['mark'])", "    s = int(r['side']); sg = 1.; mark = float(r['held']['mark'])"),
    ('the forced mid without the side', "mid=float(f['price']) + float(f['fill_time_edge']) / (sg * float(f['q']))", "mid=float(f['price']) + float(f['fill_time_edge']) / float(f['q'])"),
    ('a late reference accepted', "        if ft is not None and not (float(ft) <= t <= END):\n", "        if ft is not None and not (float(ft) <= END):\n"),
    ('no record as a missing mid', "st = 'ok' if m is not None else 'reference_missing' if ok_ref else f\"reference_unavailable:{mref.get('status')}\"",
     "st = 'ok' if m is not None else 'reference_missing'"),
    ('the fill totals unchecked against M', "(abs(sum(f['total'] for f in fl) - d['M']) > TOL or bool(fl) != bool(d['z']))", "bool(fl) != bool(d['z'])"),
    # strata
    ('the delay class boundary', "    if d <= 100.:\n        return '[0,100]'", "    if d < 100.:\n        return '[0,100]'"),
    ('the horizon intervals closed left', "        if lo < t <= hi:\n            return f'({lo:.0f},{hi:.0f}]'", "        if lo <= t < hi:\n            return f'({lo:.0f},{hi:.0f}]'"),
    ('the queue ratio boundary', "return '0' if r == 0 else '(0,1]' if r <= 1 else", "return '0' if r == 0 else '(0,1]' if r < 1 else"),
    ('the decision queue from the venue', "qbin=qbin(d['fp_queue'], d['fp_remaining'])", "qbin=qbin(d['venue_ahead'], d['fp_remaining'])"),
    ('ordinal 2 merged', "'1' if o <= 1 else '2' if o <= 2 else '>=3'", "'1' if o <= 1 else '>=3'"),
    ('the E[q] boundary', "'(3,10]' if e <= 10 else '>10'", "'(3,10]' if e < 10 else '>10'"),
    ('P = 1 outside the bins', "        if (lo <= p < hi) or (hi == 1. and p == 1.):\n", "        if lo <= p < hi:\n"),
    ('non-model decisions eligible', "or x.get('status') != 'ok' or x.get('dec_status') != 'model' or not x.get('inputs'):",
     "or x.get('status') != 'ok' or not x.get('inputs'):"),
    ('fallback as its shadow action', "    return ent.get('admission') if ent.get('dec_status') == 'model' else 'fallback'", "    return ent.get('admission')"),
    # binding and membership
    ('the row weight unbound', "bound = bool(g is not None and float(r.get('weight', np.nan)) == float(g['weight']) and int(r.get('n_cell', -1))",
     "bound = bool(g is not None and int(r.get('n_cell', -1))"),
    ('the row cell unbound', "\n                                                and list(r.get('cell') or []) == list(g['cell']) and int(g['part']) == int(k))",
     "\n                                                and int(g['part']) == int(k))"),
    ('the row part unbound', " and list(r.get('cell') or []) == list(g['cell']) and int(g['part']) == int(k))", " and list(r.get('cell') or []) == list(g['cell']))"),
    ('the weight from the row', "weight=float(g['weight']) if g else np.nan))", "weight=float(r['weight'])))"),
    ('a partition unhashed', "        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['manifest_sha256'] == msha, f'admission part {k}: identity')",
     "        C.require(rec['manifest_sha256'] == msha, f'admission part {k}: identity')"),
    ('membership a subset', "forced_equal=bool(len(f_) == len(set(f_)) == len(fk) and set(f_) == fk)", "forced_equal=bool(set(f_) <= fk)"),
    ('measure without the forced membership', "    C.require(mem['admitted_equal'] and mem['forced_equal'], f'incomplete membership: {mem}')",
     "    C.require(mem['admitted_equal'], f'incomplete membership: {mem}')"),
    ('measure overwrites', "C.require(not (out / 'audit.json').exists(), 'the measurement exists (immutable)')", "C.require(True, 'the measurement exists (immutable)')"),
    ('a mids record unhashed', " and\n                      hashlib.sha256(json.dumps(rec['mids'], sort_keys=True).encode()).hexdigest() == rec['sha256'], f'mids {k}: identity')",
     ", f'mids {k}: identity')"),
    ('mids keyed by string', "mids={int(i): v for i, v in (e.get('mids') or {}).items() if v is not None})", "mids={i: v for i, v in (e.get('mids') or {}).items() if v is not None})"),
    # tables
    ('the overlap is every day', "        ov = g0[g0.day.isin(OVERLAP)]\n", "        ov = g0\n"),
    ('the overlap views for the admitted too', "    if cohort == 'forced':\n        ov = g0[g0.day.isin(OVERLAP)]", "    if True:\n        ov = g0[g0.day.isin(OVERLAP)]"),
    ('the Brier reference is P', "brier_pooled=wm((pool - g.z) ** 2, w)", "brier_pooled=wm((g.P - g.z) ** 2, w)"),
    ('severe as above +30', "sv = (g.value < -SEVERE).to_numpy()", "sv = (g.value < SEVERE).to_numpy()"),
    ('Kish without the square', "return float(w.sum() ** 2 / (w ** 2).sum())", "return float(w.sum() / (w ** 2).sum())"),
    ('weighted concentration unweighted', "        w = ok.weight.astype(float) if wt == 'weighted' else np.ones(len(ok))", "        w = np.ones(len(ok))"),
    ('the top-6 generation share', "top5_generation_share=float(a[:5].sum() / a.sum())", "top5_generation_share=float(a[:6].sum() / a.sum())"),
    ('unweighted contributions', "sd = float((g.weight * g.D_EXPOSURE).sum())", "sd = float(g.D_EXPOSURE.sum())"),
    ('per share without the weights', "pre_per_share=float((ok.weight * ok.pre).sum()) / wq if wq > 0 else None",
     "pre_per_share=float((ok.weight * ok.pre).sum()) / float(ok.q.sum()) if wq > 0 else None"),
    ('missing references uncounted', "missing=int((g.ref_status != 'ok').sum())", "missing=0"),
    ('a changed bookTicker age as new', "new_bookticker=bool(g(a1, 'bn_age') < g(a0, 'bn_age') + dt - 1.)", "new_bookticker=bool(g(a1, 'bn_age') != g(a0, 'bn_age'))"),
    ('every PF receipt new', "new_pf_receipt=bool((v.get('view') or {}).get('received') != (e.get('view') or {}).get('received')), mid_change",
     "new_pf_receipt=True, mid_change"),
    ('the control on the admission hour', "    S.loc[m, 'hour'] = [int(dc[2]) for dc in der]\n", "    pass\n"),
    # predicates
    ('broad on two cells', "'broad' if (len(cells) == 4 and pos >= 3", "'broad' if (len(cells) == 4 and pos >= 2"),
    ('a looser tail share', "pos >= 3 and share is not None and share < .5)", "pos >= 3 and share is not None and share < .8)"),
    ('the ratio dropped', "(dly is None or (dmax is not None and dmax >= RATIO * dly))", "(dly is None or (dmax is not None and dmax >= dly))"),
    ('no_fill among the delay classes', " & S.c_m.notna() & (S.cell != 'no_fill')] if len(S) else S", " & S.c_m.notna()] if len(S) else S"),
    ('every subset in the strata read', "s['cohort'] == 'admitted' and s['subset'] == 'all' and s['stage'] == stage", "s['cohort'] == 'admitted' and s['stage'] == stage"),
    ('material at a full difference', "abs(ca - cf) >= .5 * max(abs(ca), abs(cf))", "abs(ca - cf) >= 1. * max(abs(ca), abs(cf))"),
    ('the forced reference over every day', "ff = FO[FO.day.isin(OVERLAP) & (FO.z > 0)]", "ff = FO[FO.z > 0]"),
    ('populated in one stratification', "iv = 'populated' if all(v >= 2 for v in pop.values()) else 'thin'", "iv = 'populated' if any(v >= 2 for v in pop.values()) else 'thin'"),
    ('populated without fills', "((S.dim == d) & (S.n >= THIN_N) & (S.fills >= THIN_FILLS)).sum()", "((S.dim == d) & (S.n >= THIN_N)).sum()"),
    ('defensible only on decision_state', "ii in ('decision_state', 'both')) else", "ii == 'decision_state') else"),
    ('fill timing not a representation need', " or iv == 'thin' or ii == 'fill_timing') else 'inconclusive')", " or iv == 'thin') else 'inconclusive')"),
    # the tape lookup and the guards
    ('mid_lookup at or after', "    j = int(np.searchsorted(frames[:, 0], t, side='right')) - 1\n    if j < 0:\n        return None\n    b, a = frames[j, 2], frames[j, 12]\n    return dict(",
     "    j = int(np.searchsorted(frames[:, 0], t, side='left')) - 1\n    if j < 0:\n        return None\n    b, a = frames[j, 2], frames[j, 12]\n    return dict("),
    ('mid_lookup on one best', "recv_t=float(frames[j, 1])) if b >= 0 and a >= 0 else None", "recv_t=float(frames[j, 1])) if a >= 0 else None"),
    ('frozen always', "    return bool(all(committed_equal(f'predictfun/research/{n}', HERE / n)", "    return True or bool(all(committed_equal(f'predictfun/research/{n}', HERE / n)"),
    ('inputs_frozen ignores its arguments', "    return bool(committed_equal(ADM_MAN_REL, adm_man) and committed_equal(EXPO_MAN_REL, expo_man))",
     "    return bool(committed_equal(ADM_MAN_REL, ADM / 'manifest.json') and committed_equal(EXPO_MAN_REL, EXPO_MAN))"),
    ('run_part before the freeze', "    C.require(frozen(), 'the module / selftest / declaration is not committed in HEAD: no measurement before the freeze')",
     "    C.require(True, 'the module / selftest / declaration is not committed in HEAD: no measurement before the freeze')"),
]

if __name__ == '__main__':
    base = SRC.read_text(); bad = [(i, n) for i, (n, o, _) in enumerate(MUTANTS) if base.count(o) != 1]
    print(f'{len(MUTANTS)} mutants; targets not exactly once: {bad}')
