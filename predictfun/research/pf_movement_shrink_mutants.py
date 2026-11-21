"""MUTATION CHECK of pf_movement_shrink (PI0-MOVEMENT-SHRINK-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_movement_shrink_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_movement_shrink.py'
SELFTEST = 'pf_movement_shrink_selftest'
FAIL = 'MOVEMENT SHRINK SELFTEST FAILED'
MUTANTS = [
    # support, bounds and the prior
    ('the row floor at 31', "MIN_ROWS, MIN_FILLS, DEN_EPS, TOL = 30, 10, 1e-12, 1e-9", "MIN_ROWS, MIN_FILLS, DEN_EPS, TOL = 31, 10, 1e-12, 1e-9"),
    ('the fill floor at 9', "MIN_ROWS, MIN_FILLS, DEN_EPS, TOL = 30, 10, 1e-12, 1e-9", "MIN_ROWS, MIN_FILLS, DEN_EPS, TOL = 30, 9, 1e-12, 1e-9"),
    ('the bid bounds for asks', "    return (-mid, 100. - mid) if int(side) == 0 else (mid - 100., mid)", "    return (-mid, 100. - mid)"),
    ('the projection unbounded below', "    a = np.minimum(np.maximum(raw, lo), hi); a = np.where(Eq == 0, 0., a)", "    a = np.minimum(raw, hi); a = np.where(Eq == 0, 0., a)"),
    ('the upper bind unflagged', "    return a, (raw < lo) | (raw > hi)", "    return a, (raw < lo)"),
    # the fit
    ('a training row of the evaluation day', "C.require(len(T) == 0 or (set(T.day.astype(str)) == {TRAIN_DAY}), 'chronology", "C.require(len(T) == 0 or (True), 'chronology"),
    ('a training row of another side', "C.require(len(T) == 0 or set(T.side.astype(int)) == {int(side)}, 'a training row", "C.require(len(T) == 0 or True, 'a training row"),
    ('mu over every opportunity', "    mu = float((w * M).sum() / (w * z).sum());", "    mu = float((w * M).sum() / w.sum());"),
    ('lambda unclipped', "lam=float(min(1., max(0., num / den)))", "lam=float(num / den)"),
    ('lambda against M, not M - b', "num = float((w * d * (M - b)).sum())", "num = float((w * d * M).sum())"),
    ('lambda unweighted', "den = float((w * d * d).sum()); num", "den = float((d * d).sum()); num"),
    ('the zero denominator fitted', "    if den <= DEN_EPS:\n", "    if den < 0.:\n"),
    ('the support floor ignored', "    if n < MIN_ROWS or nf < MIN_FILLS:\n", "    if n < MIN_ROWS:\n"),
    # the arms
    ('the prior arm with the fitted lambda', "('prior', np.where(np.isfinite(E.mu.to_numpy(float)), 1., 0.))", "('prior', E.lam.to_numpy(float))"),
    ('V without lambda', "E[f'V_{arm}'] = E.V.to_numpy(float) + lam * d", "E[f'V_{arm}'] = E.V.to_numpy(float) + d"),
    ('E[qm] without lambda', "E[f'Eqm_{arm}'] = E.Eqm.to_numpy(float) + lam * d;", "E[f'Eqm_{arm}'] = E.Eqm.to_numpy(float) + d;"),
    ('an unsupported group gets the prior', "        if g['mu'] is not None and m.any():\n", "        if m.any() and g['mu'] is not None or m.any():\n"),
    ('the entry tie POSTs', "    rule = (lambda v: v > 0) if head == 'entry' else (lambda v: v >= 0)", "    rule = (lambda v: v >= 0) if head == 'entry' else (lambda v: v >= 0)"),
    ('the review tie PULLs', "    rule = (lambda v: v > 0) if head == 'entry' else (lambda v: v >= 0)", "    rule = (lambda v: v > 0) if head == 'entry' else (lambda v: v > 0)"),
    # scores
    ('the movement MSE unweighted', "mse_m=wm(em ** 2, w)", "mse_m=float((em ** 2).mean())"),
    ('the conditional error unweighted', "c_m=wm(cm, f.weight) if len(f) else None", "c_m=float(cm.mean()) if len(f) else None"),
    ('the Brier unweighted', "brier=wm((g.P - g.z) ** 2, w)", "brier=float(((g.P - g.z) ** 2).mean())"),
    ('the mechanical baseline without edge', "mse_mech=wm((g.edge * g.Eq - g.value) ** 2, w)", "mse_mech=wm((g.Eq - g.value) ** 2, w)"),
    ('J over the chosen weight', "J=float((w * c * v).sum() / W) if W > 0 else None", "J=float((w * c * v).sum() / max(1e-12, float(wc.sum()))) if W > 0 else None"),
    ('severe below -40', "    sev = vc < -SEVERE;", "    sev = vc < -40.;"),
    ('the changed count against the prior', "changed_vs_original=int((mk[rule][E.index.get_indexer(idx)] != mk['original'][E.index.get_indexer(idx)]).sum())",
     "changed_vs_original=int((mk[rule][E.index.get_indexer(idx)] != mk['prior'][E.index.get_indexer(idx)]).sum())"),
    # the null
    ('the null with replacement', "m[rng.choice(g, k, replace=False)] = True", "m[rng.choice(g, k, replace=True)] = True"),
    ('the weight mismatch uncounted', "        wmis += int(abs(float((w * m).sum()) - Wc) > 1e-9 * max(1., Wc))\n", "        wmis += 0\n"),
    ('degenerate strata as free', "free = [(g, k) for g, k in zip(grp, ks) if 0 < k < len(g)]", "free = [(g, k) for g, k in zip(grp, ks) if 0 < k]"),
    ('the percentile strict', "percentile=float((J <= Jc + 1e-12).mean())", "percentile=float((J < Jc - 1e-12).mean())"),
    ('the null over side only', "[(E.cday == d).to_numpy() & (E.side == s).to_numpy() & (E.chour == h).to_numpy()", "[(E.cday == d).to_numpy() & (E.side == s).to_numpy()"),
    # the readings
    ('PASS without the p95', "\n          and J['candidate'] > NU['null_p95']):", "):"),
    ('PASS without the prior', " and J['candidate'] > J['original'] and J['candidate'] > J['prior']\n", " and J['candidate'] > J['original']\n"),
    ('PASS without the original', " and J['candidate'] > J['original'] and J['candidate'] > J['prior']\n", " and J['candidate'] > J['prior']\n"),
    ('PASS on a non-positive J', " and J['candidate'] > 0 and J['candidate'] > J['original']", " and J['candidate'] > J['original']"),
    ('equal original masks allowed', "same_o = bool((mk['candidate'] == mk['original']).all())", "same_o = False"),
    ('equal prior masks allowed', "    same_p = bool((mk['candidate'] == mk['prior']).all());", "    same_p = False;"),
    ('thin support allowed', "support = bool(NU['free_strata'] >= MIN_FREE_STRATA and NU['free_rows'] >= MIN_FREE_ROWS)", "support = bool(NU['free_strata'] >= 1)"),
    ('the null weight mismatch allowed', "('null_weight_mismatch', NU['weight_mismatch_draws'] > 0)", "('null_weight_mismatch', False)"),
    ('no fit support allowed', "('no_fit_support', not fit_ok)", "('no_fit_support', False)"),
    ('the prediction boolean on bias', "movement_mse_improves=bool(s[(side, 'candidate')]['mse_m'] < s[(side, 'original')]['mse_m'])",
     "movement_mse_improves=bool(abs(s[(side, 'candidate')]['bias_m']) < abs(s[(side, 'original')]['bias_m']))"),
    # rows, populations and the guards
    ('the audit mismatch ignored', "                O.loc[i, 'status'] = 'audit_mismatch'\n", "                O.loc[i, 'status'] = 'ok'\n"),
    ('the reference without V', "COLS = ('P', 'Eq', 'Eqm', 'V', 'edge', 'q', 'value', 'z', 'M')", "COLS = ('P', 'Eq', 'Eqm', 'edge', 'q', 'value', 'z', 'M')"),
    ('populations over every status', "pick = lambda day, nm: sorted(O[(O.status == 'ok') & (O.day == day) & (O.stage == nm)].key)",
     "pick = lambda day, nm: sorted(O[(O.day == day) & (O.stage == nm)].key)"),
    ('the cell from the world start', "    return PA.cell_of(int(t_ms), float(t_dec), int(side))", "    return PA.cell_of(int(t_ms), 0., int(side))"),
    ('the mid of the entry at every stage', "(r['stages'].get(nm) or {}).get('inputs')[RT.FEATS.index('pf_mid')])", "(r['stages'].get('entry') or {}).get('inputs')[RT.FEATS.index('pf_mid')])"),
    ('a changed population accepted', "C.require(man['version'] == VERSION and pop['train'] == man['train'] and pop['eval'] == man['eval'] and pop['descriptive'] == man['descriptive'],",
     "C.require(man['version'] == VERSION and pop['eval'] == man['eval'] and pop['descriptive'] == man['descriptive'],"),
    ('the input hashes unchecked', "    C.require(man['admission_manifest_sha256'] == msha and man['audit_opportunities_sha256'] == opp_sha and man['admission_records_sha256'] == rec_shas,",
     "    C.require(man['admission_manifest_sha256'] == msha,"),
    ('evaluation rows of the training day', " and (E.status == 'ok').all() and set(E.day) <= {EVAL_DAY}, f'{nm}: the evaluation rows differ", " and (E.status == 'ok').all(), f'{nm}: the evaluation rows differ"),
    ('frozen without the manifest', "for n in ('declaration.md', 'manifest.json')))", "for n in ('declaration.md',)))"),
    ('inputs without the records', "    return bool(MA.inputs_frozen() and all(committed_equal(f'{ADM_REC_REL}/part_{int(k):03d}.json'", "    return bool(MA.inputs_frozen() or all(committed_equal(f'{ADM_REC_REL}/part_{int(k):03d}.json'"),
    ('the fit written over', "C.require(not path.exists(), 'the fit exists (write-once)')", "C.require(True, 'the fit exists (write-once)')"),
    ('the scores written over', "C.require(not (out / 'audit.json').exists(), 'the scores exist (immutable)')", "C.require(True, 'the scores exist (immutable)')"),
    ('a score before the fit commit', "    C.require(committed_equal(f'{RECEIPT}/fit.json', rec / 'fit.json'), 'the fit is not committed", "    C.require(True, 'the fit is not committed"),
    ('a fit of another identity', "    C.require(fit['sources'] == sources() and fit['manifest_sha256'] == C.sha(rec / 'manifest.json'), 'the fit is of another identity')",
     "    C.require(True, 'the fit is of another identity')"),
    ('run before the freeze', "    C.require(frozen() and inputs_bound(), 'not frozen: the module / selftest / declaration / manifest or an input is not the committed one')\n    man = json.loads((rec / 'manifest.json').read_text()); ar, msha, O = load()",
     "    C.require(True, 'not frozen: the module / selftest / declaration / manifest or an input is not the committed one')\n    man = json.loads((rec / 'manifest.json').read_text()); ar, msha, O = load()"),
]

if __name__ == '__main__':
    base = SRC.read_text(); bad = [(i, n) for i, (n, o, _) in enumerate(MUTANTS) if base.count(o) != 1]
    print(f'{len(MUTANTS)} mutants; targets not exactly once: {bad}')
