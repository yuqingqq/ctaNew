"""MUTATION CHECK of pf_pi0_admission (PI0-ADMISSION-EXPOSURE-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_pi0_admission_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_pi0_admission.py'
SELFTEST = 'pf_pi0_admission_selftest'
FAIL = 'PI0 ADMISSION SELFTEST FAILED'
MUTANTS = [
    ('the census over excluded worlds', "            if r['status'] != 'ok':\n                continue\n            for d in r['episodes']['PI0']", "            for d in r['episodes']['PI0']"),
    ('the cell from the world start', "u = datetime.datetime.fromtimestamp((int(t_ms) + float(decision_ms)) / 1000., tz=datetime.timezone.utc)",
     "u = datetime.datetime.fromtimestamp(int(t_ms) / 1000., tz=datetime.timezone.utc)"),
    ('five per cell', "DOMAIN, PER_CELL, AUDIT_EVERY, NT, STEP, END = 'pi0_admission_exposure_v1|', 4,", "DOMAIN, PER_CELL, AUDIT_EVERY, NT, STEP, END = 'pi0_admission_exposure_v1|', 5,"),
    ('an unsalted rank', "    return hashlib.sha256((DOMAIN + key_of(g)).encode()).hexdigest()", "    return hashlib.sha256((key_of(g)).encode()).hexdigest()"),
    ('the weight inverted', "w = n / min(per_cell, n)", "w = min(per_cell, n) / n"),
    ('the snapshot after the decision', "            if key in pol.points:\n                snaps[key] = L.ctx_of(copy.deepcopy(B), k, s, B.episode)\n            BE.decide(B, g, s, pol)",
     "            BE.decide(B, g, s, pol)\n            if key in pol.points:\n                snaps[key] = L.ctx_of(copy.deepcopy(B), k, s, B.episode)"),
    ('the factual decisions hash unchecked', " and L._h(Lf['decisions']) == saved['decisions_sha256']", ""),
    ('the horizon unchecked', "horizon = bool(float(Bf.end) == END == float(saved['end'])", "horizon = bool(True or float(Bf.end) == END == float(saved['end'])"),
    ('admission unchecked', "        if d0 is None or d0[3] != 'POST':\n            out.append(dict(r, status='not_admitted')); continue\n", ""),
    ('the gid unchecked', "        if int(gidf) != int(g['gid']):\n            out.append(dict(r, status='gid_mismatch')); continue\n", ""),
    ('the override on any live PULL', "and v['state'] == 'live' and int(v['side']) == self.focal and v['gid'] == self.gid0 and d.get('action') == 'PULL'",
     "and v['state'] == 'live' and d.get('action') == 'PULL'"),
    ('the prefix ignores fills', "\n                and [f for f in RT.jl(Bk.E.fills) if float(f[0]) <= g] == [f for f in RT.jl(Bf.E.fills) if float(f[0]) <= g])", ")"),
    ('the equivalence ignores the fills', " and Lh['status'] == 'ok' and fh == fk)", " and Lh['status'] == 'ok')"),
    ('the reconciliation ignores the value', "    return bool(abs(held_value - d_selected) <= 1e-6 and ", "    return bool("),
    ('the reconciliation ignores the pre-effect fills', "factual_fills == [x for x in held_fills if float(x[0]) < eff] and ", ""),
    ('the reconciliation ignores the removal time', " and float(removed_at) == float(eff))", ")"),
    ('the review target from activation', "stages[nm]['target'] = RT.stage_target(fh, gid, s, mark, stages[nm]['effect'], ref)", "stages[nm]['target'] = RT.stage_target(fh, gid, s, mark, float(gen_h['act']), ref)"),
    ('an unweighted mean', "    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else None", "    return float(x[m].mean()) if m.any() else None"),
    ('the economics over every status', "    ok = G[G.status == 'ok']; out = []\n    for (day, side), g in [(('all', 'all'), ok)]", "    ok = G; out = []\n    for (day, side), g in [(('all', 'all'), ok)]"),
    ('MSE as an absolute error', "mse_V=wmean((g.V - g.value) ** 2, w)", "mse_V=wmean((g.V - g.value).abs(), w)"),
    ('unmatched strata', "rng = np.random.default_rng(seed); grp = [g for _, g in D.groupby(['day', 'side', 'hour'], sort=True)]", "rng = np.random.default_rng(seed); grp = [D]"),
    ('the percentile reversed', "percentile=float((null <= ms + 1e-12).mean())", "percentile=float((null >= ms - 1e-12).mean())"),
    ('never thin', "thin=bool(n_sel < THIN or len(D) - n_sel < THIN))", "thin=False)"),
    ('a new observation on any age change', "d[f'new_{ag}'] = bool(float(a1[i]) < float(a0[i]) + dt - 1.)", "d[f'new_{ag}'] = bool(float(a1[i]) != float(a0[i]))"),
    ('the PF receipt ignored', "d['new_pf_receipt'] = bool((x.get('view') or {}).get('received') != (ent.get('view') or {}).get('received'))", "d['new_pf_receipt'] = False"),
    ('the thin floor ignored', "admitted_longer_beneficial=bool(len(days) == 2 and not thin and all(x > 0 for x in sg))", "admitted_longer_beneficial=bool(len(days) == 2 and all(x > 0 for x in sg))"),
    ('one negative day suffices', "admitted_longer_harmful=bool(len(days) == 2 and not thin and all(x < 0 for x in sg))", "admitted_longer_harmful=bool(len(days) == 2 and not thin and any(x < 0 for x in sg))"),
    ('the missing admissions over every fork', "int(((J.status == 'ok') & J.admission.isna()).sum())", "int(J.admission.isna().sum())"),
    ('the retention flips on every review', "rv = S[(S.stage == 'first_review') & (S.dec_status == 'model') & S.V.notna() & (S.stage_status == 'ok')]", "rv = S[(S.stage == 'first_review') & S.V.notna() & (S.stage_status == 'ok')]"),
]
