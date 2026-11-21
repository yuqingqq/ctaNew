"""MUTATION CHECK of pf_adv2_eval (the offline prediction diagnostic and the local action-information control; review 315979d2's round;
2026-11-21, session aea5bfb9). Each planted error is a textual replacement in a copy of pf_adv2_eval.py, executed under its real path in a
fresh process by pf_mutant_one.py, which runs pf_adv2_eval_selftest.run on it. Required: the control passes; every mutant is caught BY A
CHECK; every target occurs exactly once. Per-mutant jobs: pf_mutant_queue.sh / pf_mutant_one.py / pf_mutant_summary.py."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_adv2_eval.py'
SELFTEST = 'pf_adv2_eval_selftest'
FAIL = 'ADV2 EVAL SELFTEST FAILED'
MUTANTS = [
    ('no exact zero bin', "    if p == 0.:\n        return '0 (exact)'\n", ""),
    ('left-closed bins', "i = int(np.searchsorted(EDGES, p, side='left'))", "i = int(np.searchsorted(EDGES, p, side='right'))"),
    ('the value of the other action', "value=wmean(d * y, w), oracle", "value=wmean((1 - d) * y, w), oracle"),
    ('regret without the oracle', "regret=wmean(np.maximum(y, 0.) - d * y, w)", "regret=wmean(-d * y, w)"),
    ('missed positive of the taken action', "missed_positive=float(np.sum(w * (1 - d) * np.maximum(y, 0.)))", "missed_positive=float(np.sum(w * d * np.maximum(y, 0.)))"),
    ('admitted negative unweighted', "admitted_negative=float(np.sum(w * d * np.minimum(y, 0.)))", "admitted_negative=float(np.sum(d * np.minimum(y, 0.)))"),
    ('meaningful = exact', "MEANINGFUL, TAIL, ABS_TOL, REL_TOL, MIN_ORIG, MIN_NZ = 1e-6, 30.,", "MEANINGFUL, TAIL, ABS_TOL, REL_TOL, MIN_ORIG, MIN_NZ = 0., 30.,"),
    ('a 3 c tail', "MEANINGFUL, TAIL, ABS_TOL, REL_TOL, MIN_ORIG, MIN_NZ = 1e-6, 30.,", "MEANINGFUL, TAIL, ABS_TOL, REL_TOL, MIN_ORIG, MIN_NZ = 1e-6, 3.,"),
    ('the training mean as zero', "wmse_train_mean=wmean((tm - y) ** 2, w)", "wmse_train_mean=wmean(y ** 2, w)"),
    ('the support floor unchecked', "    if st['originals'] < MIN_ORIG or st['meaningful_originals'] < MIN_NZ:", "    if st['meaningful_originals'] < MIN_NZ:"),
    ('the nonzero floor unchecked', "    if st['originals'] < MIN_ORIG or st['meaningful_originals'] < MIN_NZ:", "    if st['originals'] < MIN_ORIG:"),
    ('a zero mean label evaluated', "    if my == 0.:\n        return 'NOT_EVALUABLE', 'zero mean label'\n", ""),
    ('WAIT / PULL need a positive label', "sign_ok = my > 0 if group in ('POST', 'KEEP') else my < 0", "sign_ok = my > 0"),
    ('a wider amount tolerance', "amount_ok = abs(mp - my) <= ABS_TOL + REL_TOL * abs(my)", "amount_ok = abs(mp - my) <= 10 * ABS_TOL + REL_TOL * abs(my)"),
    ('POST on a tie', "act = np.where(ok & (p > 0), 'POST', 'WAIT')", "act = np.where(ok & (p >= 0), 'POST', 'WAIT')"),
    ('PULL on a tie', "act = np.where(ok & (p < 0), 'PULL', 'KEEP')", "act = np.where(ok & (p <= 0), 'PULL', 'KEEP')"),
    ('a keep fallback PULLs', "0. if h == 'post' else 1.)", "0.)"),
    ('unsupported rows decided by the model', "ok = (df.support == 'supported').values & np.isfinite(p)", "ok = np.isfinite(p)"),
    ('the RICH training mean of another head', "df['train_mean'] = fz['models'][f'RICH {s} {h}']['train_weighted_mean_y']", "df['train_mean'] = fz['models'][f'RICH {s} post']['train_weighted_mean_y']"),
    ('extra picks are anchors', "    a = D[((D['head'] == 'post') & (D.pick == 'mandatory')) | ((D['head'] == 'keep') & (D.pick == 'keep'))]", "    a = D[(D['head'] == 'post') | ((D['head'] == 'keep') & (D.pick == 'keep'))]"),
    ('fallback anchors kept', "    a = a[a[f'act_{cand}'] != 'fallback'].copy();", "    a = a.copy();"),
    ('a stratum denominator', "        vals /= W\n", "        vals /= max(len(strata), 1)\n"),
    ('another seed', "DRAWS, SEED = 500, 20261121", "DRAWS, SEED = 500, 1"),
    ('committed always true', "    return r.returncode == 0 and r.stdout == Path(local).read_bytes()", "    return True"),
    ('the evaluation without the committed fit', "    C.require(committed(str(rec / 'fit' / 'fit.json'), out / 'fit.json'), 'the fit manifest is not committed in HEAD: no label is read before the model hashes')\n", ""),
    ('coverage: meaningful = exact', "meaningful_originals=int(len(gv[np.abs(y) > MEANINGFUL][o].drop_duplicates()))", "meaningful_originals=int(len(gv[y != 0.][o].drop_duplicates()))"),
    ('coverage: rows as originals', "originals=int(len(gv[o].drop_duplicates()))", "originals=int(len(gv))"),
    ('coverage: no candidate groups', "for a in GROUPS[h] + ('fallback',)] if cand else [])", "for a in GROUPS[h] + ('fallback',)] if False else [])"),
    ('coverage: a dimension dropped', "COV_DIMS = ('source', 'hbin') + tuple(A.R2.DIMS)", "COV_DIMS = ('source', 'hbin') + tuple(A.R2.DIMS)[1:]"),
    ('no categories on the evaluation rows', "df['hbin'] = T.hbin(A.END - df.decision_ms.astype(float).values); df = with_cats(df, e)", "df['hbin'] = T.hbin(A.END - df.decision_ms.astype(float).values)"),
]
