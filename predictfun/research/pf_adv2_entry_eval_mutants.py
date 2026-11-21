"""MUTATION CHECK of pf_adv2_entry_eval (the ADV2-ENTRY-SMOOTH-v1 offline entry diagnostic; 2026-11-21, session aea5bfb9). Each planted error is
a textual replacement in a copy of pf_adv2_entry_eval.py, run by pf_mutant_one.py with pf_adv2_entry_eval_selftest. Required: the control
passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_adv2_entry_eval.py'
SELFTEST = 'pf_adv2_entry_eval_selftest'
FAIL = 'ADV2 ENTRY EVAL SELFTEST FAILED'
MUTANTS = [
    ('a tie POSTs', "act = np.where(ok & (p > 0), 'POST', 'WAIT')", "act = np.where(ok & (p >= 0), 'POST', 'WAIT')"),
    ('unsupported rows decided', "ok = (df.support == 'supported').values & np.isfinite(p);", "ok = np.isfinite(p);"),
    ('RICH from the SMOOTH heads', "p = N.predict(smooth[(s, h)], e['X']) if cand == N.NAME else A.predict(rich['RICH'][(s, h)], 'RICH', e['X'])", "p = N.predict(smooth[(s, h)], e['X'])"),
    ('the training mean of the other side', "df['train_mean'] = fz['models'][f'{N.NAME} {s} {h}']['train_weighted_mean_y']", "df['train_mean'] = fz['models'][f'{N.NAME} 0 {h}']['train_weighted_mean_y']"),
    ('the identity ignores Y', "y_equal=eq('Y', 'Y')", "y_equal=True"),
    ('the identity ignores w', "w_equal=eq('w', 'w')", "w_equal=True"),
    ('the identity ignores the predictions', "rich_pred_equal=eq('pred_RICH', 'pred_RICH')", "rich_pred_equal=True"),
    ('the identity ignores missing rows', "same_keys = bool(len(a) == len(b) and a.index.equals(b.index) and a.index.is_unique)", "same_keys = True"),
    ('calibration bins without their weight', "w_sum=float(gb.w.sum()), wmean_pred", "w_sum=float(len(gb)), wmean_pred"),
    ('the stratum value unweighted', "f'value_{cand}': AE.wmean(d * y, w)", "f'value_{cand}': float(np.mean(d * y))"),
    ('the predicate value of the all group', "& (rd.candidate == cand) & (rd.group == 'POST')].iloc[0]", "& (rd.candidate == cand) & (rd.group == 'all')].iloc[0]"),
    ('zero admissions admitted', "status='no_admissions' if n == 0 else 'admitted'", "status='admitted'"),
    ('the p95 predicate on the mean', "above_draw_p95=bool(n > 0 and real > p95)", "above_draw_p95=bool(n > 0 and real > dm)"),
    ('the evaluation without the committed fit', "    C.require(N.committed_equal(rec / 'fit' / 'fit.json', out / 'fit.json'), 'the fit manifest is not committed in HEAD: no label is read before the model hashes')\n", ""),
]
