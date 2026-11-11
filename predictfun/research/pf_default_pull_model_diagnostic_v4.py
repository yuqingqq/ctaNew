"""Publish the already-declared neutral-action diagnostic before portfolio completion."""
import datetime as dt
import json
import subprocess
from .pf_default_pull_known_empty import install


def run():
    S,_,_=install();declaration=S.check()
    plan=S.REVIEW/'model_diagnostic_plan.json'
    frozen=json.loads(plan.read_text())
    if frozen['source_sha']!=S.sha(__file__):raise ValueError('diagnostic wrapper changed')
    if frozen['declaration_sha']!=S.sha(S.OUT/'declaration.json'):raise ValueError('wrong experiment')
    if frozen['report_source_sha']!=S.sha(S.HERE/'pf_default_pull_report.py'):raise ValueError('declared diagnostic changed')
    # Reuse the original positive matched-null and known-bad nonfinite-value refusal.
    from .tests.test_default_pull import Controls
    Controls('test_null_and_day_units').test_null_and_day_units()
    from .pf_default_pull_report import diagnostics
    fit,diag=diagnostics()
    n=sum(x['test_n'] for x in fit['folds'])
    if any(r['n']!=n or r['draws']!=declaration['null']['draws'] for r in diag.values()):
        raise ValueError('diagnostic cohort/draw mismatch')
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='NEUTRAL_DIAGNOSTIC_ONLY',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.HERE,text=True).strip(),
        source_sha=S.sha(__file__),plan_sha=S.sha(plan),declaration_sha=S.sha(S.OUT/'declaration.json'),
        fit_sha=S.sha(S.OUT/'fit.json'),predictions_sha=fit['predictions_sha'],days=S.TEST,
        eligible_actions=n,statuses=[x for x in fit['statuses'] if x['day'] in S.TEST],diagnostics=diag,
        predicates={k:dict(local_value_above_matched_null=v['observed_c']>v['null_mean_c']) for k,v in diag.items()},
        interpretation='Same predeclared 500 matched-count shuffles as the final report, without changing models, thresholds, nulls or selection. These overlapping local keep/reset counterfactuals approximate portfolio responses; sums are not realizable portfolio P&L. Consumed dates and inherited LODO guard caches preclude untouched/full historical validation. Actual sequential trading value, inventory and execution-stress comparisons remain pending.',
        controls='Original matched-null positive and nonfinite-value refusal pass; saved model identities and predictions reconstructed by the frozen report diagnostic.')
    S.write(S.OUT/'model_diagnostic.json',result);S.write(S.REVIEW/'model_diagnostic.json',result)
    print('NEUTRAL_MODEL_DIAGNOSTIC',n,flush=True)


if __name__=='__main__':run()
