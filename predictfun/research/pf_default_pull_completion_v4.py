"""Complete v4 artifact audit and verify the known-empty admission correction."""
import datetime as dt
import json
import subprocess
from . import pf_default_pull_completion as C
from . import pf_default_pull_known_empty as V4


def run():
    C.configure=lambda:V4.install()[0]
    C.run()
    from . import pf_default_pull_spec as S
    d=S.check();p=S.OUT/'runtime_parity.json';r=json.loads(p.read_text());c=json.loads((S.OUT/'known_empty_controls.json').read_text())
    if r['status']!='PASS' or r['policy_output_equality_claim'] is not False:raise ValueError('wrong inherited parity scope')
    C.need_hash(r['inherited_path'],r['inherited_sha'])
    if r['inherited_sha']!=d['inherited_runtime_parity']['sha']:raise ValueError('wrong inherited cache parity')
    if c['status']!='PASS' or c['new_level_quote']!=[50.,5.] or c['planted_old_behavior']!=[None,0.]:raise ValueError('known-empty regression/falsifier missing')
    if not all(c[k] for k in ('missing_history_refused','outside_depth_refused','zero_to_zero_neutral','positive_to_zero_thinning')):raise ValueError('known-empty coverage incomplete')
    C.need_hash(S.OUT/'declaration.json',c['declaration_sha']);C.need_hash(S.HERE/'pf_default_pull_known_empty.py',c['source_sha'])
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',source_sha=S.sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.HERE,text=True).strip(),
        completion_audit_sha=S.sha(S.OUT/'completion_audit.json'),runtime_parity_sha=S.sha(p),known_empty_controls_sha=S.sha(S.OUT/'known_empty_controls.json'),
        declaration_sha=S.sha(S.OUT/'declaration.json'),comparison_sha=S.sha(S.OUT/'comparison_final.json'),
        scope='All full artifact/model/action checks plus v4 known-empty correction controls. Prior cache equivalence is inherited implementation evidence, not a v3/v4 policy-equality claim.',
        status_semantics='Raw generation status describes probe/label state. Inherited accepted_no_cancel on reference rows means no training probe, not absence of an actual cancellation. Use first_cancel and engine request records for lifecycle; the audit computes resting intervals from those actual lifecycle fields.')
    S.write(S.OUT/'completion_runtime_v4.json',result);S.write(S.REVIEW/'completion_runtime_v4.json',result)
    print('V4_COMPLETION_AUDIT_PASS',flush=True)


if __name__=='__main__':run()
