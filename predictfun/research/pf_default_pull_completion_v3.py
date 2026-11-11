"""Apply the complete artifact audit to v3 and verify its full-day runtime parity."""
import datetime as dt
import json
import subprocess
from . import pf_default_pull_completion as C
from . import pf_default_pull_fast as F


def run():
    C.configure=lambda:F.install()[0]
    C.run()
    from . import pf_default_pull_spec as S
    d=S.check();p=S.OUT/'runtime_parity.json';r=json.loads(p.read_text())
    if r['status']!='PASS' or r['windows']!=288 or r['arms']!=2 or r['day']!=S.DAYS[0]:raise ValueError('incomplete runtime parity')
    C.need_hash(S.OUT/'declaration.json',r['declaration_sha'])
    old=F.GOLD/'runs/build'/S.DAYS[0];new=S.folder(S.DAYS[0],'build')
    for rel,h in d['runtime_golden_sha'].items():C.need_hash(old/rel,h)
    for row in r['matches']:
        C.need_hash(old/row['path'],row['old_sha']);C.need_hash(new/row['path'],row['new_sha'])
    for root,key in [(old,'old_independent_sha'),(new,'new_independent_sha')]:
        C.need_hash(root/'independent_windows.parquet',r[key])
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',source_sha=S.sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.HERE,text=True).strip(),
        completion_audit_sha=S.sha(S.OUT/'completion_audit.json'),runtime_parity_sha=S.sha(p),
        declaration_sha=S.sha(S.OUT/'declaration.json'),comparison_sha=S.sha(S.OUT/'comparison_final.json'),
        scope='All original completion requirements plus frozen old/new full-day runtime parity identities. The shared auditor carries hash, score and interval positive/falsifier controls.')
    S.write(S.OUT/'completion_runtime_v3.json',result);S.write(S.REVIEW/'completion_runtime_v3.json',result)
    print('V3_COMPLETION_RUNTIME_AUDIT_PASS',flush=True)


if __name__=='__main__':run()
