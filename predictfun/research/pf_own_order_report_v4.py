"""Supersede failed JSON serialization without changing any numerical comparison."""
import datetime as dt
import hashlib
import json
import subprocess
import numpy as np
from . import pf_order_policy_spec as S
from . import pf_own_order_runner_v3 as C


def native(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    return x


def declare():
    C.check()
    if json.loads(json.dumps(native(dict(n=np.int64(2),ok=np.bool_(True))),allow_nan=False))!={'n':2,'ok':True}:raise ValueError('native positive control')
    for bad in (dict(n=np.float64(np.nan)),dict(n=object())):
        try:json.dumps(native(bad),allow_nan=False)
        except (ValueError,TypeError):pass
        else:raise ValueError('known-bad JSON accepted')
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),version=4,source_sha=S.sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.HERE,text=True).strip(),
        causal_declaration_sha=S.sha(S.OUT/'causal_v3_declaration.json'),
        failed_output_sha=S.sha(S.OUT/'comparison.json'),
        supersedes='Incomplete original comparison.json: NumPy int64 in matched-null group count failed JSON serialization. File preserved; no numerical or policy changes.',
        controls='Native NumPy integer/bool conversion passes; NaN and unsupported-object refusals pass. Serialize completely before exclusive file creation.')
    S.write(S.OUT/'report_v4_declaration.json',d);S.write(S.REVIEW/'report_v4_declaration.json',d)


def report():
    causal=C.check();d=json.loads((S.OUT/'report_v4_declaration.json').read_text())
    if d['source_sha']!=S.sha(__file__):raise ValueError('report source changed')
    for day in S.TEST:
        c=json.loads((S.OUT/'runs/stress'/day/'causal_v3.json').read_text())
        if c['status']!='PASS' or c['declaration_sha']!=S.sha(S.OUT/'causal_v3_declaration.json'):raise ValueError('stress source identity')
    original=S.write
    def write(path,obj):
        if path.name=='comparison.json':path=path.with_name('comparison_native_v4.json')
        body=json.dumps(native(obj),indent=2,allow_nan=False)+'\n'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:f.write(body)
    S.write=write
    try:
        from .pf_order_policy_report import run
        run()
        p=S.HERE/'pf_own_order_mechanism.py';source=p.read_text()
        if source.count("S.OUT/'comparison.json'")!=2:raise ValueError('mechanism read-path anchor')
        source=source.replace("S.OUT/'comparison.json'","S.OUT/'comparison_native_v4.json'")
        namespace=dict(__name__='mechanism_v4',__package__=__package__,__file__=str(p))
        exec(compile(source,str(p),'exec'),namespace);namespace['run']()
        base=S.OUT/'comparison_native_v4.json';result=json.loads(base.read_text())
        result.update(version=4,status='CANONICAL',supersedes=dict(path=str(S.OUT/'comparison.json'),sha=d['failed_output_sha']),
            comparison_native_sha=S.sha(base),causal_correction=causal,serialization_correction=d,
            mechanism_sha=S.sha(S.OUT/'mechanism.json'),mechanism_generated_sha=hashlib.sha256(source.encode()).hexdigest())
        write(S.OUT/'comparison_v4.json',result);write(S.REVIEW/'comparison_v4.json',result)
        old=S.HERE.parent/'handoffs/20261111_own_orders_calm_results.md';new=old.with_name('20261111_own_orders_calm_results_v4.md')
        prefix='# Canonical own-order continuation and calm-entry results (v4)\n\n'+(
            'This receipt supersedes the incomplete base JSON output; NumPy scalars are serialized as native JSON types, without numerical changes. '
            'It also records the causal warm-up correction: imbalance derivatives are unknown until250ms of past history exists. '
            f"All {causal['feature_decisions_checked']:,} saved feature decisions in42 neutral/primary populations were later than that boundary; no original model, primary action, fill or result was affected. "
            'Stress runs use the corrected wrapper. Original sources/artifacts are preserved. Training v2 deduplicates neutral actions. All12 saved-model predictions were independently reconstructed with known-bad score refusals. '
            'Canonical machine receipt: reviews/own_orders_calm_20261111/comparison_v4.json.\n\n')
        with new.open('x') as f:f.write(prefix+'\n'.join(old.read_text().splitlines()[1:])+'\n')
        print('CANONICAL_REPORT',new,flush=True)
    finally:S.write=original


if __name__=='__main__':
    import sys
    declare() if sys.argv[1]=='declare' else report()
