"""Recompute oracle report metrics and verify identical action/label populations."""
import json
from pathlib import Path
import subprocess
import pf_value_bridge as V
import pf_value_bridge_run as R
import pf_flow_control_checks as CHECK
C,np,pd,F=V.C,V.np,V.pd,V.F


def check(rows,original,report):
    C.require(not rows.duplicated(['uid','case']).any(),'duplicate diagnostic action')
    target=original[original.case=='EXACT_JOINT'].set_index('uid')
    for case,g in rows.groupby('case'):
        C.require(set(g.uid)==set(target.index),'diagnostic identity coverage')
        x=g.set_index('uid').loc[target.index]
        C.require(np.allclose(x.actual,target.actual,atol=1e-9),'diagnostic changed actual target')
    good=rows[rows.evaluation_status=='eligible'];counts=good.groupby('uid').case.nunique();common=set(counts[counts==3].index)
    C.require(report['common_supported_actions']==len(common),'common support count')
    C.require(C.clean(rows.groupby(['case','evaluation_status']).size().to_dict())==report['statuses'],'diagnostic statuses')
    combined=pd.concat([original[original.case.isin(['EXACT_JOINT','SOFT_JOINT'])],rows],ignore_index=True)
    for (case,kind),g in combined[combined.evaluation_status=='eligible'].groupby(['case','kind']):
        C.require(C.clean(R.action_metrics(g))==report['available_metrics'][case+':'+kind],'available diagnostic metrics')
        z=g[g.uid.isin(common)]
        if len(z):C.require(C.clean(R.action_metrics(z))==report['common_support_metrics'][case+':'+kind],'matched diagnostic metrics')


def run():
    V.verify();out=V.OUT/'oracle_v1';r=json.loads((out/'receipt.json').read_text())
    C.require(C.sha(F.HERE/'pf_value_bridge_oracle.py')==r['source_sha256'],'oracle source changed')
    C.require(C.sha(V.FREEZE)==r['candidate_freeze_sha256'],'candidate was changed')
    C.require(C.sha(V.OUT/'actions_20261027/receipt.json')==r['input_receipt_sha256'],'source actions changed')
    for p,h in r['outputs'].items():C.require(C.sha(out/p)==h,'oracle artifact changed')
    df=pd.read_parquet(out/'values.parquet');original=pd.read_parquet(V.OUT/'actions_20261027/values.parquet');report=json.loads((out/'report.json').read_text())
    check(df,original,report);bad=df.copy();bad.loc[bad.index[0],'actual']+=1.
    caught=CHECK.refused(lambda:check(bad,original,report),'corrupt oracle actual value')
    result=dict(asof=C.stamp(),n=int(df.uid.nunique()),all_identities_statuses_and_matched_metrics_recomputed=True,corrupted_actual_refused=caught,
        source_sha256=C.sha(Path(__file__)),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        receipt_sha256=C.sha(out/'receipt.json'),report_sha256=C.sha(out/'report.json'))
    C.require(not (out/'audit_v1.json').exists(),'oracle audit exists');C.dump(out/'audit_v1.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':run()
