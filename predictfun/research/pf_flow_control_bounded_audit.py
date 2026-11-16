"""Independent bounded-diagnostic artifact reconciliation and falsifier."""
import json
from pathlib import Path
import subprocess
import pf_flow_control_bounded as B
import pf_flow_control_run as RUN
F,C,np,pd=B.F,B.C,B.np,B.pd


def check(rows,candidates,report):
    keys=['m','side','rung','generation','decision_ms']
    C.require(not rows.duplicated(keys).any() and not candidates.duplicated(keys).any(),'duplicate diagnostic identity')
    selected=candidates[candidates.selected]
    C.require(set(rows[keys].itertuples(index=False,name=None))==set(selected[keys].itertuples(index=False,name=None)), 'selected/evaluated action identities differ')
    C.require(selected.status.isin(['eligible','eligible_estimated_timing']).all(),'invalid selected action')
    C.require(np.isfinite(rows.select_dtypes('number').to_numpy()).all(),'nonfinite diagnostic')
    C.require((rows.known_inventory.abs()<=F.CAP+1e-7).all(),'inventory support')
    for name in ('actual_keep_quantity','predicted_keep_quantity'):
        C.require(rows[name].between(0.,30.+1e-7).all(),'focal quantity bound')
    for prefix in ('predicted','realized'):
        C.require(np.allclose(rows[prefix+'_advantage_cents'],rows[prefix+'_wealth_advantage_cents']+rows[prefix+'_risk_advantage_cents'],rtol=1e-10,atol=1e-8),'wealth/risk decomposition')
    C.require(len(rows)==report['evaluated_actions'] and len(candidates)==report['generated_actions'],'population counts')
    C.require(candidates.status.value_counts().to_dict()==report['candidate_statuses'],'candidate statuses')
    for name,col in [('joint','predicted_advantage_cents'),('empirical','empirical_advantage_cents')]:
        pred=rows[col];actual=rows.realized_advantage_cents;positive=pred>0;r=report['models'][name]
        expected=dict(spearman=pred.corr(actual,method='spearman'),predicted_mean_cents=pred.mean(),actual_mean_cents=actual.mean(),
            positive_predictions=int(positive.sum()),positive_predicted_mean_cents=pred[positive].mean(),positive_realized_mean_cents=actual[positive].mean())
        for field,v in expected.items():C.require(np.isclose(v,r[field],rtol=1e-10,atol=1e-8),'diagnostic metric '+name+'/'+field)
    C.require(C.clean(RUN.matched_null(rows))==report['matched_null'],'matched action null recomputation')


def run():
    B.verify();r=json.loads((B.OUT/'receipt.json').read_text())
    C.require(r['source_manifest']==C.sha(B.FREEZE) and r['controller_freeze_sha256']==C.sha(F.FREEZE),'bounded provenance')
    for name,h in r['outputs'].items():C.require(C.sha(B.OUT/name)==h,'bounded artifact changed '+name)
    rows=pd.read_parquet(B.OUT/'probes.parquet');c=pd.read_parquet(B.OUT/'candidates.parquet');report=json.loads((B.OUT/'report.json').read_text())
    check(rows,c,report)
    bad=rows.copy();bad.loc[bad.index[0],'realized_advantage_cents']+=1.
    caught=B.CHECK.refused(lambda:check(bad,c,report),'corrupted realized action value')
    result=dict(asof=C.stamp(),n=len(rows),all_action_identities_and_statuses_reconciled=True,
        all_reported_metrics_recomputed=True,matched_200_draw_null_recomputed=True,corrupt_value_refused=caught,
        source_sha256=C.sha(Path(__file__)),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        receipt_sha256=C.sha(B.OUT/'receipt.json'),report_sha256=C.sha(B.OUT/'report.json'))
    C.dump(B.OUT/'audit_v1.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':run()
