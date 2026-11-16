"""Reconcile the bank-only sensitivity at its named state and artifacts."""
import json
from pathlib import Path
import pickle
import subprocess
import pf_flow_churn_diagnostic as D
C,np,pd,F=D.C,D.np,D.pd,D.F


def run():
    F.verify();report=json.loads((D.OUT/'bank_check_v1.json').read_text())
    C.require(C.sha(D.OUT/'bank_check_v1.parquet')==report['data_sha256'],'bank sensitivity artifact changed')
    C.require(C.sha(F.HERE/'pf_flow_churn_bank_check.py')==report['source_sha256'],'bank evaluator changed')
    df=pd.read_parquet(D.OUT/'bank_check_v1.parquet').set_index('uid')
    pairs=pickle.loads((D.OUT/'states.pkl').read_bytes());paths=D.PATH.load();bounds=0;count=0
    C.require(set(df.index)=={p['uid'] for p in pairs},'bank state identities differ')
    for p in pairs:
        row=df.loc[p['uid']];e=p['entry'];r=p['review'];sp=r['spec'];s=r['state'];idx=r['index']
        old=D.PATH.select(paths,e['spec']['phase'],e['spec']['bid'],e['spec']['spread'])
        valid=all(0<=z['min_price']+sp['bid']-z['bid'] and z['max_price']+sp['bid']-z['bid']<=100 for z in old)
        C.require(valid==(row.status=='eligible'),'counterfactual bounds status mismatch')
        if not valid:bounds+=1;continue
        packed=D.K.pack(old,sp['bid'],1000.);D.K.current_book(packed,sp['prices'],sp['sizes'])
        weight,_=D.PATH.weights(old,sp['prob']);keep=r['prefix'].copy();keep[idx]=1.;cancel=keep.copy();cancel[idx]=2.
        out,q=D.K.evaluate(packed,s['orders'],np.stack([keep,cancel]),s['position'],sp['bid']+sp['spread']/2,1000.)
        value=weight@(out[:,0,0]-out[:,1,0]-F.RISK*(out[:,0,2]**2-out[:,1,2]**2))
        C.require(np.isclose(value,row.old_bank_keep,atol=1e-7),'old-bank value not reproducible')
        C.require(np.isclose(row.current_keep-row.old_bank_keep,row.bank_only_change_cents,atol=1e-7),'bank attribution arithmetic')
        count+=1
    for name,cancel in [('cancel',True),('keep',False)]:
        g=df[df.cancel==cancel];z=g[(g.status=='eligible')&g.bank_changed];saved=report['groups'][name]
        C.require(saved['statuses']==g.status.value_counts().to_dict(),'reported statuses differ')
        C.require(saved['n']==len(g) and saved['changed_bank_eligible_n']==len(z) and saved['old_bank_keep_nonnegative']==int((z.old_bank_keep>=-1e-9).sum()),'reported count differs')
        for key,col in [('mean_old_bank_keep_cents','old_bank_keep'),('mean_current_keep_cents','current_keep'),('mean_bank_only_change_cents','bank_only_change_cents'),('mean_old_bank_unsupported_mass','old_bank_unsupported_mass'),('mean_current_bank_unsupported_mass','current_bank_unsupported_mass')]:
            C.require(np.isclose(saved[key],z[col].mean(),atol=1e-10),'reported mean differs '+key)
    bad=D.CHECK.refused(lambda:C.require(np.isclose(value,row.old_bank_keep+1.,atol=1e-7),'corrupted bank value'),'corrupted bank value')
    result=dict(asof=C.stamp(),n=len(df),eligible_values_recomputed=count,out_of_bounds_statuses_verified=bounds,
        all_report_counts_and_means_recomputed=True,corrupted_value_refused=bad,
        report_sha256=C.sha(D.OUT/'bank_check_v1.json'),source_sha256=C.sha(Path(__file__)),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip())
    C.require(not (D.OUT/'bank_audit_v1.json').exists(),'bank audit exists');C.dump(D.OUT/'bank_audit_v1.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':run()
