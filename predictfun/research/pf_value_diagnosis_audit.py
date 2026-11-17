"""Identity-bound saved-model, calibration and first-review diagnostic audit."""
import json
import pickle
from unittest.mock import patch
import pf_value_diagnosis as D
import pf_value_diagnosis_calibration as CAL
import pf_value_diagnosis_controller as CTRL
F,C,np,pd,J=D.F,D.C,D.np,D.pd,D.J


def run():
    D.verify();checks=dict(calibration=CAL.selftest(),controller=CTRL.selftest())
    out=D.OUT/'calibration';r=D.checked_receipt(out)
    train=pd.concat([D.FIT.read(day)[0] for day in F.TRAIN],ignore_index=True)
    med,scale=CAL.geometry(train);calibration_counts={};allfiles=[]
    for history in D.HISTORY:
        dest=out/history;receipt=D.checked_receipt(dest)
        C.require(C.sha(dest/'receipt.json')==r['children'][history],'history receipt identity')
        expected=CAL.selected(train,history);ids=pd.read_parquet(dest/'training_ids.parquet')
        C.require(set(ids.uid)==set(expected.uid) and len(ids)==len(expected),'history sample identity')
        training_saved=json.loads((dest/'training_calibration.json').read_text())
        training_heads=CAL.attach(expected,CAL.predict(expected,receipt['artifacts']))
        C.require(C.clean(CAL.calibration(training_heads,True))=={k:training_saved[k] for k in ('role','groups')},'training calibration reconstruction')
        for day in (F.CAL,F.TEST):
            heads=pd.read_parquet(dest/(day+'_heads.parquet'));a,ar=D.FIT.read(day)
            C.require(set(heads.uid)==set(a.uid),'evaluation support')
            a=a.set_index('uid').loc[heads.uid].reset_index();D.check_equal(heads[J.FEATURES],a[J.FEATURES],'causal feature identity')
            for col in ('y_q','y_m','y_w','y_value','y_severe'):D.check_equal(heads[col],a[col],'target identity')
            p=CAL.predict(a,receipt['artifacts'])
            for name,value in p.items():D.check_equal(value,heads['raw_'+name],'saved head reproduction')
            base=pd.read_parquet(J.D.OUT/day/'values.parquet');base=base[(base.case=='SOFT_JOINT')&(base.channel=='ordinary')]
            new=CAL.attach(a,p,base)
            for col in ('q_full','m_full','q_mechanical','m_mechanical','w_direct','value_soft','value_full','value_mechanical','value_direct'):
                D.check_equal(new[col],heads[col],'value reconstruction '+col)
            cov,reference=CAL.coverage(expected,a,med,scale);cov=cov.set_index('uid').loc[heads.uid]
            for col in ('nearest_distance','reference_q95','outside_reference95','cell_n','cell_severe_n'):D.check_equal(cov[col],heads[col],'coverage '+col)
            saved=json.loads((dest/(day+'_report.json')).read_text());values,econ=CAL.economic_report(heads)
            C.require(C.clean(econ)==saved['economic'],'economic calibration/null report')
            C.require(C.clean(CAL.calibration(heads))==saved['calibration'],'head calibration report')
            allfiles.append(dest/(day+'_report.json'))
        calibration_counts[history]=dict(training=len(ids),evaluation=861,all_saved_heads_values_metrics_coverage_reproduced=True)
        allfiles.extend([dest/'receipt.json',dest/'training_calibration.json'])
    ctrl=D.OUT/'controller';receipt=D.checked_receipt(ctrl)
    rows=pd.read_parquet(ctrl/'pairs.parquet');entries=pd.read_parquet(ctrl/'entries.parquet')
    attrib=pd.read_parquet(ctrl/'attribution.parquet');sens=pd.read_parquet(ctrl/'sensitivity.parquet')
    pairs=pickle.loads((ctrl/'states.pkl').read_bytes());paths=CTRL.PATH.load()
    C.require(set(rows.uid)==set(entries.loc[entries.status=='paired_first_review','uid']),'entry statuses')
    expected=set()
    for _,g in rows.groupby(['m','side','cancel']):expected.update(sorted(g.uid,key=D.CH.digest)[:D.CH.PER_GROUP])
    C.require(expected==set(attrib.uid)==set(sens.uid)=={p['uid'] for p in pairs},'first-review hash sample')
    a=attrib.set_index('uid');s=sens.set_index('uid')
    with patch.object(D.CH,'context',CTRL.context):
        for pair in pairs:
            got=D.CH.decompose(pair,paths);other=CTRL.extra(pair,paths)
            for key,value in got.items():
                if isinstance(value,(bool,int,float,np.number)):D.check_equal(value,a.loc[pair['uid'],key],'attribution reconstruction')
            for key,value in other.items():
                if isinstance(value,(bool,int,float,np.number)):D.check_equal(value,s.loc[pair['uid'],key],'sensitivity reconstruction')
                else:C.require(value==s.loc[pair['uid'],key],'sensitivity status identity')
    saved=json.loads((ctrl/'report.json').read_text());report=CTRL.summarize(rows,attrib,entries,sens)
    report['asof']=saved['asof'];report.update(controls=saved['controls'],limits=D.verify()['limits'])
    C.require(C.clean(report)==saved,'controller report/null reconstruction')
    checks['artifact_corruption_refused']=D.FIT.CHECK.refused(lambda:D.check_equal(attrib.current_keep,attrib.current_keep+1.,'bad attribution'),'bad attribution')
    audit=dict(asof=C.stamp(),controls=checks,calibration=calibration_counts,
        controller=dict(entries=len(entries),paired=len(rows),sample=len(pairs),all_attributions_and_sensitivities_reproduced=True),
        source_manifest=C.sha(D.FREEZE),limits=D.verify()['limits'])
    allfiles += [out/'report.json',out/'receipt.json',ctrl/'report.json',ctrl/'receipt.json']
    for p in allfiles:
        dest=D.REVIEW/p.relative_to(D.OUT);C.require(not dest.exists(),'published artifact exists');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    C.dump(D.REVIEW/'audit_v1.json',audit)
    C.dump(D.REVIEW/'index.json',dict(asof=C.stamp(),source_manifest=C.sha(D.FREEZE),
        outputs={str(p.relative_to(D.OUT)):C.sha(p) for p in allfiles},audit_sha256=C.sha(D.REVIEW/'audit_v1.json'),limits=D.verify()['limits']))
    print(json.dumps(C.clean(audit)),flush=True)


if __name__=='__main__':run()
