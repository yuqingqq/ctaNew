"""Positive controls, planted failures, and independent artifact checks."""
import argparse
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_flow_price_common as C
import pf_flow_price_data as D
import pf_flow_price_model as M
import pf_flow_price_report as R
import pf_stop_ev_accounting_check as ACheck
np,pd=C.np,C.I.pd


class Checks(unittest.TestCase):
    def test_flow_side_clock_and_boundaries(self):
        w=dict(side=0,anc=np.array([86.,87.,1050.,1051.,250.]),ptl=[50,50,49,50,51],shl=[99.,3.,7.,99.,99.],
            tf=np.array([0.,1000.,2000.,3000.]),P5=np.array([[50],[48],[48],[48]]),opp=np.array([52,50,50,50]),ws=0.)
        y=D.public_target(w,0.,50.,51.)
        self.assertEqual(y['flow_quantity'],10.)
        self.assertEqual(y['move_cents'],-2.)
        self.assertNotEqual(D.public_target(w,0.,50.,51.,shift=0.)['flow_quantity'],y['flow_quantity'])
        a=dict(w,side=1,P5=np.array([[52],[54],[54],[54]]),opp=np.array([50,52,52,52]),ptl=[52,52,53,52,51])
        z=D.public_target(a,0.,52.,51.);self.assertEqual(z['flow_quantity'],10.);self.assertEqual(z['move_cents'],-2.)

    def test_classes_and_invalid(self):
        f,m,j=C.classes([0,1,30,31],[-3,-.5,.5,3])
        np.testing.assert_array_equal(f,[0,1,1,2]);np.testing.assert_array_equal(m,[0,2,3,4]);np.testing.assert_array_equal(j,5*f+m)
        with self.assertRaises(ValueError):C.classes([-1],[0])
        with self.assertRaises(ValueError):C.classes([1],[np.nan])

    def test_joint_not_product_of_marginals(self):
        p=np.array([[.5,0.,0.,.5]])
        value=np.array([[0.,0.,10.,-10.]])
        self.assertEqual(M.integrate(p,value)[0],-5.)
        self.assertEqual(M.integrate(np.full((1,4),.25),value)[0],0.)
        with self.assertRaises(ValueError):M.integrate(p*2,value)

    def test_mark_support_independent_of_fill(self):
        w=dict(side=0,tf=np.array([0.,500.,1000.,1500.,2000.,2500.]),P5=np.full((6,1),50),opp=np.full(6,52),ws=0.)
        self.assertEqual(D.mark_support(w,0.),'ok')
        self.assertEqual(D.mark_support(w,299000.),'expiry_censored')
        bad=dict(w,tf=np.array([0.,500.,2500.]),P5=np.full((3,1),50),opp=np.full(3,52))
        self.assertEqual(D.mark_support(bad,0.),'stale_future_marks')

    def test_empirical_unseen_group_and_temperature(self):
        x=pd.DataFrame(dict(side=[0,0,1],rung=[0,0,0],phase=[1,1,1],joint_class=[0,0,14]))
        e=M.empirical_fit(x);pred=M.empirical_predict(x,e)
        self.assertTrue(np.allclose(pred.sum(1),1))
        unseen=x.copy();unseen['phase']=5
        np.testing.assert_allclose(M.empirical_predict(unseen,e),np.tile(e[0],(3,1)))
        self.assertLess(M.logloss(np.array([0,1]),np.array([[.9,.1],[.1,.9]])),M.logloss(np.array([0,1]),np.full((2,2),.5)))
        for t in C.TEMPERATURES:self.assertTrue(np.allclose(M.temperature(pred,t).sum(1),1))

    def test_feature_allowlist(self):
        self.assertEqual(len(C.FEATURES),len(set(C.FEATURES)))
        self.assertFalse(set(C.FEATURES)&set(C.TARGETS+C.COND+['joint_class','flow_class','move_class','move_cents']))
        self.assertTrue(set(C.MECHANICAL)<=set(C.FEATURES))


def selftest():
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Checks),
                             unittest.defaultTestLoader.loadTestsFromTestCase(ACheck.Checks)])
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    C.require(result.wasSuccessful(),'selftests failed')
    return dict(tests=result.testsRun,passed=True)


def audit():
    C.verify();controls=selftest();reports=[]
    for day in C.DAYS:
        p=C.OUT/day;r=json.loads((p/'receipt.json').read_text());df=pd.read_parquet(p/'actions.parquet')
        C.require(C.sha(p/'actions.parquet')==r['output_sha256'],'dataset hash')
        C.require(len(df)==r['rows'] and df.status.value_counts().to_dict()==r['statuses'],'row/status reconciliation')
        C.require(not df.duplicated(['day','m','side','rung','generation']).any(),'duplicate generation')
        g=df[df.status.isin(['eligible','eligible_estimated_timing'])]
        C.require(len(g)==r['eligible'],'eligible count')
        for name,v in zip(['flow_class','move_class','joint_class'],C.classes(g.flow_quantity,g.move_cents)):
            C.require(np.array_equal(g[name],v),'class target '+name)
        C.require(np.allclose(g.value_cents,g.at_price_value_cents+g.through_price_value_cents),'value attribution')
        C.require((g.own_quantity>=0).all() and (g.own_quantity<=g.fp_remaining+1e-6).all(),'own quantity bounds')
        C.require((g.own_quantity<=g.flow_quantity+1e-6).all(),'own/public quantity bounds')
        C.require(np.all(g.loc[g.own_quantity==0,'value_cents']==0),'no-fill value')
        reports.append(dict(day=day,rows=len(df),eligible=len(g),no_fill=int((g.own_quantity==0).sum()),
                            causal_checks=sum(x.get('causal_checks',0) for x in r['markets'])))
        for path,h in r['input_hashes'].items():C.require(C.sha(path)==h,'raw input changed '+path)
    allpred=[]
    for day in C.TESTS:
        import lightgbm as lgb
        p=C.OUT/('fit_'+day);r=json.loads((p/'receipt.json').read_text());df=pd.read_parquet(p/'test.parquet');raw=M.read(day)
        for name,h in r['outputs'].items():C.require(C.sha(p/name)==h,'fit output hash '+name)
        for artifact in r['artifacts'].values():C.require(C.sha(artifact['path'])==artifact['sha256'],'model artifact hash')
        keys=['day','m','side','rung','generation','decision_ms'];C.require(df[keys].equals(raw[keys].reset_index(drop=True)),'prediction identities')
        ix=np.linspace(0,len(raw)-1,min(256,len(raw)),dtype=int);x=raw.iloc[ix][C.FEATURES].to_numpy(np.float32)
        b=lgb.Booster(model_file=str(p/'joint.txt'));prob=M.temperature(b.predict(x,num_threads=1),r['temperature']['joint']['value'])
        C.require(np.allclose(prob,df.iloc[ix][[f'joint_p{k}' for k in range(15)]].to_numpy(),rtol=0,atol=1e-10),'saved probability reload')
        b=lgb.Booster(model_file=str(p/'conditional_value_cents.txt'));v=np.empty((len(ix),15))
        for k in range(15):
            inp=np.column_stack([x,np.full(len(ix),k//5),np.full(len(ix),k%5)]).astype(np.float32)
            v[:,k]=b.predict(inp,num_threads=1) if k//5 else 0.
        C.require(np.allclose(np.sum(prob*v,axis=1),df.iloc[ix].joint_value_cents,rtol=0,atol=1e-8),'independent mixture reload')
        allpred.append(df)
    result=json.loads((C.REVIEW/'results_v1/report.json').read_text());computed=R.metrics(pd.concat(allpred,ignore_index=True))
    C.require(C.clean(computed)==result['pooled'],'report not equal prediction artifacts')
    wrong=copy.deepcopy(result['pooled']);wrong['n']+=1
    C.require(wrong!=C.clean(computed),'report falsifier did not fire')
    C.dump(C.REVIEW/'results_v1/audit.json',dict(asof=C.stamp(),controls=controls,data=reports,predictions=sum(len(x) for x in allpred),
        model_probability_and_value_reload='passed',all_raw_hashes='passed',report_recompute='passed',corrupt_report_refused=True,
        report_sha256=C.sha(C.REVIEW/'results_v1/report.json')))
    print('Independent artifact audit passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['selftest','freeze','audit']);a=p.parse_args()
    if a.mode=='selftest':print(json.dumps(selftest()))
    elif a.mode=='freeze':selftest();C.freeze()
    else:audit()
