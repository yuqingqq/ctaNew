import copy
import unittest
import numpy as np
import pandas as pd
import pf_wait_post_signal_data as S


def frame(n=20):
    f=pd.DataFrame({k:np.ones(n) for k in S.M.FEATURES})
    f['side_sign']=np.where(np.arange(n)%2==0,1.,-1.);f['side']=(f.side_sign<0).astype(int)
    f['sigma']=.0001;f['spread']=2.;f['price']=50.;f['time_remaining']=60000.;return f


class SignalDataTests(unittest.TestCase):
    def test_compact_units_and_sides(self):
        f=frame();x=S.compact(f)
        self.assertAlmostEqual(x.bn_z1000.iloc[0],1.)
        self.assertAlmostEqual(x.bn_z1000.iloc[1],-1.)
        self.assertAlmostEqual(x.micro_spread.iloc[0],.5)
        self.assertAlmostEqual(x.micro_spread.iloc[1],-.5)
        self.assertAlmostEqual(x.remaining_fraction.iloc[0],1.)

    def test_compact_ignores_outcomes_and_future_rows(self):
        f=frame();a=S.compact(f);bad=f.copy();bad['qpost']=1e20;bad['delta']=-1e20
        pd.testing.assert_frame_equal(a,S.compact(bad))
        future=f.copy()*100;both=pd.concat([f,future],ignore_index=True)
        pd.testing.assert_frame_equal(a,S.compact(both).iloc[:len(f)])

    def test_processor_freezes_training_stats_and_preserves_extreme_flags(self):
        f=pd.DataFrame({'x':np.r_[np.arange(200,dtype=float),np.nan]})
        p=S.Processor().fit(f);state=copy.deepcopy(p.metadata())
        x=p.transform(pd.DataFrame({'x':[1e20,np.nan,-1e20]}))
        self.assertEqual(p.metadata(),state);self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[0,p.output.index('x_high')],1)
        self.assertEqual(x[1,p.output.index('x_missing')],1)
        self.assertEqual(x[2,p.output.index('x_low')],1)
        with self.assertRaises(ValueError):p.transform(pd.DataFrame({'wrong':[1.]}))
        with self.assertRaises(ValueError):p.transform(pd.DataFrame({'x':[np.inf]}))

    def test_processor_removes_training_duplicates_and_refuses_no_signal(self):
        f=pd.DataFrame({'x':np.arange(100,dtype=float),'clone':np.arange(100,dtype=float),'constant':1.})
        p=S.Processor().fit(f);self.assertIn('clone',p.removed);self.assertIn('constant',p.removed)
        self.assertNotIn('clone',p.output)
        with self.assertRaises(ValueError):S.Processor().fit(pd.DataFrame({'x':[1.,1.]}))

    def test_chronology_positive_and_future_refusal(self):
        f=pd.DataFrame({'day':['20261025','20261026']});S.chronology(f,'20261027')
        with self.assertRaises(ValueError):S.chronology(f,'20261026')

    def test_marginal_shift_fires_and_identical_control(self):
        f=frame(200);z=f.copy();z['price']=60.
        a={x['feature']:x for x in S.feature_shift(f,f)}
        b={x['feature']:x for x in S.feature_shift(f,z)}
        self.assertEqual(a['price']['ks_distance'],0)
        self.assertEqual(b['price']['ks_distance'],1)
        self.assertEqual(b['price']['outside_train_range'],200)

    def test_distance_whole_window_calibration_and_missing_support_refusal(self):
        rng=np.random.default_rng(1);f=pd.DataFrame(dict(day=['20261025']*1000,market=np.arange(1000)%40,x=rng.uniform(-1,1,1000),z=rng.uniform(-1,1,1000)))
        near=f[['x','z']].iloc[:100].copy();far=near*0+1000
        a,am=S.distances(f,near,['x','z']);b,bm=S.distances(f,far,['x','z'])
        self.assertGreater(int(bm.sum()),int(am.sum()))
        self.assertEqual(a['training_calibration_q95'],b['training_calibration_q95'])
        with self.assertRaises(ValueError):S.distances(f.iloc[:5],far,['x','z'])

    def test_ridge_preserves_negative_base_value_and_learns_direction(self):
        x=np.linspace(-1,1,1000)[:,None];y=-2+3*x[:,0]
        pred,info=S.ridge(x,y,x)
        self.assertLess(np.mean((pred-y)**2),np.mean((y-y.mean())**2));self.assertLess(pred[0],pred[-1])
        self.assertAlmostEqual(info['y_mean'],-2.)

    def test_policy_requires_absolute_and_incremental_value(self):
        f=pd.DataFrame({'qpost':[-3.,2.,4.],'delta':[2.,-1.,3.]})
        m=S.metrics(f,np.array([-1.,2.,4.]),np.array([2.,-1.,3.]),0.,0.)
        self.assertEqual(m['selection']['n'],1);self.assertEqual(m['selection']['actual_post'],4.)


if __name__=='__main__':unittest.main()
