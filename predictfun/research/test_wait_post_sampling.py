import unittest
import numpy as np
import pandas as pd
import pf_wait_post_sampling as P


class SamplingTests(unittest.TestCase):
    def test_probabilities_and_invalid_generation(self):
        f=pd.DataFrame({'generation':[0,1,2]});np.testing.assert_equal(P.weights(f),[2,32,32])
        with self.assertRaises(ValueError):P.weights(pd.DataFrame({'generation':[-1]}))
        with self.assertRaises(ValueError):P.weights(pd.DataFrame({'generation':[.5]}))

    def test_weighted_metrics_and_refusal(self):
        y=np.array([1.,-1.]);w=np.array([2.,32.]);m=P.metric(y,y,w,0.)
        self.assertEqual(m['r2_vs_prior_mean'],1.);self.assertAlmostEqual(m['actual_mean'],-30/34)
        self.assertLess(P.metric(y,-y,w,0.)['r2_vs_prior_mean'],0.)
        with self.assertRaises(ValueError):P.metric(y,y,[0.,32.],0.)

    def test_sampling_correction_recovers_known_population(self):
        # 160 first and160 later opportunities; expected hash sample80 and5.
        f=pd.DataFrame({'generation':[0]*80+[1]*5});w=P.weights(f)
        self.assertAlmostEqual(w[f.generation==0].sum()/w.sum(),.5)
        self.assertGreater((f.generation==0).mean(),.9)

    def test_weight_matched_null_oracle_anti_and_bad_weight(self):
        f=pd.DataFrame(dict(day=['synthetic']*800,side=[0]*800,rung=[0]*800,probe_t=np.zeros(800),generation=[0]*400+[1]*400,
            qpost=([-2.]*200+[2.]*200)*2,delta=([-2.]*200+[2.]*200)*2,post_action=([False]*200+[True]*200)*2))
        f['sampling_weight']=P.weights(f);self.assertTrue(P.matched(f,200)['passes'])
        anti=f.copy();anti['post_action']=~anti.post_action;self.assertFalse(P.matched(anti,200)['passes'])
        bad=f.copy();bad.loc[0,'sampling_weight']=3.
        with self.assertRaises(ValueError):P.matched(bad,200)
        with self.assertRaises(ValueError):P.matched(f,199)

    def test_good_timing_cannot_override_negative_absolute_value(self):
        f=pd.DataFrame(dict(day=['synthetic']*400,side=[0]*400,rung=[0]*400,probe_t=np.zeros(400),generation=np.zeros(400),
            qpost=[-1.]*400,delta=[-2.]*200+[2.]*200,post_action=[False]*200+[True]*200))
        f['sampling_weight']=P.weights(f);r=P.matched(f,200)
        self.assertGreater(r['observed_delta'],r['null_p95']);self.assertFalse(r['passes'])


if __name__=='__main__':unittest.main()
