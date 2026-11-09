import unittest
from unittest.mock import patch
import pandas as pd
from predictfun.research import pf_exit_engine as e
from predictfun.research.pf_exit_report import fill_profile


class ExitControls(unittest.TestCase):
    def order(self,**kw):
        return dict(side=1,pt=51,size=25,kq=15,**kw)

    def test_keep_safe_above_prediction_size(self):
        o=self.order()
        original=e.old.plan(10,[(None,0),(51,5)],[None,o],[o],'late_reserved',120)
        self.assertEqual(original,((None,0),(None,0)))
        self.assertEqual(e.plan(10,[(None,0),(51,5)],[None,o],[o],'exit_keep',120),((None,0),(51,25)))
        self.assertEqual(e.desired_size(10,1,51,5,o,[o],'exit_keep',120),25)

    def test_retain_before_minimum_and_no_upsize(self):
        o=dict(side=1,pt=90,size=25,kq=20)
        self.assertFalse(e.old.prior.entry.legal(90,1,5))
        size=e.desired_size(5,1,90,5,o,[o],'exit_keep',120)
        self.assertEqual(size,25)
        self.assertEqual(e.plan(5,[(None,0),(90,size)],[None,o],[o],'exit_keep',120),((None,0),(90,25)))
        self.assertEqual(e.desired_size(30,1,90,5,o,[o],'exit_close25',120),25)

    def test_pending_reservations_no_physical_credit(self):
        o=self.order(rem=0,cancelled=True)
        p=dict(side=1,pt=52,size=1,kq=0)
        self.assertEqual(e.plan(10,[(None,0),(51,5)],[None,o],[o,p],'exit_keep',120),((None,0),(None,0)))
        self.assertEqual(e.admission(10,1,5,[o,o],'exit_keep',120),0)
        with self.assertRaises(ValueError):e.assert_admission(10,1,1,[o],'exit_keep',120)

    def test_full_close_size_and_symmetry(self):
        self.assertEqual(e.desired_size(20,1,51,5,None,[],'exit_close25',120),20)
        self.assertEqual(e.desired_size(-40,0,49,5,None,[],'exit_close25',120),25)
        self.assertEqual(e.plan(20,[(None,0),(51,5)],[None]*2,[],'exit_close25',120),((None,0),(51,20)))
        self.assertEqual(e.plan(-20,[(49,5),(None,0)],[None]*2,[],'exit_close25',120),((49,20),(None,0)))
        with self.assertRaises(ValueError):e.assert_admission(10,1,11,[],'exit_close25',120)

    def test_guards_minimum_and_ordering(self):
        o=self.order()
        self.assertEqual(e.plan(10,[(None,0),(None,0)],[None,o],[o],'exit_close25',120),((None,0),(None,0)))
        self.assertEqual(e.plan(1,[(None,0),(51,5)],[None]*2,[],'exit_close25',120),((None,0),(None,0)))
        w=[(None,0),(51,5),(None,0),(52,5)]
        self.assertEqual(e.plan(30,w,[None]*4,[],'exit_close25',120),((None,0),(51,25),(None,0),(52,5)))

    def test_early_and_boundary_identity(self):
        w=[(49,25),(51,25)]
        self.assertEqual(e.plan(0,w,[None]*2,[],'exit_close25',59.99),tuple(w))
        self.assertEqual(e.desired_size(20,1,51,5,None,[],'exit_close25',59.99),5)
        self.assertEqual(e.admission(20,1,25,[],'exit_keep',240),0)
        self.assertEqual(e.admission(-1,1,25,[],'exit_keep',120),0)
        self.assertEqual(e.price(20,1,51,49,51,120,'exit_close25'),51)

    def test_reference_dispatch_and_source_identity(self):
        w=[(None,0),(51,5)]
        self.assertEqual(e.plan(20,w,[None]*2,[],'late_reserved',120),e.old.plan(20,w,[None]*2,[],'late_reserved',120))
        compile(e.source(),'exit','exec')
        with patch('predictfun.research.pf_exit_engine.sha',return_value='wrong'):
            with self.assertRaises(ValueError):e.source()

    def test_fill_age_attribution_control(self):
        f=pd.DataFrame(dict(age_ms=[20.,200.,300.],zombie=[False,False,True],q=[2.,3.,4.],mo1=[1.,-2.,3.]))
        a=fill_profile(f)
        self.assertEqual(a['young_or_pending_shares'],6)
        self.assertEqual(a['young_or_pending_plus1'],4)
        self.assertEqual(a['older_resting_plus1'],-2)
        f.loc[0,'age_ms']=float('nan')
        with self.assertRaises(ValueError):fill_profile(f)


if __name__=='__main__':unittest.main()
