import unittest
from predictfun.research.pf_quote_capture import QuotePlanner
from predictfun.research.pf_reentry_study import aligned_intervals,difference_time


class ReentryStudyControls(unittest.TestCase):
    def test_exact_lost_and_gained_coverage(self):
        w=[(50,5)]+[(None,0.)]*3
        a=QuotePlanner(incremental=True);b=QuotePlanner(incremental=True)
        a.plan(0,w);b.plan(50,w)
        aa=aligned_intervals(a,0,150,w);bb=aligned_intervals(b,0,150,w)
        self.assertEqual(difference_time(aa,bb),50)
        self.assertEqual(difference_time(bb,aa),0)
        a.plan(100,[(None,0.)]*4)
        aa=aligned_intervals(a,100,150,w);bb=aligned_intervals(b,100,150,w)
        self.assertEqual(difference_time(bb,aa),28)
        self.assertEqual(aligned_intervals(a,100,150,[(51,5)]+w[1:]),[[],[],[],[]])
    def test_split_tickets_do_not_duplicate_coverage(self):
        w=[(50,25)]+[(None,0.)]*3;p=QuotePlanner();p.plan(0,w)
        self.assertEqual(aligned_intervals(p,0,100,w),[[(40,100)],[],[],[]])


if __name__=='__main__':unittest.main()
