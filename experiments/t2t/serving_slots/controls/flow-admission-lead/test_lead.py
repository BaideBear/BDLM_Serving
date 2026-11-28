import unittest
from lead import LeadState


class LeadTest(unittest.TestCase):
    def test_forecast_and_resets(self):
        s=LeadState()
        for head,opened,f,allowed in [(2,1,.6,False),(2,1,.71,True),(2,2,.75,False),
                                      (2,1,.76,False),(3,1,.79,False),(3,1,.78,False),
                                      (3,1,.795,True),(3,1,.8,False)]:
            s.observe(head,opened,False,f)
            self.assertEqual(s.opportunity(head,head+1,10,False,f),allowed)

    def test_only_unfinished_physical_successor(self):
        s=LeadState()
        s.observe(0,1,False,.4)
        s.observe(0,1,False,.7)
        for c,right,total,finished in [(0,1,1,False),(0,2,5,False),(0,1,5,True),(1,2,5,False)]:
            self.assertFalse(s.opportunity(c,right,total,finished,.7))
        s.observe(0,1,True,.79)
        self.assertIsNone(s.prev_f)
