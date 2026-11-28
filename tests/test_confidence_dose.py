import unittest

from bdlm_serving.methods.t2t.confidence_dose import ConfidenceDose


class TestConfidenceDose(unittest.TestCase):
    def test_repeat_supplies_dose_not_placement(self):
        led = ConfidenceDose()
        o = dict(block=4,positions=[1101,1102,1103,1104,1105],proposals=[2,3,4,5,7],
                 probabilities=[.8,.8,.7,.99,.99],native=[False,False,False,True,False],
                 excluded_ids={7})
        self.assertEqual(led.observe(tick=1, **o), ())
        o["proposals"]=[9,8,4,5,7]
        self.assertEqual(led.observe(tick=2, **o), (1101,))
        self.assertEqual(led.observe(tick=4, **o), ())
        o["block"]=5
        self.assertEqual(led.observe(tick=5, **o), ())

    def test_disallows_below_floor_and_special(self):
        led=ConfidenceDose()
        o=dict(block=0,positions=[1,2,3],proposals=[1,2,3],probabilities=[.5,.99,.51],
               native=[False]*3,excluded_ids={2})
        led.observe(tick=0,**o)
        self.assertEqual(led.observe(tick=1,**o),(3,))
