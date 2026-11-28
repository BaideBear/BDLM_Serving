import unittest

from rank import Versions


class Ranking(unittest.TestCase):
    def test_change_after_projection_on_same_tick(self):
        v=Versions()
        v.project(2,[1,2,3])
        v.write(2,[0])
        v.write(3,[0,33])
        d=v.plan([1,2,3],2,0,32)
        self.assertEqual(d["right"],[2,3])
        self.assertEqual(d["change"],[1,2])
        self.assertEqual(d["candidates"][0],(1,4,[0]))
        v.project(4,[1,2,3])
        self.assertEqual(v.plan([1,2,3],2,0,32)["change"],[2,3])

    def test_prompt_pad_excluded_and_zero_seats(self):
        v=Versions()
        v.project(0,[30,31])
        v.write(0,[1,32])
        d=v.plan([30,31],1,30,32)
        self.assertEqual(d["right"],d["change"])
        self.assertEqual(v.plan([31],0,30,32)["change"],[])
