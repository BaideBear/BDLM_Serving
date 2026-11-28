import unittest
from types import SimpleNamespace as NS
from experiments.t2t.serving_slots.gemma_native_visibility import eligible, representation


class TestVisibility(unittest.TestCase):
    def test_reject_boundaries(self):
        kw=dict(size=256,native_mask=None,dynamic=True,solo=True,bidirectional=True)
        for live,head in (([9,12],0),([266,300],1)):
            self.assertTrue(eligible(live,9,head,**kw))
        for live in ([],[8,12],[264,265]):
            self.assertFalse(eligible(live,9,0,**kw))
        for key,val in (("native_mask",object()),("dynamic",False),("solo",False),("bidirectional",False)):
            self.assertFalse(eligible([9],9,0,**dict(kw,**{key:val})))

    def test_real_write_before_mask_and_restore(self):
        writes=[]
        k,v=object(),object()
        mask=NS(all=lambda:True)
        def pay(layer,*args):
            writes.append(layer)
            return k,v,mask
        bank=NS(extend=pay,live=[9],canvas_origin=9)
        req=NS(bank=bank,cache=NS(is_compileable=False),model=NS(model=NS(decoder=NS(
               layers=[NS(self_attn=NS(is_causal=False))]))),tail=None,rebuild=False,ordinal=0)
        for enabled in (True,False):
            hits=[]
            with representation(req,enabled,hits):
                self.assertEqual(bank.extend(2,None,None,None),(k,v,None if enabled else mask))
                req.tail=object()
                self.assertIs(bank.extend(2,None,None,None)[2],mask)
                req.tail=None
            self.assertIs(bank.extend,pay)
            self.assertEqual(hits,[2])
        self.assertEqual(writes,[2]*4)
        mask.all=lambda:False
        with self.assertRaises(ValueError),representation(req,True,[]):
            bank.extend(2,None,None,None)
        self.assertIs(bank.extend,pay)
