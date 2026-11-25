import unittest
from tools.analyze_t2t import final_fate, retention_counts


class RetentionTest(unittest.TestCase):
    def test_native_output_is_authoritative_including_final_eos(self):
        for i,expected in enumerate(("old_id","new_id","other_id","not_returned")):
            self.assertEqual(final_fate(i,[4,5,2],4,5),expected)
        for i,b,a in ((-1,4,5),(0,4,4)):
            with self.assertRaises(ValueError):
                final_fate(i,[],b,a)

    def test_strict_only_stratified_on_stage_and_previous_freshness(self):
        def row(ev, phase, fresh, fate):
            return dict(event=ev,native_phase=phase,previously_fresh=fresh,final_fate=fate)
        c = retention_counts([row("t2t","selected",0,"old_id"),
                              row("t2t","verify",1,"not_returned"),
                              row("readmit_different","selected",0,"")])
        self.assertEqual([r["strict_edits"] for r in c],[1,1])
        self.assertEqual(c[0]["old_id"],1)
        self.assertEqual(c[1]["not_returned"],1)
