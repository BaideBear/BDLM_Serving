import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from plot_summary import COLUMNS, read_grid


class PlotSummaryTest(unittest.TestCase):
    def test_invalid_input(self):
        for body in ("4,IBS,1,,4", "4,IBS,nan,2,3", "4,IBS,1,101,2",
                     "4,IBS,1,1,31", "4,IBS,1,1,3\n4,IBS,,,"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as d:
                src = Path(d)/"data.csv"
                src.write_text(",".join(COLUMNS)+"\n"+body+"\n")
                with self.assertRaises(ValueError):
                    read_grid(src)

    def test_cli_template_and_values(self):
        template = ROOT/"tools/templates/gemma-optimus.csv"
        cols, rows = read_grid(template)
        self.assertEqual(cols, ["IBS","OBS"])
        self.assertEqual([r for r, _ in rows], [f"C={2**i}" for i in range(2,9)])
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d)/"input.csv", Path(d)/"charts"/"result"
            for content, marks in [(template.read_text(), ["-"] * 42),
                                   (",".join(COLUMNS)+"\nC=64,IBS,0,0,0\nC=64,OBS,12.345,8.42,29\n",
                                    ["0.000","0.00%","0/30","12.345","8.42%","29/30"])]:
                src.write_text(content)
                subprocess.run([sys.executable,ROOT/"tools/plot_summary.py",src,"--out",out],check=True)
                labels = [t.text for t in ET.parse(out.with_suffix(".svg")).iter("{http://www.w3.org/2000/svg}text")]
                for mark in set(marks):
                    self.assertEqual(labels.count(mark),marks.count(mark))
                self.assertTrue(out.with_suffix(".png").read_bytes().startswith(b"\x89PNG"))
