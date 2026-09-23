"""
Full reproduction check (about 15 s). Runs the complete analysis on the data in ./data and
compares the key numbers against expected_results/. Run from the repository root:
    python tests/test_reproduce.py
"""
import os, subprocess, sys, tempfile, unittest
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.join(ROOT, "expected_results")


class Reproduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp()
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "ajmer_lst_analysis.py"),
                        os.path.join(ROOT, "data"), cls.out], check=True, capture_output=True)

    def _compare(self, name, key, cols, tol):
        a = pd.read_csv(os.path.join(self.out, name)).set_index(key)
        b = pd.read_csv(os.path.join(EXP, name)).set_index(key)
        self.assertEqual(list(a.index), list(b.index))
        for c in cols:
            diff = (a[c] - b[c]).abs().max()
            self.assertLessEqual(diff, tol, f"{name}:{c} differs by {diff}")

    def test_table1(self):
        self._compare("table1_trends.csv", "series",
                      ["MODIS_slope", "MODIS_p", "ERA5 skt_slope", "ERA5 skt_p", "ERA5 t2m_slope"], 1e-6)

    def test_table2(self):
        self._compare("table2_scene_comparison.csv", "series", ["n", "bias", "sd", "r"], 1e-6)

    def test_table3(self):
        self._compare("table3_phenology.csv", "metric", ["slope", "lo", "hi", "p"], 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
