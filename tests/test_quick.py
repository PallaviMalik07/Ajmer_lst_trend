"""
Quick test (runs in a few seconds, no project data needed).

Checks the statistical building blocks of the pipeline on a synthetic example file with a
known answer, plus the bit-count and ERA5 interpolation helpers.

Run from the repository root:
    python tests/test_quick.py
"""
import os, sys, unittest
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from lstlib import sen, month_anom, annual, popcount, interp_hourly, partial_r

EXAMPLE = os.path.join(HERE, "example_data", "synthetic_lst_8day.csv")
TRUE_TREND = -1.5   # K per decade built into the synthetic file (-0.15 K per year)


class QuickTest(unittest.TestCase):

    def test_example_file_trend_recovered(self):
        """Month-anomaly + annual mean + Theil-Sen recovers the known -1.5 K/decade."""
        df = pd.read_csv(EXAMPLE, parse_dates=["date"]).set_index("date")
        self.assertEqual(len(df), 1051)
        df["anom"] = month_anom(df, "mean")
        r = sen(annual(df["anom"], 2001, 2021))
        print(f"\n  synthetic example: recovered {r['slope']:+.2f} K/dec "
              f"[{r['lo']:+.2f}, {r['hi']:+.2f}], p = {r['p']:.2g} (true {TRUE_TREND:+.2f})")
        self.assertEqual(r["n"], 21)
        self.assertAlmostEqual(r["slope"], TRUE_TREND, delta=0.2)
        self.assertLess(r["lo"], TRUE_TREND); self.assertGreater(r["hi"], TRUE_TREND)
        self.assertLess(r["p"], 0.001)

    def test_null_series_not_significant(self):
        rng = np.random.default_rng(1)
        s = pd.Series(rng.normal(0, 1, 21), index=np.arange(2001, 2022))
        self.assertGreater(sen(s)["p"], 0.05)

    def test_clear_sky_bitcount(self):
        """MODIS Clear_sky_days is an 8-bit mask: 249 = 0b11111001 -> 6 clear days."""
        df = pd.DataFrame([[249, 255, 0, np.nan]])
        self.assertEqual(popcount(df).values.tolist(), [[6, 8, 0, 0]])

    def test_hourly_interpolation(self):
        idx = pd.date_range("2010-01-01 05:00", periods=2, freq="h")
        out = interp_hourly(idx.values, np.array([300.0, 306.0]),
                            [np.datetime64("2010-01-01T05:30:00")])
        self.assertAlmostEqual(out[0], 303.0)
        idx2 = pd.DatetimeIndex(["2010-01-01 06:00", "2010-01-01 08:00"])   # 2 h gap -> NaN
        out2 = interp_hourly(idx2.values, np.array([300.0, 306.0]),
                             [np.datetime64("2010-01-01T07:00:00")])
        self.assertTrue(np.isnan(out2[0]))

    def test_partial_correlation(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=200); x = z + rng.normal(size=200); y = z + rng.normal(size=200)
        r, p = partial_r(x, y, z)            # x and y related only through z
        self.assertLess(abs(r), 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
