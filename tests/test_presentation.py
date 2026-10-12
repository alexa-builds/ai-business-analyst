"""Tests for display-only table formatting and left alignment."""

import unittest

import pandas as pd

from analysis import profile_dataset
from presentation import business_table_for_display, dataframe_column_config


class TablePresentationTests(unittest.TestCase):
    def test_every_column_type_is_left_aligned_without_changing_data(self):
        frame = pd.DataFrame({
            "label": ["North", "South"],
            "amount": [1234, 9876],
            "rate": [0.125, 0.25],
            "source_number_text": ["1.25", "2.50"],
            "observed_at": pd.to_datetime(["2026-01-02", "2026-02-03"]),
        })
        original_dtypes = frame.dtypes.copy()
        profiles = profile_dataset(frame)[1]

        configs = dataframe_column_config(frame, profiles)

        self.assertEqual(set(configs), set(frame.columns))
        self.assertTrue(all(config["alignment"] == "left" for config in configs.values()))
        self.assertEqual(configs["amount"]["type_config"]["format"], "%,.0f")
        self.assertEqual(configs["rate"]["type_config"]["format"], "%,.2f")
        self.assertEqual(configs["observed_at"]["type_config"]["format"], "YYYY-MM-DD HH:mm")
        pd.testing.assert_series_equal(frame.dtypes, original_dtypes)

    def test_business_margin_table_displays_percent_without_mutating_analysis(self):
        frame = pd.DataFrame({"region": ["West"], "profit_margin": [0.512]})
        displayed = business_table_for_display(frame)
        self.assertEqual(displayed.columns[-1], "Profit margin (%)")
        self.assertAlmostEqual(displayed.iloc[0, 1], 51.2)
        self.assertTrue(pd.api.types.is_numeric_dtype(displayed.iloc[:, 1]))
        self.assertAlmostEqual(frame.loc[0, "profit_margin"], 0.512)


if __name__ == "__main__":
    unittest.main()
