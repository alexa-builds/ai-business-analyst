"""Tests for data-driven chart labels and plotted values."""

import unittest

import pandas as pd

from analysis import aggregate_data
from charts import build_category_bar_chart, build_category_pie_chart, build_distribution_histogram, style_figure


class ChartTests(unittest.TestCase):
    def test_chart_titles_legends_and_annotations_are_left_aligned(self):
        figure = build_category_bar_chart(
            pd.DataFrame({"segment": ["North", "South"], "sales": [20, 10]}),
            "sales",
            ["segment"],
            "Sum",
        )
        figure.add_annotation(text="Evidence note", x=0, y=1, xref="paper", yref="paper")
        figure = style_figure(figure)

        self.assertEqual(figure.layout.title.x, 0)
        self.assertEqual(figure.layout.title.xanchor, "left")
        self.assertEqual(figure.layout.legend.x, 0)
        self.assertEqual(figure.layout.legend.xanchor, "left")
        self.assertEqual(figure.layout.annotations[0].align, "left")

    def test_histogram_labels_observations_and_ignores_group_aggregation(self):
        frame = pd.DataFrame({"Sales Volume": [1, 1, 3, None]})
        figure = build_distribution_histogram(frame, "Sales Volume")

        self.assertEqual(figure.layout.title.text, "Distribution of Sales Volume")
        self.assertEqual(figure.layout.xaxis.title.text, "Sales Volume value")
        self.assertEqual(figure.layout.yaxis.title.text, "Number of observations")
        self.assertEqual(len(figure.data[0].x), 3)
        self.assertEqual(figure.data[0].histfunc, "count")
        self.assertGreaterEqual(figure.data[0].nbinsx, 1)

    def test_grouped_aggregation_updates_bar_chart_values(self):
        frame = pd.DataFrame({"segment": ["A", "A", "B", "B"], "amount": [2, 4, 10, 14]})
        sum_data = aggregate_data(frame, ["amount"], ["segment"], "Sum")
        mean_data = aggregate_data(frame, ["amount"], ["segment"], "Average")

        sum_chart = build_category_bar_chart(sum_data, "amount", ["segment"], "Sum")
        mean_chart = build_category_bar_chart(mean_data, "amount", ["segment"], "Average")

        self.assertEqual(list(sum_chart.data[0].y), [6.0, 24.0])
        self.assertEqual(list(mean_chart.data[0].y), [3.0, 12.0])
        self.assertEqual(sum_chart.layout.title.text, "Sum amount by category")
        self.assertEqual(mean_chart.layout.title.text, "Average amount by category")
        self.assertEqual(mean_chart.layout.title.x, 0)
        self.assertEqual(mean_chart.layout.legend.x, 0)
        self.assertEqual(mean_chart.layout.legend.orientation, "h")

    def test_pie_chart_shows_category_shares_from_summed_values(self):
        grouped = pd.DataFrame({"region": ["East", "West", "North"], "revenue": [60, 30, 10]})
        figure, note = build_category_pie_chart(grouped, "revenue", ["region"])

        self.assertIsNone(note)
        self.assertEqual(list(figure.data[0].labels), ["East", "West", "North"])
        self.assertEqual(list(figure.data[0].values), [60.0, 30.0, 10.0])
        self.assertEqual(figure.data[0].textinfo, "label+percent")
        self.assertIn("Share: %{percent}", figure.data[0].hovertemplate)
        self.assertEqual(figure.layout.title.x, 0)

    def test_pie_chart_rejects_negative_and_zero_totals_and_summarizes_many_groups(self):
        negative, reason = build_category_pie_chart(
            pd.DataFrame({"group": ["A", "B"], "value": [10, -2]}), "value", ["group"]
        )
        self.assertIsNone(negative)
        self.assertIn("negative", reason)
        zero, reason = build_category_pie_chart(
            pd.DataFrame({"group": ["A", "B"], "value": [0, 0]}), "value", ["group"]
        )
        self.assertIsNone(zero)
        self.assertIn("zero total", reason)
        crowded, note = build_category_pie_chart(
            pd.DataFrame({"group": [f"G{i}" for i in range(20)], "value": list(range(20, 0, -1))}),
            "value", ["group"],
        )
        self.assertEqual(len(crowded.data[0].labels), 12)
        self.assertEqual(crowded.data[0].labels[-1], "Other")
        self.assertIn("combined into Other", note)
        missing_dimension, reason = build_category_pie_chart(
            pd.DataFrame({"value": [1]}), "value", ["region"]
        )
        self.assertIsNone(missing_dimension)
        self.assertIn("unavailable", reason)


if __name__ == "__main__":
    unittest.main()
