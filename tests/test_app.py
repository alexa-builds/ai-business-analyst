import unittest
import base64
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def category_filter(self, app, column):
        matches = [widget for widget in app.multiselect if widget.label == column]
        self.assertEqual(len(matches), 1, f"Expected one category filter for {column!r}; got {[item.label for item in app.multiselect]}")
        return matches[0]

    def test_ai_analyst_uses_uploaded_columns_and_answers_from_filtered_data(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        csv = b"period_name,market_area,gross_margin\nJanuary,East,3\nFebruary,East,8\nMarch,West,99\nFebruary,East,7\n"
        app.file_uploader[0].set_value(("portfolio.csv", csv, "text/csv")).run()

        self.assertFalse(app.exception)
        self.assertTrue(any("AI Business Analyst" in header.value for header in app.header))
        self.assertIn("Which period_name has the highest gross_margin?", app.selectbox(key="ai-example-question").options)
        self.category_filter(app, "market_area").set_value(["East"]).run()
        app.text_input(key="ai-question").set_value("Which period_name had the highest gross_margin?")
        app.button(key="ai-question-submit").click().run()

        self.assertFalse(app.exception)
        markdown_text = " ".join(str(element.value) for element in app.markdown)
        self.assertIn("period_name: February has the highest", markdown_text, markdown_text)
        self.assertTrue(any("15.00" in str(element.value) for element in app.metric))

    def test_ai_analyst_uses_full_source_for_duplicate_id_questions(self):
        project_root = Path(__file__).resolve().parents[1]
        app_path = project_root / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        csv = (project_root / "retail_business_test.csv").read_bytes()
        app.file_uploader[0].set_value(("retail_business_test.csv", csv, "text/csv")).run()

        app.text_input(key="ai-question").set_value("How many duplicate order IDs are in the dataset?")
        app.button(key="ai-question-submit").click().run()
        markdown_text = " ".join(str(element.value) for element in app.markdown)
        self.assertFalse(app.exception)
        self.assertIn("RB-2026-0027 (2 rows)", markdown_text)

        app.text_input(key="ai-question").set_value(
            "What is the total revenue after removing duplicate order IDs, keeping the first occurrence?"
        )
        app.button(key="ai-question-submit").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("54,992.40" in str(element.value) for element in app.metric))

    def test_business_performance_duplicate_policy_is_explicit_and_updates_metrics(self):
        project_root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(project_root / "app.py"), default_timeout=10).run()
        csv = (project_root / "retail_business_test.csv").read_bytes()
        app.file_uploader[0].set_value(("retail_business_test.csv", csv, "text/csv")).run()

        metric_values = [str(element.value) for element in app.metric]
        self.assertIn("55,282.68", metric_values)
        app.checkbox(key="analysis-deduplicate-ids").check().run()
        self.assertFalse(app.exception)
        metric_values = [str(element.value) for element in app.metric]
        self.assertIn("54,992.40", metric_values)
        text = " ".join(str(element.value) for element in app.caption)
        self.assertIn("first identifier occurrence retained", text)
        app.text_input(key="ai-question").set_value("What is the total revenue across all transactions?")
        app.button(key="ai-question-submit").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("54,992.40" in str(element.value) for element in app.metric))

    def test_grouped_calculation_control_updates_displayed_group_results(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        csv = b"segment,amount\nA,2\nA,4\nB,10\nB,14\n"
        app.file_uploader[0].set_value(("grouped.csv", csv, "text/csv")).run()
        app.multiselect(key="group-dimensions").set_value(["segment"]).run()

        def grouped_values(test_app):
            for element in test_app.dataframe:
                frame = element.value
                if isinstance(frame, pd.DataFrame) and len(frame) == 2 and {"segment", "amount"}.issubset(frame.columns):
                    return frame.set_index("segment")["amount"].to_dict()
            self.fail("Grouped comparison table was not rendered")

        self.assertEqual(grouped_values(app), {"A": 6.0, "B": 24.0})
        app.selectbox(key="group-aggregation").select("Average").run()
        self.assertFalse(app.exception)
        self.assertEqual(grouped_values(app), {"A": 3.0, "B": 12.0})
        app.selectbox(key="chart-type").select("Pie").run()
        self.assertFalse(app.exception)
        chart_spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual(chart_spec["data"][0]["type"], "pie")
        pie_values = chart_spec["data"][0]["values"]
        decoded_values = np.frombuffer(base64.b64decode(pie_values["bdata"]), dtype=float).tolist()
        self.assertEqual(decoded_values, [24.0, 6.0])
        self.assertEqual(app.selectbox(key="chart-type").value, "Pie")

    def test_date_and_numeric_filters_reset_and_update_shared_row_counts(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        csv = (
            b"order_date,region,revenue,profit\n"
            b"2026-01-01,North,10,1\n"
            b"2026-01-02,North,20,3\n"
            b"2026-01-03,South,30,5\n"
        )
        app.file_uploader[0].set_value(("filtered.csv", csv, "text/csv")).run()
        app.date_input(key="date-filter-order_date").set_value((pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-03").date())).run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "2")
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Records"), "2")

        app.button(key="reset-analysis-filters").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "3")
        app.slider(key="numeric-filter-revenue").set_value((10.0, 15.0)).run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "1")
        app.button(key="reset-analysis-filters").click().run()
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "3")
        self.category_filter(app, "region").set_value(["North"]).run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "2")
        app.button(key="reset-analysis-filters").click().run()
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "3")

    def test_missing_filter_default_applies_consistently_and_reset_restores_missing_category(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        csv = (
            b"order_id,region,revenue,note\n"
            b"A,North,0,ok\n"
            b"B,North,10,ok\n"
            b"C,,20,ok\n"
            b"D,West,30,\n"
            b"D,West,999,ok\n"
        )
        app.file_uploader[0].set_value(("missing.csv", csv, "text/csv")).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "5")
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Revenue"), "1,059.00")
        self.assertEqual(app.selectbox(key="chart-type").value, "Bar")
        self.assertIn("Pie", app.selectbox(key="chart-type").options)

        category_filter = self.category_filter(app, "region")
        self.assertIn("(Missing)", category_filter.options)
        app.selectbox(key="chart-type").select("Pie").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="chart-type").value, "Pie")
        self.assertTrue(app.get("plotly_chart"))

        app.checkbox(key="analysis-deduplicate-ids").check().run()
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "4")
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Revenue"), "60.00")
        self.assertEqual(self.category_filter(app, "region").options, ["North", "West", "(Missing)"])
        app.checkbox(key="exclude-missing-values").check().run()
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "2")
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Revenue"), "10.00")
        # The category domain is built after both row rules, so West and the
        # missing-region option disappear when their only rows are excluded.
        self.assertEqual(self.category_filter(app, "region").options, ["North"])
        app.button(key="reset-analysis-filters").click().run()
        self.assertFalse(app.checkbox(key="analysis-deduplicate-ids").value)
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "5")
        restored_after_reset = self.category_filter(app, "region")
        self.assertEqual(restored_after_reset.options, ["North", "West", "(Missing)"])
        self.assertEqual(restored_after_reset.value, ["North", "West", None])
        self.assertTrue(any("Active filters: None" in str(caption.value) for caption in app.caption))

        self.category_filter(app, "region").set_value(["North", "West"]).run()
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "4")
        app.button(key="reset-analysis-filters").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "5")
        restored = self.category_filter(app, "region")
        self.assertIn(None, restored.value)

        app.checkbox(key="exclude-missing-values").check().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "3")
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Revenue"), "1,009.00")
        self.assertEqual(self.category_filter(app, "region").options, ["North", "West"])
        self.assertTrue(any("2 missing-value row(s) excluded" in str(caption.value) for caption in app.caption))
        app.text_input(key="ai-question").set_value("What is the total revenue across all transactions?")
        app.button(key="ai-question-submit").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("1,009.00" in str(element.value) for element in app.markdown))
        app.button(key="reset-analysis-filters").click().run()
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "5")
        self.assertEqual(self.category_filter(app, "region").options, ["North", "West", "(Missing)"])

    def test_missing_exclusion_benchmarks_are_calculated_from_the_loaded_dataset(self):
        project_root = Path(__file__).resolve().parents[1]
        source = pd.read_csv(project_root / "retail_business_test.csv")
        blank_cell = source.apply(lambda column: column.map(lambda value: isinstance(value, str) and not value.strip()))
        missing_rows = source.isna().any(axis=1) | blank_cell.any(axis=1)
        expected = source.loc[~missing_rows]

        app = AppTest.from_file(str(project_root / "app.py"), default_timeout=10).run()
        app.file_uploader[0].set_value(("retail_business_test.csv", (project_root / "retail_business_test.csv").read_bytes(), "text/csv")).run()
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        app.checkbox(key="exclude-missing-values").check().run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), f"{len(expected):,}")
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Revenue"), f"{expected['revenue'].sum():,.2f}")
        self.assertTrue(any("missing-value row(s) excluded" in str(caption.value) for caption in app.caption))

    def test_retail_region_options_reconcile_when_missing_rows_are_excluded_and_reset(self):
        project_root = Path(__file__).resolve().parents[1]
        source_path = project_root / "retail_business_test.csv"
        app = AppTest.from_file(str(project_root / "app.py"), default_timeout=10).run()
        app.file_uploader[0].set_value((source_path.name, source_path.read_bytes(), "text/csv")).run()
        self.assertFalse(app.exception)

        region = self.category_filter(app, "region")
        self.assertEqual(set(region.options), {"North", "South", "East", "West", "(Missing)"})
        # Select only the missing value to prove that changing the option domain
        # clears an obsolete selection instead of leaving stale widget state.
        region.set_value([None]).run()
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "1")
        app.multiselect(key="selected-measures").set_value(["revenue", "profit"]).run()
        app.selectbox(key="chart-type").select("Pie").run()
        app.selectbox(key="chart-measure").select("profit").run()
        self.assertEqual(self.category_filter(app, "region").options, region.options)

        app.checkbox(key="exclude-missing-values").check().run()
        self.assertFalse(app.exception)
        region = self.category_filter(app, "region")
        self.assertEqual(set(region.options), {"North", "South", "East", "West"})
        self.assertNotIn(None, region.value)
        self.assertEqual(set(region.value), set(region.options))
        self.assertTrue(any("Rows analyzed" == metric.label and metric.value == "297" for metric in app.metric))
        self.assertEqual(app.selectbox(key="chart-type").value, "Pie")
        self.assertEqual(app.selectbox(key="chart-measure").value, "profit")

        # An empty category selection is a legitimate zero-row state. Reset
        # must create a fresh widget with default selections and restore rows.
        region.set_value([]).run()
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "0")
        self.assertEqual(self.category_filter(app, "region").options, region.options)

        # The source quality profile remains the full-upload profile even while
        # the analysis frame excludes missing rows.
        source_quality = app.dataframe[1].value
        region_quality = source_quality.loc[source_quality["Column"] == "region", "Missing"].iloc[0]
        self.assertEqual(int(region_quality), 1)

        app.checkbox(key="exclude-missing-values").uncheck().run()
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "300")
        restored_by_toggle = self.category_filter(app, "region")
        self.assertIn("(Missing)", restored_by_toggle.options)
        self.assertIn(None, restored_by_toggle.value)
        app.checkbox(key="exclude-missing-values").check().run()
        self.assertNotIn("(Missing)", self.category_filter(app, "region").options)

        app.button(key="reset-analysis-filters").click().run()
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "300")
        restored = self.category_filter(app, "region")
        self.assertEqual(set(restored.options), {"North", "South", "East", "West", "(Missing)"})
        self.assertIn(None, restored.value)
        restored_labels = {"(Missing)" if value is None else value for value in restored.value}
        self.assertEqual(restored_labels, set(restored.options))

        app.checkbox(key="exclude-missing-values").check().run()
        self.assertNotIn("(Missing)", self.category_filter(app, "region").options)
        app.checkbox(key="exclude-missing-values").uncheck().run()
        self.category_filter(app, "region").set_value([]).run()
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "0")
        app.button(key="reset-analysis-filters").click().run()
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "300")
        restored_after_empty = self.category_filter(app, "region")
        self.assertEqual(set(restored_after_empty.options), {"North", "South", "East", "West", "(Missing)"})
        self.assertIn(None, restored_after_empty.value)

    def test_retail_report_renders_three_distinct_recommendations_under_both_duplicate_policies(self):
        project_root = Path(__file__).resolve().parents[1]
        source_path = project_root / "retail_business_test.csv"
        app = AppTest.from_file(str(project_root / "app.py"), default_timeout=10).run()
        app.file_uploader[0].set_value((source_path.name, source_path.read_bytes(), "text/csv")).run()

        def recommendations():
            return [str(markdown.value) for markdown in app.markdown if "Priority " in str(markdown.value)]

        included = recommendations()
        self.assertEqual(len(included), 3)
        self.assertTrue(any("Jan 2026" in item and "Jun 2026" in item for item in included))
        self.assertTrue(any("Air Fryer" in item and "Yoga Mat" in item for item in included))
        self.assertEqual(len({item.split(":", 1)[0] for item in included}), 3)
        self.assertFalse(any("duplicate" in item.lower() or "identifier" in item.lower() for item in included))
        self.assertTrue(any("South" in item and "4,231.33" in item and "West" in item and "9,167.50" in item for item in included))
        self.assertFalse(any("Priority 1" in item and ("duplicate" in item.lower() or "identifier" in item.lower()) for item in included))

        app.checkbox(key="analysis-deduplicate-ids").check().run()
        self.assertFalse(app.exception)
        kept_first = recommendations()
        self.assertEqual(len(kept_first), 3)
        self.assertTrue(any("Air Fryer" in item and "Yoga Mat" in item for item in kept_first))
        self.assertFalse(any("remove duplicate" in item.lower() or "keep first" in item.lower() for item in kept_first))
        self.assertTrue(any("Duplicate IDs: 1 later row(s) excluded" in str(caption.value) for caption in app.caption))

    def test_new_upload_resets_stale_analysis_and_filter_controls(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        app.file_uploader[0].set_value(("first.csv", b"area,amount\nA,1\nB,2\n", "text/csv")).run()
        app.multiselect(key="group-dimensions").set_value(["area"]).run()
        self.category_filter(app, "area").set_value(["A"]).run()
        app.checkbox(key="exclude-missing-values").check().run()
        app.file_uploader[0].set_value(("second.csv", b"region,revenue\nWest,10\nEast,20\n", "text/csv")).run()

        self.assertFalse(app.exception)
        self.assertIn("revenue", app.multiselect(key="selected-measures").options)
        self.assertIn("region", app.multiselect(key="group-dimensions").options)
        self.assertTrue(self.category_filter(app, "region"))
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "2")

    def test_worksheet_switch_restores_filter_defaults_and_clears_stale_state(self):
        workbook_bytes = BytesIO()
        with pd.ExcelWriter(workbook_bytes, engine="openpyxl") as writer:
            pd.DataFrame({"region": ["North", None], "revenue": [10, 20]}).to_excel(writer, index=False, sheet_name="First")
            pd.DataFrame({"product": ["A", "B"], "amount": [3, 4]}).to_excel(writer, index=False, sheet_name="Second")
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        app.file_uploader[0].set_value(("workbook.xlsx", workbook_bytes.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")).run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "2")
        app.checkbox(key="exclude-missing-values").check().run()
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "1")
        worksheet = next(item for item in app.selectbox if item.label == "Worksheet")
        worksheet.select("Second").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.checkbox(key="exclude-missing-values").value)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "2")
        self.assertIn("amount", app.multiselect(key="selected-measures").options)

    def test_sample_csvs_expose_month_filters_grouping_and_line_charts(self):
        project_root = Path(__file__).resolve().parents[1]
        app_path = project_root / "app.py"
        for filename in ("sales.csv", "test.csv"):
            with self.subTest(filename=filename):
                app = AppTest.from_file(str(app_path), default_timeout=10).run()
                csv_content = (project_root / filename).read_bytes()
                app.file_uploader[0].set_value((filename, csv_content, "text/csv")).run()

                self.assertFalse(app.exception)
                self.assertIn("sales", app.multiselect(key="selected-measures").options)
                self.assertIn("month", app.multiselect(key="group-dimensions").options)
                self.assertTrue(self.category_filter(app, "month"))

                app.multiselect(key="group-dimensions").set_value(["month"]).run()
                self.assertFalse(app.exception)
                self.assertIn("Line", app.selectbox(key="chart-type").options)
                self.assertTrue(app.get("plotly_chart"))

                self.category_filter(app, "month").set_value(["January"]).run()
                app.selectbox(key="chart-type").select("Bar").run()
                self.assertFalse(app.exception)
                self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Valid values"), "1")
                app.selectbox(key="chart-type").select("Line").run()
                self.assertFalse(app.exception)
                self.assertTrue(app.get("plotly_chart"))

    def test_empty_upload_state_renders_without_an_exception(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()

        self.assertFalse(app.exception)
        self.assertTrue(any("Choose a CSV" in info.value for info in app.info))

    def test_uploaded_data_renders_analysis_and_supports_date_line_chart(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        csv = (
            b"date,region,sales,cost\n"
            b"2025-01-01,A,10,5\n"
            b"2025-02-01,A,12,4\n"
            b"2025-01-01,B,0,2\n"
            b"2025-02-01,B,3,1\n"
        )
        app.file_uploader[0].set_value(("sample.csv", csv, "text/csv")).run()

        self.assertFalse(app.exception)
        self.assertTrue(any("Business insights" in subheader.value for subheader in app.subheader))
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "4")
        app.multiselect(key="group-dimensions").set_value(["region"]).run()
        app.selectbox(key="chart-type").select("Bar").run()
        self.assertFalse(app.exception)
        app.selectbox(key="chart-type").select("Box").run()
        self.assertFalse(app.exception)
        app.selectbox(key="chart-type").select("Scatter").run()
        self.assertFalse(app.exception)
        self.category_filter(app, "region").set_value(["A"]).run()
        self.assertFalse(app.exception)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "Rows analyzed"), "2")
        app.selectbox(key="analysis-date-column").select("date").run()
        app.selectbox(key="chart-type").select("Line").run()
        app.selectbox(key="chart-type").select("Line").run()

        self.assertFalse(app.exception)
        self.assertTrue(app.get("plotly_chart"))


if __name__ == "__main__":
    unittest.main()
