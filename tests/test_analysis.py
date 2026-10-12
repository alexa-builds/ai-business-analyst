import unittest
from io import BytesIO
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from analysis import (
    aggregate_data,
    apply_filters,
    calculate_business_performance,
    build_business_insights,
    build_html_report,
    detect_business_columns,
    exclude_rows_with_missing_values,
    dataframe_to_excel,
    generate_business_insights,
    generate_management_summary,
    get_excel_sheet_names,
    load_dataset,
    month_number,
    numeric_summary,
    profile_dataset,
    remove_duplicate_identifier_rows,
)


class NumericSummaryTests(unittest.TestCase):
    def test_summary_handles_zero_negative_missing_and_nonnumeric(self):
        result = numeric_summary(["0", "-2", None, "bad", 4])

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["nonnumeric_count"], 1)
        self.assertEqual(result["sum"], 2.0)
        self.assertAlmostEqual(result["mean"], 2 / 3)
        self.assertEqual(result["median"], 0.0)
        self.assertEqual(result["minimum"], -2.0)
        self.assertEqual(result["maximum"], 4.0)
        self.assertIsNotNone(result["standard_deviation"])

    def test_non_finite_values_do_not_produce_infinite_metrics(self):
        result = numeric_summary([1, float("inf"), float("-inf")])

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["nonnumeric_count"], 2)
        self.assertEqual(result["sum"], 1.0)

    def test_empty_and_single_observation_are_safe(self):
        empty = numeric_summary([None, "", "not a number"])
        one = numeric_summary([0])

        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["sum"])
        self.assertIsNone(empty["standard_deviation"])
        self.assertEqual(one["sum"], 0.0)
        self.assertIsNone(one["standard_deviation"])


class DataQualityTests(unittest.TestCase):
    def test_missing_row_rule_excludes_null_and_blank_cells_but_keeps_zero(self):
        frame = pd.DataFrame({"region": ["North", "", None, "West"], "amount": [0, 2, 3, 4]})
        selected, excluded = exclude_rows_with_missing_values(frame)

        self.assertEqual(excluded, 2)
        self.assertEqual(selected["region"].tolist(), ["North", "West"])
        self.assertEqual(selected["amount"].tolist(), [0, 4])
        self.assertEqual(len(frame), 4)

    def test_missing_row_rule_is_safe_for_empty_and_zero_column_frames(self):
        empty = pd.DataFrame(columns=["a", "b"])
        selected, excluded = exclude_rows_with_missing_values(empty)
        self.assertTrue(selected.empty)
        self.assertEqual(excluded, 0)
        no_columns = pd.DataFrame(index=[0, 1])
        selected, excluded = exclude_rows_with_missing_values(no_columns)
        self.assertEqual(len(selected), 2)
        self.assertEqual(excluded, 0)

    def test_sample_files_expose_month_as_category_and_sales_as_measure(self):
        project_root = Path(__file__).resolve().parents[1]
        for filename in ("sales.csv", "test.csv"):
            with self.subTest(filename=filename):
                frame = pd.read_csv(project_root / filename)
                _, profiles = profile_dataset(frame)
                self.assertFalse(profiles["month"]["is_numeric"])
                self.assertFalse(profiles["month"]["is_date"])
                self.assertTrue(profiles["month"]["is_month"])
                self.assertTrue(profiles["sales"]["is_numeric"])
                grouped = aggregate_data(frame, ["sales"], ["month"], "Sum")
                self.assertEqual(set(grouped["month"]), set(frame["month"]))

    def test_month_labels_have_calendar_order_without_date_parsing(self):
        self.assertEqual(month_number("January"), 1)
        self.assertEqual(month_number("Sept"), 9)
        self.assertIsNone(month_number("2025-01-01"))

    def test_profiles_mixed_numeric_text_dates_missing_and_duplicates(self):
        frame = pd.DataFrame(
            {
                "amount": ["10", "invalid", None, "10"],
                "when": ["2025-01-01", "bad-date", "2025-01-03", "2025-01-01"],
                "segment": ["A", "B", "B", "A"],
            }
        )
        before = frame.copy(deep=True)
        quality, profiles = profile_dataset(frame)

        self.assertTrue(profiles["amount"]["is_numeric"])
        self.assertEqual(profiles["amount"]["nonnumeric_count"], 1)
        self.assertEqual(profiles["amount"]["missing_count"], 1)
        self.assertTrue(profiles["when"]["is_date"])
        self.assertEqual(profiles["when"]["date_parse_failures"], 1)
        self.assertEqual(int(frame.duplicated().sum()), 1)
        self.assertEqual(list(frame.columns), list(before.columns))
        self.assertTrue(pd.isna(quality.loc[quality["Column"] == "segment", "Nonnumeric in numeric data"].iloc[0]))

    def test_empty_dataset_keeps_column_profiles(self):
        quality, profiles = profile_dataset(pd.DataFrame(columns=["sales", "region"]))

        self.assertEqual(len(quality), 2)
        self.assertEqual(profiles["sales"]["numeric_count"], 0)

    def test_date_named_column_reports_parse_failures_even_if_most_values_are_bad(self):
        _, profiles = profile_dataset(pd.DataFrame({"order_date": ["bad", "also bad", "2025-01-01"]}))

        self.assertTrue(profiles["order_date"]["is_date"])
        self.assertEqual(profiles["order_date"]["date_parse_failures"], 2)


class FilteringAndAggregationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "region": ["North", "South", None, "North"],
                "sales": ["0", "-5", "12", "bad"],
                "date": ["2025-01-01", "2025-01-02", "2025-02-01", "2025-02-03"],
            }
        )

    def test_filters_category_numeric_and_date_without_mutating_source(self):
        original = self.frame.copy(deep=True)
        selected = apply_filters(
            self.frame,
            category_filters={"region": ["North", None]},
            numeric_ranges={"sales": (-1, 0)},
            date_ranges={"date": ("2025-01-01", "2025-02-28")},
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["region"], "North")
        pd.testing.assert_frame_equal(self.frame, original)

    def test_empty_category_selection_returns_empty_frame(self):
        selected = apply_filters(self.frame, category_filters={"region": []})

        self.assertTrue(selected.empty)

    def test_grouping_supports_multiple_dimensions_and_text_numbers(self):
        frame = pd.DataFrame({"region": ["A", "A", "B"], "channel": ["Web", "Store", "Web"], "sales": ["2", "3", "-1"]})
        grouped = aggregate_data(frame, ["sales"], ["region", "channel"], "Sum")

        self.assertEqual(len(grouped), 3)
        self.assertEqual(grouped["sales"].sum(), 4.0)
        self.assertIn("region", grouped.columns)
        self.assertIn("channel", grouped.columns)

    def test_aggregation_options_change_values_and_count_valid_observations(self):
        frame = pd.DataFrame(
            {
                "segment": ["A", "A", "B", "B", "B", "C"],
                "amount": [2, 4, 10, None, "invalid", 0],
            }
        )

        sums = aggregate_data(frame, ["amount"], ["segment"], "Sum").set_index("segment")["amount"]
        means = aggregate_data(frame, ["amount"], ["segment"], "Average").set_index("segment")["amount"]
        counts = aggregate_data(frame, ["amount"], ["segment"], "Count").set_index("segment")["amount"]
        minimums = aggregate_data(frame, ["amount"], ["segment"], "Minimum").set_index("segment")["amount"]
        maximums = aggregate_data(frame, ["amount"], ["segment"], "Maximum").set_index("segment")["amount"]

        self.assertEqual(sums.to_dict(), {"A": 6.0, "B": 10.0, "C": 0.0})
        self.assertEqual(means.to_dict(), {"A": 3.0, "B": 10.0, "C": 0.0})
        self.assertEqual(counts.to_dict(), {"A": 2, "B": 1, "C": 1})
        self.assertEqual(minimums.to_dict(), {"A": 2.0, "B": 10.0, "C": 0.0})
        self.assertEqual(maximums.to_dict(), {"A": 4.0, "B": 10.0, "C": 0.0})
        with self.assertRaisesRegex(ValueError, "Unsupported aggregation"):
            aggregate_data(frame, ["amount"], ["segment"], "Mode")

    def test_ungrouped_aggregation_is_safe_for_empty_and_invalid_values(self):
        frame = pd.DataFrame({"amount": [None, "invalid"]})
        result = aggregate_data(frame, ["amount"], aggregation="Sum")
        count = aggregate_data(frame, ["amount"], aggregation="Count")

        self.assertTrue(pd.isna(result.loc[0, "amount"]))
        self.assertEqual(count.loc[0, "amount"], 0)

    def test_date_grouping_is_orderable(self):
        grouped = aggregate_data(self.frame, ["sales"], [], "Sum", "date", "Month")

        self.assertEqual(list(grouped["Date"].dt.month), [1, 2])
        self.assertEqual(list(grouped["sales"]), [-5.0, 12.0])


class InsightTests(unittest.TestCase):
    def test_month_dimension_supports_ordered_change_insight(self):
        frame = pd.DataFrame({"month": ["January", "February", "March"], "sales": [0, 8, 12]})
        text = " ".join(item["text"] for item in build_business_insights(frame, "sales", ["month"]))

        self.assertIn("From January to March in calendar-month order", text)
        self.assertIn("percentage change unavailable because the first value is zero", text)

    def test_zero_denominator_is_explicit_and_categories_are_calculated(self):
        frame = pd.DataFrame(
            {
                "period": ["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"],
                "region": ["A", "B", "A", "B"],
                "sales": [0, 0, 10, 1],
            }
        )
        insights = build_business_insights(frame, "sales", ["region"], "period", "Month")
        text = " ".join(item["text"] for item in insights)

        self.assertIn("Highest grouped sum", text)
        self.assertIn("lowest grouped sum", text)
        self.assertIn("percentage change unavailable because the first value is zero", text)

    def test_outlier_finding_uses_iqr_and_reports_duplicates(self):
        frame = pd.DataFrame({"value": [1, 1, 1, 1, 100, 1]})
        insights = build_business_insights(frame, "value")
        text = " ".join(item["text"] for item in insights)

        self.assertIn("Potential outliers by the 1.5×IQR rule: 1", text)
        self.assertIn("duplicate row(s)", text)
        self.assertIn("no date field was selected", text)

    def test_empty_or_all_invalid_measure_gets_insufficient_data_finding(self):
        insights = build_business_insights(pd.DataFrame({"value": [None, "bad"]}), "value")

        self.assertIn("No valid numeric observations", insights[0]["text"])


class BusinessPerformanceTests(unittest.TestCase):
    def test_retail_benchmark_keeps_duplicate_policy_explicit(self):
        project_root = Path(__file__).resolve().parents[1]
        source = pd.read_csv(project_root / "retail_business_test.csv")
        all_rows = calculate_business_performance(source)
        clean = calculate_business_performance(source, remove_duplicate_identifiers=True)

        self.assertEqual(all_rows["columns"]["rows"], 300)
        self.assertAlmostEqual(all_rows["metrics"]["revenue"]["total"], 55282.68)
        self.assertEqual(clean["columns"]["rows"], 299)
        self.assertEqual(clean["columns"]["duplicate_ids"], ["RB-2026-0027"])
        self.assertAlmostEqual(clean["metrics"]["revenue"]["total"], 54992.40)
        self.assertAlmostEqual(clean["metrics"]["cost"]["total"], 26834.70)
        self.assertAlmostEqual(clean["metrics"]["profit"]["total"], 28157.70)
        self.assertAlmostEqual(clean["metrics"]["profit_margin"], 0.5120, places=4)
        top_month = clean["time_series"].sort_values("revenue").iloc[-1]
        self.assertEqual(top_month["period"], pd.Timestamp("2026-06-01"))
        self.assertAlmostEqual(top_month["revenue"], 11135.49)
        product = clean["breakdowns"]["product"].sort_values("profit").iloc[-1]
        region = clean["breakdowns"]["region"].sort_values("profit").iloc[-1]
        self.assertEqual((product["product"], product["profit"]), ("Air Fryer", 7255.82))
        self.assertEqual((region["region"], region["profit"]), ("West", 9167.50))

    def test_management_report_uses_the_same_validated_benchmark_metrics(self):
        project_root = Path(__file__).resolve().parents[1]
        source = pd.read_csv(project_root / "retail_business_test.csv")
        performance = calculate_business_performance(source, remove_duplicate_identifiers=True)
        performance["columns"]["duplicate_policy"] = "first identifier occurrence retained"
        performance["columns"]["duplicate_scope"] = "uploaded source before filters"
        findings = generate_business_insights(performance)
        summary = generate_management_summary(performance, findings)
        quality, _ = profile_dataset(performance["frame"])
        report = build_html_report(
            performance["frame"], quality, {}, [], dataset_name="retail_business_test.csv",
            scope="299 rows; first identifier occurrence retained", management_summary=summary,
            performance={**performance, "business_insights": findings},
        )
        self.assertIn("54,992.40", report)
        self.assertIn("26,834.70", report)
        self.assertIn("28,157.70", report)
        self.assertIn("51.20%", report)
        self.assertIn("299", report)
        self.assertIn("first identifier occurrence retained", report)

    def test_management_report_uses_filtered_missing_value_scope_and_metrics(self):
        source = pd.read_csv(Path(__file__).resolve().parents[1] / "retail_business_test.csv")
        filtered, excluded = exclude_rows_with_missing_values(source)
        quality, _ = profile_dataset(filtered)
        performance = calculate_business_performance(filtered)
        insights = generate_business_insights(performance)
        summary = generate_management_summary(performance, insights)
        report = build_html_report(
            filtered, quality, {}, [],
            dataset_name="retail_business_test.csv",
            scope=f"{len(filtered)} filtered rows; missing exclusion on ({excluded} rows excluded)",
            management_summary=summary,
            performance={**performance, "business_insights": insights},
        )

        self.assertIn(f"{len(filtered)} filtered rows; missing exclusion on ({excluded} rows excluded)", report)
        self.assertIn(f"{filtered['revenue'].sum():,.2f}", report)
        self.assertIn(f"{len(filtered):,}</td>", report)

    def test_aliases_group_breakdowns_and_margin_are_ratio_of_totals(self):
        frame = pd.DataFrame({
            "sales": [100, 100, 0], "costs": [90, 20, 5], "net_profit": [10, 80, -5],
            "quantity": [1, 2, 1], "product": ["A", "A", "B"],
            "category": ["X", "X", "Y"], "territory": ["North", "North", "South"],
            "channel": ["Web", "Store", "Web"],
        })
        result = calculate_business_performance(frame)
        self.assertEqual(result["columns"]["revenue"], "sales")
        self.assertEqual(result["columns"]["cost"], "costs")
        self.assertEqual(result["columns"]["region"], "territory")
        self.assertAlmostEqual(result["metrics"]["profit_margin"], 85 / 200)
        self.assertEqual(set(result["breakdowns"]), {"product", "category", "region", "channel"})
        product_a = result["breakdowns"]["product"].set_index("product").loc["A"]
        self.assertAlmostEqual(product_a["profit_margin"], 90 / 200)

    def test_missing_invalid_zero_negative_and_filtered_metrics_are_safe(self):
        frame = pd.DataFrame({
            "revenue": [0, "bad", 50, None], "cost": [5, 4, 60, 1], "profit": [-5, -4, -10, -1],
            "region": ["A", "A", "B", "B"], "order_date": ["2026-01-01", "bad", "2026-02-01", None],
        })
        selected = apply_filters(frame, category_filters={"region": ["A"]})
        result = calculate_business_performance(selected)
        self.assertEqual(result["columns"]["rows"], 2)
        self.assertEqual(result["metrics"]["revenue"]["missing_or_invalid"], 1)
        self.assertIsNone(result["metrics"]["profit_margin"])
        self.assertEqual(result["metrics"]["profit"]["total"], -9.0)
        self.assertEqual(len(result["time_series"]), 1)
        empty = calculate_business_performance(frame.iloc[:0])
        self.assertEqual(empty["columns"]["rows"], 0)
        self.assertIsNone(empty["metrics"]["revenue"]["total"])
        self.assertTrue(generate_business_insights(empty))

    def test_deduplicate_before_filters_and_insights_summary(self):
        frame = pd.DataFrame({"order_id": ["a", "a", "b"], "revenue": [10, 100, 30], "profit": [2, 20, 3], "region": ["N", "S", "N"]})
        unique = remove_duplicate_identifier_rows(frame, "order_id")
        selected = apply_filters(unique, category_filters={"region": ["S", "N"]})
        performance = calculate_business_performance(selected, identifier="order_id")
        findings = generate_business_insights(performance)
        summary = generate_management_summary(performance, findings)
        self.assertEqual(performance["metrics"]["revenue"]["total"], 40)
        self.assertTrue(findings)
        self.assertIn("2 rows", summary["summary"])
        self.assertEqual(summary["recommendations"], [])

    def test_management_recommendations_are_distinct_and_include_supporting_evidence(self):
        frame = pd.DataFrame({
            "order_id": ["1", "1", "2", "3", "4", "5"],
            "revenue": [100, 100, 80, 30, 20, 10],
            "profit": [30, 30, -10, 1, 2, 3],
            "cost": [70, 70, 90, 29, 18, 7],
            "product": ["A", "A", "B", "C", "D", "E"],
            "region": ["North", "North", "South", "East", "West", "Central"],
        })
        performance = calculate_business_performance(frame)
        insights = generate_business_insights(performance)
        summary = generate_management_summary(performance, insights)
        recommendations = summary["recommendations"]
        recommendation_text = [item.split(" — ", 1)[1] for item in recommendations]

        self.assertGreaterEqual(len(recommendations), 2)
        self.assertLessEqual(len(recommendations), 3)
        self.assertEqual(len(set(recommendation_text)), len(recommendation_text))
        self.assertTrue(any("-10.00" in item or "B" in item for item in recommendation_text))
        self.assertTrue(any("Repeated identifiers" in insight["headline"] for insight in insights))
        # Repeated IDs are a data-quality observation, not automatically an
        # action item or proof that a transaction is invalid.
        self.assertFalse(any("identifier" in item.lower() or "duplicate" in item.lower() for item in recommendation_text))
        self.assertFalse(any("below-median-margin segment to identify possible drivers" in item for item in recommendation_text))

    def test_concentration_requires_a_small_subset_of_more_than_three_known_groups(self):
        frame = pd.DataFrame({
            "sales_channel": ["Online Store", "Retail Store", "Marketplace", None],
            "revenue": [500.0, 300.0, 200.0, 1.0],
            "profit": [200.0, 120.0, 80.0, 0.4],
        })
        performance = calculate_business_performance(frame)
        findings = generate_business_insights(performance)
        self.assertFalse(any("concentrated" in item["headline"].casefold() for item in findings))

    def test_duplicate_policy_is_reflected_in_report_findings_not_duplicate_recommendations(self):
        project_root = Path(__file__).resolve().parents[1]
        source = pd.read_csv(project_root / "retail_business_test.csv")

        included = calculate_business_performance(source, identifier="order_id")
        included_insights = generate_business_insights(included)
        included_summary = generate_management_summary(included, included_insights)
        included_duplicate = next(item for item in included_insights if item["headline"] == "Repeated identifiers observed")
        self.assertEqual(included["columns"]["rows"], 300)
        self.assertAlmostEqual(included["metrics"]["revenue"]["total"], 55282.68)
        self.assertIn("all rows are included", included_duplicate["comparison"].lower())
        self.assertIn("contribute to row-based totals", included_duplicate["comparison"].lower())
        self.assertIn("repetition alone does not establish an invalid transaction", " ".join(included_summary["caveats"]).lower())
        self.assertEqual(len(included_summary["recommendations"]), 3)
        self.assertFalse(any("concentrated" in item["headline"].casefold() for item in included_insights))
        self.assertFalse(any("duplicate" in recommendation.lower() or "identifier" in recommendation.lower()
                             for recommendation in included_summary["recommendations"]))

        first_only = remove_duplicate_identifier_rows(source, "order_id")
        kept = calculate_business_performance(first_only, identifier="order_id")
        # The application records source-level duplicate context after applying
        # keep-first so that the report can explain the rows excluded.
        kept["columns"]["duplicate_rows"] = int(source["order_id"].duplicated(keep="first").sum())
        kept["columns"]["duplicate_ids"] = ["RB-2026-0027"]
        kept["columns"]["duplicate_policy"] = "first identifier occurrence retained"
        kept["columns"]["duplicate_scope"] = "uploaded source before filters"
        kept_insights = generate_business_insights(kept)
        kept_summary = generate_management_summary(kept, kept_insights)
        kept_duplicate = next(item for item in kept_insights if item["headline"] == "Repeated identifiers observed")
        self.assertEqual(kept["columns"]["rows"], 299)
        self.assertAlmostEqual(kept["metrics"]["revenue"]["total"], 54992.40)
        self.assertIn("299", kept_summary["summary"])
        self.assertIn("excluded from the current analysis scope", kept_duplicate["comparison"])
        self.assertIn("were excluded", " ".join(kept_summary["caveats"]).lower())
        self.assertEqual(len(kept_summary["recommendations"]), 3)
        self.assertFalse(any("remove duplicate" in recommendation.lower() or "keep first" in recommendation.lower()
                             for recommendation in kept_summary["recommendations"]))
        self.assertTrue(any("Jan 2026" in recommendation and "Jun 2026" in recommendation
                            for recommendation in kept_summary["recommendations"]))
        self.assertTrue(any("Air Fryer" in recommendation and "Yoga Mat" in recommendation
                            for recommendation in kept_summary["recommendations"]))

    def test_optional_management_report_is_standalone_and_escapes_user_content(self):
        frame = pd.DataFrame({"sales": [10, 20], "profit": [2, 5], "product": ["<A>", "B"]})
        quality, _ = profile_dataset(frame)
        performance = calculate_business_performance(frame)
        findings = generate_business_insights(performance)
        summary = generate_management_summary(performance, findings)
        report = build_html_report(
            frame, quality, {}, [], dataset_name="<unsafe & data>", scope="2 filtered rows",
            management_summary=summary, performance={**performance, "business_insights": findings},
        )
        self.assertIn("&lt;unsafe &amp; data&gt;", report)
        self.assertIn("Management summary", report)
        self.assertIn("Profitability breakdown", report)
        self.assertIn("Duplicate policy: all uploaded rows included", report)
        self.assertIn("profit divided by aggregate revenue", report)
        self.assertNotIn("<A>", report)

    def test_management_report_omits_customer_level_grouped_content(self):
        frame = pd.DataFrame({"revenue": [10], "customer_id": ["customer-sensitive-id"]})
        quality, _ = profile_dataset(frame)
        grouped = pd.DataFrame({"customer_id": ["customer-sensitive-id"], "revenue": [10]})
        report = build_html_report(frame, quality, {}, [], grouped_data=grouped)
        self.assertIn("customer-level grouped findings are omitted", report)
        self.assertIn("Omitted from this shareable report because the selected dimension may identify customers or people", report)
        self.assertNotIn("customer-sensitive-id", report)


class UploadAndExportTests(unittest.TestCase):
    def test_loads_csv_and_rejects_empty_and_unsupported_files(self):
        frame = load_dataset(b"sales,region\n1,North\n", "sample.CSV")
        self.assertEqual(frame.iloc[0]["region"], "North")
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            load_dataset(b"x", "sample.json")
        with self.assertRaisesRegex(ValueError, "uploaded file is empty"):
            load_dataset(b"", "empty.csv")
        with self.assertRaisesRegex(ValueError, "no readable header"):
            load_dataset(b"\n", "blank.csv")
        with self.assertRaisesRegex(ValueError, "no usable column headers"):
            load_dataset(b",\n1,2\n", "blank-headers.csv")

    def test_excel_sheet_selection_and_downloads(self):
        source = BytesIO()
        with pd.ExcelWriter(source, engine="openpyxl") as writer:
            pd.DataFrame({"value": [1]}).to_excel(writer, index=False, sheet_name="First")
            pd.DataFrame({"value": [2]}).to_excel(writer, index=False, sheet_name="Second")
        content = source.getvalue()
        self.assertEqual(get_excel_sheet_names(content, "xlsx"), ["First", "Second"])
        selected = load_dataset(content, "sample.xlsx", "Second")
        self.assertEqual(selected.loc[0, "value"], 2)
        exported = dataframe_to_excel(selected)
        self.assertTrue(exported.startswith(b"PK"))

    def test_excel_export_includes_styled_grouped_results(self):
        exported = dataframe_to_excel(
            pd.DataFrame({"segment": ["A"], "amount": [1234.5]}),
            grouped_data=pd.DataFrame({"segment": ["A"], "amount": [1234.5]}),
        )
        workbook = load_workbook(BytesIO(exported))
        grouped = workbook["Grouped analysis"]

        self.assertEqual(grouped["A1"].alignment.horizontal, "left")
        self.assertEqual(grouped["B1"].alignment.horizontal, "left")
        self.assertEqual(grouped["A2"].alignment.horizontal, "left")
        self.assertEqual(grouped["B2"].alignment.horizontal, "left")
        self.assertEqual(grouped["B2"].number_format, "#,##0.00")
        self.assertEqual(grouped.freeze_panes, "A2")

    def test_html_report_is_portable_and_escapes_dataset_text(self):
        frame = pd.DataFrame({"value": [2], "label": ["<unsafe>"]})
        quality, _ = profile_dataset(frame)
        report = build_html_report(frame, quality, {"value": numeric_summary(frame["value"])}, [], title="Safe <report>")

        self.assertIn("Safe &lt;report&gt;", report)
        self.assertIn("Generated from the current filtered analysis", report)
        self.assertIn("td,th{border:1px solid #d9e2ec;padding:.55rem;text-align:left;vertical-align:top}", report)
        self.assertNotIn('<tr style="text-align: right;">', report)

    def test_html_report_includes_grouped_results_for_the_selected_aggregation(self):
        report = build_html_report(
            pd.DataFrame({"value": [2]}),
            pd.DataFrame(),
            {},
            [],
            grouped_data=pd.DataFrame({"region": ["North"], "value": [2.5]}),
            aggregation="Average",
        )

        self.assertIn("Grouped comparison (Average)", report)
        self.assertIn("North", report)
        self.assertIn("2.50", report)


if __name__ == "__main__":
    unittest.main()
