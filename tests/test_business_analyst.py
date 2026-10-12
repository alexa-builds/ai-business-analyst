"""Synthetic tests for deterministic, dataset-grounded business answers."""

import unittest
from pathlib import Path

import pandas as pd

from analysis import apply_filters, exclude_rows_with_missing_values, remove_duplicate_identifier_rows
from business_analyst import (
    answer_business_question,
    build_question_examples,
    generate_dataset_insights,
    generate_executive_summary,
)


class BusinessQuestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_path = Path(__file__).resolve().parents[1] / "retail_business_test.csv"
        cls.retail_data = pd.read_csv(dataset_path)

    def test_synthetic_dataset_total_includes_duplicate_row_and_explains_difference(self):
        answer = answer_business_question(
            self.retail_data,
            "What is the total revenue across all transactions?",
            ["revenue"],
            ["product"],
            source_frame=self.retail_data,
        )

        self.assertEqual(answer["intent"], "summary_metric")
        self.assertEqual(answer["facts"][0]["value"], "55,282.68")
        self.assertIn("RB-2026-0027", answer["calculation"])
        self.assertIn("included because removal was not requested", answer["calculation"])

    def test_synthetic_dataset_duplicate_id_count_names_repeated_identifier(self):
        filtered = self.retail_data[self.retail_data["region"].eq("West")]
        answer = answer_business_question(
            filtered,
            "How many duplicate order IDs are in the dataset?",
            source_frame=self.retail_data,
        )

        self.assertEqual(answer["intent"], "duplicate_id_count")
        self.assertEqual(answer["facts"][0]["value"], 1)
        self.assertEqual(answer["facts"][1]["value"], 1)
        self.assertIn("RB-2026-0027 (2 rows)", answer["headline"])
        self.assertIn("uploaded dataset", answer["calculation"])

    def test_synthetic_dataset_total_after_first_occurrence_dedup_is_verified(self):
        answer = answer_business_question(
            self.retail_data,
            "What is the total revenue after removing duplicate order IDs, keeping the first occurrence?",
            ["revenue"],
            ["product"],
            source_frame=self.retail_data,
        )

        self.assertEqual(answer["intent"], "deduplicated_total")
        self.assertEqual(answer["facts"][0]["value"], "54,992.40")
        self.assertEqual(answer["facts"][1]["value"], 1)
        self.assertIn("RB-2026-0027", answer["calculation"])
        self.assertIn("keeping the first", answer["calculation"])

    def test_deduplicated_total_deduplicates_source_before_applying_filters(self):
        source = self.retail_data
        filtered = apply_filters(source, category_filters={"product": ["Smart Watch"]})
        expected = apply_filters(
            source.drop_duplicates(subset=["order_id"], keep="first"),
            category_filters={"product": ["Smart Watch"]},
        )["revenue"].sum()

        answer = answer_business_question(
            filtered,
            "What is the total revenue after removing duplicate order IDs, keeping the first occurrence?",
            ["revenue"],
            ["product"],
            source_frame=source,
            category_filters={"product": ["Smart Watch"]},
        )

        self.assertEqual(answer["facts"][0]["value"], f"{expected:,.2f}")
        self.assertIn("current filters", answer["calculation"])

    def test_deduplicated_question_uses_same_missing_value_rule_as_analysis(self):
        after_first_id = remove_duplicate_identifier_rows(self.retail_data, "order_id")
        expected, excluded = exclude_rows_with_missing_values(after_first_id)
        current_frame, _ = exclude_rows_with_missing_values(self.retail_data)
        answer = answer_business_question(
            current_frame,
            "What is the total revenue after removing duplicate order IDs, keeping the first occurrence?",
            ["revenue"],
            source_frame=self.retail_data,
            exclude_missing_values=True,
        )

        self.assertEqual(answer["status"], "answered")
        self.assertEqual(answer["facts"][0]["value"], f"{expected['revenue'].sum():,.2f}")
        self.assertIn(f"excluded {excluded:,} row(s) under the current missing-value rule", answer["calculation"])

    def test_highest_month_uses_dynamic_columns_and_current_filters(self):
        frame = pd.DataFrame(
            {
                "period_name": ["January", "January", "February", "March", "February"],
                "market_area": ["East", "West", "East", "East", "East"],
                "gross_margin": [3, 90, 8, 2, 7],
            }
        )
        filtered = apply_filters(frame, category_filters={"market_area": ["East"]})

        result = answer_business_question(
            filtered,
            "Which period_name had the highest gross_margin?",
            selected_measures=["gross_margin"],
            aggregation="Sum",
        )

        self.assertEqual(result["status"], "answered")
        self.assertIn("February", result["headline"])
        self.assertIn("15.00", result["headline"])
        self.assertEqual(result["facts"][0]["value"], "15.00")

    def test_overall_total_phrases_sum_rows_instead_of_ranking_months(self):
        frame = pd.DataFrame({
            "month": ["January", "February", "March", "April", "May", "June"],
            "sales": [1200, 1500, 1100, 1800, 2200, 1800],
        })

        for question in (
            "What were the total sales across all months?",
            "What were total sales overall?",
            "What is the sum of all sales?",
        ):
            with self.subTest(question=question):
                answer = answer_business_question(frame, question, ["sales"], ["month"], "Average")
                self.assertEqual(answer["intent"], "summary_metric")
                self.assertEqual(answer["facts"][0]["value"], "9,600.00")
                self.assertIn("current analysis rows", answer["calculation"])

    def test_total_respects_active_filter_and_discloses_scope(self):
        frame = pd.DataFrame({
            "month": ["January", "February", "March", "April", "May", "June"],
            "region": ["East", "West", "East", "West", "East", "West"],
            "sales": [1200, 1500, 1100, 1800, 2200, 1800],
        })
        filtered = apply_filters(frame, category_filters={"region": ["East"]})

        answer = answer_business_question(filtered, "What were the total sales across all months?", ["sales"], ["month"])

        self.assertEqual(answer["facts"][0]["value"], "4,500.00")
        self.assertIn("active filters", answer["calculation"])
        self.assertTrue(any("filters restrict its scope" in note for note in answer["limitations"]))

    def test_retail_intent_examples_cover_total_breakdown_rank_and_average(self):
        frame = pd.DataFrame({
            "product": ["A", "A", "B", "C"],
            "region": ["North", "South", "North", "South"],
            "revenue": [100, 200, 400, 50],
            "profit": [20, 50, 120, 10],
        })
        measures = ["revenue", "profit"]
        dimensions = ["product", "region"]

        total_revenue = answer_business_question(frame, "What is the total revenue across all transactions?", measures, dimensions)
        total_profit = answer_business_question(frame, "What is the total profit across all transactions?", measures, dimensions)
        breakdown = answer_business_question(frame, "What is total revenue by product?", measures, dimensions)
        top_product = answer_business_question(frame, "Which product has the highest revenue?", measures, dimensions)
        top_region = answer_business_question(frame, "Which region generates the most profit?", measures, dimensions)
        average = answer_business_question(frame, "What is the average revenue per transaction?", measures, dimensions)

        self.assertEqual((total_revenue["intent"], total_revenue["facts"][0]["value"]), ("summary_metric", "750.00"))
        self.assertEqual((total_profit["intent"], total_profit["facts"][0]["value"]), ("summary_metric", "200.00"))
        self.assertEqual(breakdown["intent"], "group_comparison")
        self.assertEqual([(item["label"], item["value"]) for item in breakdown["facts"]], [("product: B", "400.00"), ("product: A", "300.00"), ("product: C", "50.00")])
        self.assertIn("product: B", top_product["headline"])
        self.assertIn("400.00", top_product["headline"])
        self.assertIn("region: North", top_region["headline"])
        self.assertIn("140.00", top_region["headline"])
        self.assertEqual((average["intent"], average["facts"][0]["value"]), ("summary_metric", "187.50"))

        transaction_rows = pd.DataFrame({"transaction": ["T1", "T2"], "revenue": [100, 300]})
        average_per_transaction = answer_business_question(
            transaction_rows,
            "What is the average revenue per transaction?",
            ["revenue"],
            ["transaction"],
        )
        self.assertEqual(average_per_transaction["intent"], "summary_metric")
        self.assertEqual(average_per_transaction["facts"][0]["value"], "200.00")

    def test_highest_and_lowest_month_rankings_are_not_overall_totals(self):
        frame = pd.DataFrame({
            "month": ["January", "February", "March", "April", "May", "June"],
            "sales": [1200, 1500, 1100, 1800, 2200, 1800],
        })

        highest = answer_business_question(frame, "Which month had the highest sales?", ["sales"], ["month"])
        lowest = answer_business_question(frame, "Which month had the lowest sales?", ["sales"], ["month"])

        self.assertEqual(highest["intent"], "group_ranking")
        self.assertIn("month: May", highest["headline"])
        self.assertIn("2,200.00", highest["headline"])
        self.assertEqual(lowest["intent"], "group_ranking")
        self.assertIn("month: March", lowest["headline"])
        self.assertIn("1,100.00", lowest["headline"])

    def test_explicit_per_month_total_remains_a_grouped_breakdown(self):
        frame = pd.DataFrame({"month": ["January", "February"], "sales": [1200, 1500]})

        answer = answer_business_question(frame, "Show total sales by month", ["sales"], ["month"])

        self.assertEqual(answer["intent"], "group_comparison")
        self.assertIn("2 groups", answer["headline"])

    def test_category_contribution_uses_sum_even_if_display_setting_is_mean(self):
        frame = pd.DataFrame({"team": ["A", "A", "B"], "net_value": [6, 6, 10]})
        result = answer_business_question(
            frame,
            "Which category contributes the most net_value?",
            selected_measures=["net_value"],
            selected_dimensions=["team"],
            aggregation="Average",
        )

        self.assertIn("team: A", result["headline"])
        self.assertIn("12.00", result["headline"])
        self.assertIn("grouped sum", result["headline"])

    def test_average_vs_highest_returns_calculated_comparison(self):
        result = answer_business_question(
            pd.DataFrame({"service_cost": [10, 20, 30]}),
            "How does the average compare with the highest value?",
            selected_measures=["service_cost"],
        )

        self.assertEqual(result["intent"], "average_vs_highest")
        self.assertEqual([fact["value"] for fact in result["facts"][:3]], ["20.00", "30.00", "10.00"])

    def test_month_trend_handles_zero_baseline_safely(self):
        result = answer_business_question(
            pd.DataFrame({"month_label": ["January", "February", "March"], "units": [0, 8, 12]}),
            "What are the biggest trends in units over time?",
            selected_measures=["units"],
            selected_dimensions=["month_label"],
        )

        self.assertEqual(result["intent"], "trend")
        self.assertEqual(result["facts"][-1]["value"], "Unavailable (zero baseline)")
        self.assertTrue(any("do not include years" in item.lower() for item in result["limitations"]))

    def test_date_ranking_and_percentage_change_use_parsed_date_buckets(self):
        frame = pd.DataFrame({"booked_on": ["2025-01-03", "2025-02-01"], "amount_usd": [0, 30]})
        highest = answer_business_question(
            frame,
            "Which month had the highest amount_usd?",
            selected_measures=["amount_usd"],
            date_column="booked_on",
            date_granularity="Month",
        )
        trend = answer_business_question(
            frame,
            "What is the percentage change over time?",
            selected_measures=["amount_usd"],
            date_column="booked_on",
        )

        self.assertEqual(highest["intent"], "time_ranking")
        self.assertIn("2025-02-01", highest["headline"])
        self.assertEqual(trend["facts"][-1]["value"], "Unavailable (zero baseline)")

    def test_group_comparison_respects_selected_aggregation_and_dimensions(self):
        frame = pd.DataFrame({"division": ["East", "East", "West", "West"], "orders": [2, 6, 9, 11]})
        result = answer_business_question(
            frame,
            "How does orders compare across division?",
            selected_measures=["orders"],
            selected_dimensions=["division"],
            aggregation="Average",
        )

        self.assertEqual(result["intent"], "group_comparison")
        self.assertIn("Average orders ranges from 4.00 to 10.00", result["headline"])
        self.assertEqual(result["facts"][0]["label"], "division: West")

    def test_quality_answer_reports_missing_invalid_duplicate_and_outlier_evidence(self):
        frame = pd.DataFrame({"team": ["A", "A", "B", "C", "D", "E"], "amount": [1, 1, None, "bad", 2, 100]})
        result = answer_business_question(frame, "Are there unusual values or data-quality concerns?", ["amount"])
        text = " ".join([result["headline"], result["calculation"]] + [fact["label"] + fact["value"] for fact in result["facts"]])

        self.assertEqual(result["intent"], "quality")
        self.assertIn("missing", text)
        self.assertIn("nonnumeric", text)
        self.assertIn("Duplicate rows", text)
        self.assertIn("IQR", text)

    def test_correlation_is_labeled_as_association_not_causation(self):
        result = answer_business_question(
            pd.DataFrame({"ad_spend": [1, 2, 3], "orders": [2, 4, 6]}),
            "What is the correlation between ad_spend and orders?",
        )

        self.assertIn("correlation", result["headline"])
        self.assertTrue(any("does not show" in item for item in result["limitations"]))

    def test_empty_unsupported_and_insufficient_questions_do_not_invent_facts(self):
        empty = answer_business_question(pd.DataFrame({"amount": []}), "What is the highest amount?")
        unsupported = answer_business_question(pd.DataFrame({"amount": [2, 4]}), "Will this business beat its competitors?")
        no_numeric = answer_business_question(pd.DataFrame({"category": ["A", "B"]}), "What is the average?")

        self.assertEqual(empty["status"], "insufficient")
        self.assertEqual(unsupported["status"], "unsupported")
        self.assertEqual(unsupported["facts"], [])
        self.assertEqual(no_numeric["status"], "insufficient")

    def test_question_examples_adapt_to_arbitrary_columns(self):
        examples = build_question_examples(
            pd.DataFrame({"week_name": ["January", "February"], "active_users": [9, 12]}),
            ["active_users"],
            ["week_name"],
        )

        self.assertIn("Which week_name has the highest active_users?", examples)
        self.assertTrue(any("active_users" in example for example in examples))


class InsightGenerationTests(unittest.TestCase):
    def test_executive_summary_has_python_figures_and_filtered_context(self):
        frame = pd.DataFrame({"area": ["A", "A", "B"], "value": [2, 4, 8]})
        summary = generate_executive_summary(frame, ["value"], ["area"], "Sum")

        self.assertEqual(summary["status"], "answered")
        self.assertIn("3 rows", summary["headline"])
        self.assertIn({"label": "value total", "value": "14.00"}, summary["facts"])
        findings = generate_dataset_insights(frame, ["value"], ["area"], "Sum")
        self.assertTrue(any("Highest grouped sum" in finding["text"] for finding in findings))

    def test_insights_report_insufficient_data_without_forced_performance_claim(self):
        findings = generate_dataset_insights(pd.DataFrame({"label": ["a", "b"]}))

        self.assertTrue(findings)
        self.assertTrue(any("not contain enough numeric" in finding["text"] for finding in findings))

    def test_concentration_finding_includes_calculated_groups_and_share(self):
        frame = pd.DataFrame({"segment": ["A", "B", "C", "D"], "amount": [50, 30, 15, 5]})
        findings = generate_dataset_insights(frame, ["amount"], ["segment"])
        concentration = next(item for item in findings if "concentration flag" in item["text"])

        self.assertIn("95.0%", concentration["text"])
        self.assertIn("segment: A", concentration["text"])
        self.assertEqual(concentration["evidence"][0]["value"], "95.0%")

    def test_empty_executive_summary_is_safe(self):
        summary = generate_executive_summary(pd.DataFrame(columns=["measure"]))

        self.assertEqual(summary["status"], "insufficient")


if __name__ == "__main__":
    unittest.main()
