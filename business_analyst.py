"""Deterministic business question answering and data-backed findings."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from analysis import (
    aggregate_data,
    apply_filters,
    build_business_insights,
    coerce_numeric,
    numeric_summary,
    profile_dataset,
)


def _column_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _mentioned_columns(question: str, columns: list[Any]) -> list[Any]:
    normalized = f" {_column_key(question)} "
    matches = []
    for column in columns:
        key = _column_key(column)
        if key and f" {key} " in normalized:
            matches.append(column)
    return sorted(matches, key=lambda column: len(_column_key(column)), reverse=True)


def _profiles(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return profile_dataset(frame)[1]


def _available_columns(frame: pd.DataFrame) -> tuple[list[Any], list[Any], list[Any]]:
    profiles = _profiles(frame)
    measures = [column for column in frame.columns if profiles[str(column)]["is_numeric"]]
    dates = [column for column in frame.columns if profiles[str(column)]["is_date"] and profiles[str(column)]["valid_date_count"]]
    dimensions = [column for column in frame.columns if not profiles[str(column)]["is_numeric"] and not profiles[str(column)]["is_date"]]
    return measures, dimensions, dates


def _choose_measure(question: str, frame: pd.DataFrame, selected_measures: list[Any] | None) -> Any | None:
    measures, _, _ = _available_columns(frame)
    mentioned = _mentioned_columns(question, measures)
    if mentioned:
        return mentioned[0]
    if selected_measures:
        return next((measure for measure in selected_measures if measure in measures), None)
    return measures[0] if measures else None


def _choose_dimensions(question: str, frame: pd.DataFrame, selected_dimensions: list[Any] | None) -> list[Any]:
    _, dimensions, _ = _available_columns(frame)
    mentioned = _mentioned_columns(question, dimensions)
    if mentioned:
        return mentioned
    if selected_dimensions:
        return [column for column in selected_dimensions if column in dimensions]
    return dimensions[:1]


def _identifier_columns(frame: pd.DataFrame) -> list[Any]:
    """Return columns whose names indicate identifiers or keys."""
    candidates = []
    for column in frame.columns:
        key = _column_key(column)
        if re.search(r"\b(?:id|identifier|key)\b", key):
            candidates.append(column)
    return sorted(
        candidates,
        key=lambda column: (0 if re.search(r"\border\s+(?:id|identifier)\b", _column_key(column)) else 1, str(column).casefold()),
    )


def _choose_identifier_column(frame: pd.DataFrame, question: str) -> Any | None:
    candidates = _identifier_columns(frame)
    named = _mentioned_columns(question, candidates)
    if named:
        return named[0]
    if re.search(r"\border\s+ids?\b", _column_key(question)):
        order_columns = [column for column in candidates if re.search(r"\border\s+(?:id|identifier)\b", _column_key(column))]
        if order_columns:
            return order_columns[0]
    return candidates[0] if candidates else None


def _duplicate_identifier_counts(frame: pd.DataFrame, column: Any) -> pd.Series:
    values = frame[column]
    valid = values.notna()
    if values.dtype == object or pd.api.types.is_string_dtype(values.dtype):
        valid &= ~values.map(lambda value: isinstance(value, str) and not value.strip())
    counts = values.loc[valid].value_counts(sort=False, dropna=True)
    return counts[counts > 1]


def _remove_repeated_identifiers(frame: pd.DataFrame, column: Any) -> tuple[pd.DataFrame, pd.Series]:
    values = frame[column]
    valid = values.notna()
    if values.dtype == object or pd.api.types.is_string_dtype(values.dtype):
        valid &= ~values.map(lambda value: isinstance(value, str) and not value.strip())
    duplicated = pd.Series(False, index=frame.index)
    duplicated.loc[valid] = values.loc[valid].duplicated(keep="first")
    return frame.loc[~duplicated].copy(), _duplicate_identifier_counts(frame, column)


def _duplicate_scope_is_explicitly_filtered(question: str) -> bool:
    return bool(re.search(r"\b(?:filtered|filter|visible|selected|current)\b", _column_key(question)))


def _answer_duplicate_id_count(frame: pd.DataFrame, question: str, identifier: Any | None, scope: str) -> dict[str, Any]:
    if identifier is None:
        return _result(
            "insufficient", "duplicate_id_count",
            "I could not identify an ID/key column to check for repeated identifiers.",
            limitations=["Use a column named with an ID, identifier, or key, or specify the identifier column in the question."],
            question=question,
        )
    duplicate_counts = _duplicate_identifier_counts(frame, identifier)
    duplicate_ids = [f"{value} ({int(count):,} rows)" for value, count in duplicate_counts.items()]
    extra_rows = int((duplicate_counts - 1).sum()) if not duplicate_counts.empty else 0
    if duplicate_ids:
        shown_ids = duplicate_ids[:8]
        remaining = len(duplicate_ids) - len(shown_ids)
        id_summary = "; ".join(shown_ids) + (f"; and {remaining:,} more" if remaining else "")
        headline = f"Found {len(duplicate_counts):,} duplicate {identifier!s} value(s) in {scope}: {id_summary}."
    else:
        headline = f"No duplicate {identifier!s} values were found in {scope}."
    return _result(
        "answered", "duplicate_id_count", headline,
        facts=[
            {"label": "Duplicate ID values", "value": len(duplicate_counts)},
            {"label": "Extra duplicate rows", "value": extra_rows},
            {"label": "Repeated identifiers", "value": "; ".join(duplicate_ids[:8]) if duplicate_ids else "None"},
        ],
        calculation=f"Counted nonmissing {identifier!s} values occurring more than once. The count is based on {scope}.",
        limitations=["Missing or blank identifier values are not treated as duplicate IDs."],
        question=question,
    )


def _answer_total_after_deduplication(
    frame: pd.DataFrame,
    question: str,
    selected_measures: list[Any] | None,
    selected_dimensions: list[Any] | None,
    source_frame: pd.DataFrame | None,
    category_filters: dict | None,
    numeric_ranges: dict | None,
    date_ranges: dict | None,
    exclude_missing_values: bool = False,
) -> dict[str, Any]:
    source = source_frame if source_frame is not None else frame
    identifier = _choose_identifier_column(source, question)
    if identifier is None:
        return _answer_duplicate_id_count(source, question, None, "the current analysis rows")
    deduplicated, duplicate_counts = _remove_repeated_identifiers(source, identifier)
    rows_removed = int((duplicate_counts - 1).sum()) if not duplicate_counts.empty else 0
    if exclude_missing_values:
        from analysis import exclude_rows_with_missing_values

        deduplicated, missing_removed = exclude_rows_with_missing_values(deduplicated)
    else:
        missing_removed = 0
    if source_frame is not None:
        deduplicated = apply_filters(deduplicated, category_filters, numeric_ranges, date_ranges)
        scope = (
            "the deduplicated uploaded data under the missing-value rule after applying the current filters"
            if len(frame) < len(source_frame)
            else "the uploaded dataset after keeping the first occurrence of each identifier and applying the missing-value rule"
        )
    else:
        scope = "the current analysis rows after keeping the first occurrence of each identifier"
    measure = _choose_measure(question, source, selected_measures)
    if measure is None:
        return _result(
            "insufficient", "deduplicated_total",
            "I could not identify a numeric measure to total after duplicate removal.",
            limitations=["Name a numeric measure, such as a revenue or amount column."],
            question=question,
        )
    summary = numeric_summary(deduplicated[measure].tolist())
    value = summary["sum"]
    if value is None:
        return _result("insufficient", "deduplicated_total", f"There are no valid {measure!s} values in {scope}.", question=question)
    formatted = _number(value)
    duplicate_id_facts = [
        f"{value!s} ({int(count):,} rows)" for value, count in duplicate_counts.items()
    ]
    result = _result(
        "answered", "deduplicated_total",
        f"Total {measure!s} after keeping the first row for each {identifier!s} is {formatted}.",
        facts=[
            {"label": f"Deduplicated total {measure!s}", "value": formatted},
            {"label": "Duplicate rows removed", "value": rows_removed},
            {"label": "Rows included", "value": summary["count"]},
        ],
        calculation=(
            f"Kept the first row for each nonmissing {identifier!s}, removing {rows_removed:,} later repeated row(s); "
            f"excluded {missing_removed:,} row(s) under the current missing-value rule. "
            f"Summed {summary['count']:,} valid numeric {measure!s} observations from {scope}."
        ),
        limitations=["This is a deduplicated analysis view; the uploaded source data is not modified."],
        follow_ups=[f"How many duplicate {identifier!s} values are in the dataset?"],
        question=question,
    )
    if duplicate_id_facts:
        result["calculation"] += " Repeated identifiers removed: " + "; ".join(duplicate_id_facts[:8]) + "."
    return result


def _number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value):,.2f}"


def _follow_ups(frame: pd.DataFrame, measure: Any | None, dimensions: list[Any]) -> list[str]:
    measures, category_columns, date_columns = _available_columns(frame)
    suggestions = []
    if measure is not None and dimensions:
        dimension = dimensions[0]
        suggestions.extend([
            f"How does {measure} compare across {dimension}?",
            f"Which {dimension} has the lowest {measure}?",
        ])
    elif measure is not None and category_columns:
        suggestions.append(f"Which {category_columns[0]} contributes the most {measure}?")
    if measure is not None and (date_columns or any(_profiles(frame)[str(column)]["is_month"] for column in category_columns)):
        suggestions.append(f"What are the biggest trends in {measure} over time?")
    if measures:
        suggestions.append("Are there unusual values or data-quality concerns?")
    return suggestions[:3]


def build_question_examples(
    frame: pd.DataFrame,
    selected_measures: list[Any] | None = None,
    selected_dimensions: list[Any] | None = None,
) -> list[str]:
    """Make useful question examples from columns present in the current data."""
    measures, dimensions, dates = _available_columns(frame)
    measure = next((item for item in (selected_measures or []) if item in measures), measures[0] if measures else None)
    dimension = next((item for item in (selected_dimensions or []) if item in dimensions), dimensions[0] if dimensions else None)
    examples = []
    if measure is not None and dimension is not None:
        examples.extend([
            f"Which {dimension} has the highest {measure}?",
            f"Which {dimension} contributes the most {measure}?",
        ])
    if measure is not None:
        examples.append(f"How does the average compare with the highest {measure} value?")
        if dates or (dimension is not None and _profiles(frame)[str(dimension)]["is_month"]):
            examples.append(f"What are the biggest trends in {measure} over time?")
    examples.extend([
        "Are there any unusual values or data-quality concerns?",
        "Summarize the main findings for a management meeting.",
    ])
    return list(dict.fromkeys(examples))


def _result(
    status: str,
    intent: str,
    headline: str,
    facts: list[dict[str, Any]] | None = None,
    calculation: str = "",
    limitations: list[str] | None = None,
    follow_ups: list[str] | None = None,
    question: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "intent": intent,
        "headline": headline,
        "facts": facts or [],
        "calculation": calculation,
        "limitations": limitations or [],
        "follow_ups": follow_ups or [],
        "question": question,
        "model_interpretation": None,
        "model_error": None,
    }


def _trend_result(
    frame: pd.DataFrame,
    measure: Any,
    selected_dimensions: list[Any],
    aggregation: str,
    date_column: Any | None,
    date_granularity: str,
    question: str,
) -> dict[str, Any]:
    if date_column is not None:
        trend = aggregate_data(frame, [measure], [], aggregation, date_column, date_granularity)
        if trend.empty or "Date" not in trend or trend[measure].notna().sum() < 2:
            return _result("insufficient", "trend", "There is not enough valid time data to compare periods.", calculation="At least two distinct time buckets with numeric observations are required.", question=question)
        trend = trend.dropna(subset=[measure]).sort_values("Date")
        first, last = trend.iloc[0], trend.iloc[-1]
        first_label, last_label = first["Date"].strftime("%Y-%m-%d"), last["Date"].strftime("%Y-%m-%d")
        temporal_note = f"Date values were parsed from {date_column!s} and grouped by {date_granularity.lower()}."
    else:
        profiles = _profiles(frame)
        month_dimension = next((column for column in selected_dimensions if profiles[str(column)]["is_month"]), None)
        if month_dimension is None:
            return _result("insufficient", "trend", "A time trend cannot be established from the available columns.", calculation="No valid date column or selected month-name dimension is available.", limitations=["Add a date/time column or month labels to support time-based comparisons."], question=question)
        from analysis import month_number

        work = pd.DataFrame({
            "_order": frame[month_dimension].map(month_number),
            "_value": coerce_numeric(frame[measure]),
        }).dropna()
        reducer = {"Sum": "sum", "Average": "mean", "Median": "median", "Minimum": "min", "Maximum": "max", "Count": "count"}.get(aggregation, "sum")
        monthly = work.groupby("_order", sort=True)["_value"].agg(reducer)
        if len(monthly) < 2:
            return _result("insufficient", "trend", "There are not enough distinct month categories to compare.", calculation="At least two month categories with numeric observations are required.", question=question)
        from calendar import month_name

        first_label, last_label = month_name[int(monthly.index[0])], month_name[int(monthly.index[-1])]
        first_value, last_value = float(monthly.iloc[0]), float(monthly.iloc[-1])
        temporal_note = f"Month labels in {month_dimension!s} were sorted in calendar order."
        return _change_result(question, "trend", measure, first_label, last_label, first_value, last_value, aggregation, temporal_note, month_labels=True)

    return _change_result(question, "trend", measure, first_label, last_label, float(first[measure]), float(last[measure]), aggregation, temporal_note)


def _change_result(question, intent, measure, first_label, last_label, first_value, last_value, aggregation, temporal_note, month_labels=False):
    change = last_value - first_value
    if first_value == 0:
        percent = None
        percent_text = "Percentage change is unavailable because the starting value is zero."
    else:
        percent = change / abs(first_value) * 100
        percent_text = f"Percentage change: {percent:+.1f}%."
    return _result(
        "answered",
        intent,
        f"{aggregation} {measure!s} changed by {_number(change)} from {first_label} to {last_label}.",
        facts=[
            {"label": f"{first_label} {aggregation.lower()}", "value": _number(first_value)},
            {"label": f"{last_label} {aggregation.lower()}", "value": _number(last_value)},
            {"label": "Absolute change", "value": _number(change)},
            {"label": "Percentage change", "value": f"{percent:+.1f}%" if percent is not None else "Unavailable (zero baseline)"},
        ],
        calculation=f"Compared the first and last available periods using {aggregation.lower()} of {measure!s}. {percent_text} {temporal_note}",
        limitations=["This comparison describes the observed endpoints; it does not establish a cause or guarantee the intervening trend."] + (["Month labels do not include years, so repeated month names are combined into one calendar-month cycle."] if month_labels else []),
        follow_ups=[f"Which {measure!s} periods had the highest and lowest values?", f"How does {aggregation.lower()} {measure!s} vary across the full time range?"],
        question=question,
    )


def answer_business_question(
    frame: pd.DataFrame,
    question: str,
    selected_measures: list[Any] | None = None,
    selected_dimensions: list[Any] | None = None,
    aggregation: str = "Sum",
    date_column: Any | None = None,
    date_granularity: str = "Month",
    source_frame: pd.DataFrame | None = None,
    category_filters: dict | None = None,
    numeric_ranges: dict | None = None,
    date_ranges: dict | None = None,
    exclude_missing_values: bool = False,
) -> dict[str, Any]:
    """Answer a supported question using current rows and optional uploaded-source context."""
    question = str(question or "").strip()
    if not question:
        return _result("unsupported", "unknown", "Enter a question about the current dataset.")

    lowered = _column_key(question)
    mentions = lambda *terms: any(term in lowered for term in terms)
    asks_about_duplicate_ids = bool(re.search(r"\b(?:duplicate|duplicates|duplicated|deduplicate|deduplicated)\b", lowered))
    asks_to_remove_duplicates = asks_about_duplicate_ids and (
        mentions("remove", "removing", "drop", "dedup", "deduplicate")
        and bool(re.search(r"\b(?:first|keep|keeping|retain|retaining)\b|first occurrence", lowered))
    )
    if asks_to_remove_duplicates:
        return _answer_total_after_deduplication(
            frame, question, selected_measures, selected_dimensions, source_frame,
            category_filters, numeric_ranges, date_ranges, exclude_missing_values,
        )

    asks_duplicate_count = (
        asks_about_duplicate_ids
        and bool(re.search(r"\b(?:id|ids|identifier|identifiers|key|keys)\b", lowered))
        and bool(re.search(r"\b(?:how many|count|number of|identify|which|are there)\b", lowered))
    )
    if asks_duplicate_count:
        use_filtered_rows = _duplicate_scope_is_explicitly_filtered(question) or source_frame is None
        duplicate_frame = frame if use_filtered_rows else source_frame
        scope = "the current filtered rows" if _duplicate_scope_is_explicitly_filtered(question) else (
            "the uploaded dataset" if source_frame is not None else "the dataset rows supplied to analysis"
        )
        return _answer_duplicate_id_count(
            duplicate_frame,
            question,
            _choose_identifier_column(duplicate_frame, question),
            scope,
        )

    if frame.empty:
        return _result("insufficient", "unknown", "The current filters leave no rows to analyze.", limitations=["Clear or adjust filters and ask the question again."], question=question)

    measures, dimensions, dates = _available_columns(frame)
    if mentions("summary", "summarize", "management meeting", "main findings", "overview"):
        return generate_executive_summary(frame, selected_measures or [], selected_dimensions or [], aggregation, date_column, date_granularity, question)
    if mentions("unusual", "outlier", "anomal", "data quality", "missing", "invalid", "duplicate"):
        return _answer_quality(frame, None, question, [])
    if not measures:
        return _result("insufficient", "unknown", "This dataset has no numeric measure to calculate that answer.", limitations=["Add a column containing numeric values for measure questions. Data-quality checks and the dataset summary remain available."], question=question)

    measure = _choose_measure(question, frame, selected_measures)
    chosen_dimensions = _choose_dimensions(question, frame, selected_dimensions)
    if measure is None:
        return _result("insufficient", "unknown", "I could not identify a numeric measure for this question.", question=question)
    follow_ups = _follow_ups(frame, measure, chosen_dimensions)
    if mentions("correlation", "relationship", "related", "association"):
        named = _mentioned_columns(question, measures)
        pair = named[:2] if len(named) >= 2 else [item for item in (selected_measures or []) if item in measures]
        if len(pair) < 2:
            pair = measures[:2]
        return _answer_correlation(frame, pair, question, follow_ups)
    temporal_rank = mentions("month", "period", "date", "quarter", "year", "week") and mentions("highest", "most", "top", "largest", "lowest", "least", "smallest", "maximum", "minimum")
    if temporal_rank and dates:
        selected_date = date_column if date_column in dates else dates[0]
        return _answer_time_ranking(frame, measure, selected_date, date_granularity, aggregation, mentions("lowest", "least", "smallest", "minimum"), question, follow_ups)
    if mentions("trend", "over time", "change", "growth", "increased", "decreased", "month by month", "percentage change", "percent change"):
        selected_date = date_column if date_column in dates else (dates[0] if dates else None)
        return _trend_result(frame, measure, chosen_dimensions, aggregation, selected_date, date_granularity, question)
    if mentions("average", "mean") and mentions("highest", "maximum", "max", "largest"):
        return _answer_average_vs_highest(frame, measure, question, follow_ups)
    if mentions("average", "mean") and re.search(r"\bper\s+transactions?\b", lowered):
        return _answer_summary_metric(frame, measure, lowered, question, follow_ups)

    wants_low = mentions("lowest", "least", "smallest", "minimum", "worst")
    wants_high = mentions("highest", "most", "top", "best", "largest")
    has_ranking_intent = wants_high or wants_low
    asks_for_total = mentions("total", "sum", "summation")
    asks_for_grouping = bool(re.search(r"\b(?:per|each|by)\b|for each|grouped by|broken down by", lowered))
    # Explicit ranking/comparison intent takes precedence. Otherwise total/sum
    # means one overall metric unless the user requests a per-category breakdown.
    if asks_for_total and not has_ranking_intent and not asks_for_grouping:
        result = _answer_summary_metric(frame, measure, lowered, question, follow_ups)
        if result["status"] == "answered":
            result["calculation"] += (
                " This total covers the current analysis rows: all dataset rows when no app filters are active, "
                "or only rows matching active filters."
            )
            result["limitations"].append(
                "The total reflects the rows currently supplied to analysis; active app filters restrict its scope."
            )
            for identifier in _identifier_columns(frame):
                duplicate_counts = _duplicate_identifier_counts(frame, identifier)
                if not duplicate_counts.empty:
                    duplicate_ids = "; ".join(
                        f"{value} ({int(count):,} rows)" for value, count in duplicate_counts.items()
                    )
                    result["calculation"] += (
                        f" Repeated {identifier!s} values are included because removal was not requested: {duplicate_ids}."
                    )
                    break
        return result

    explicitly_named_dimension = bool(_mentioned_columns(question, dimensions))
    names_a_dimension = explicitly_named_dimension or mentions("category", "categories", "group", "groups")
    requests_grouping = bool(re.search(r"\b(?:by|per|each|across|between|compare|comparison|breakdown|grouped)\b|for each|broken down", lowered))
    # A selected dimension alone is not a grouping request. Group when the
    # question names a dimension and asks for a rank, comparison, or breakdown.
    wants_group = bool(chosen_dimensions) and names_a_dimension and (has_ranking_intent or requests_grouping)
    if wants_group and wants_high and wants_low:
        return _answer_group_comparison(frame, measure, chosen_dimensions, aggregation, question, follow_ups)
    if wants_group and mentions("highest", "most", "top", "best", "largest", "contribute", "contributes", "contribution", "lowest", "least", "smallest", "worst"):
        group_aggregation = "Sum" if mentions("contribute", "contributes", "contribution") else aggregation
        return _answer_group_ranking(frame, measure, chosen_dimensions, group_aggregation, wants_low, question, follow_ups)
    if wants_group and mentions("compare", "comparison", "across", "breakdown", "by", "total", "sum", "average", "mean", "count", "between", "difference", "differences"):
        group_aggregation = "Sum" if mentions("total", "sum") else "Average" if mentions("average", "mean") else "Count" if mentions("count") else aggregation
        return _answer_group_comparison(frame, measure, chosen_dimensions, group_aggregation, question, follow_ups)
    if mentions("highest", "maximum", "max", "largest", "lowest", "minimum", "smallest"):
        return _answer_extreme(frame, measure, wants_low, question, follow_ups)
    if mentions("average", "mean", "total", "sum", "median", "count", "how many"):
        return _answer_summary_metric(frame, measure, lowered, question, follow_ups)

    return _result(
        "unsupported",
        "unknown",
        "I can’t answer that question reliably with the current analytical rules.",
        calculation="No supported question type matched. Numerical answers are not guessed.",
        limitations=["Try asking about a measure's summary, highest or lowest value, category ranking, time change, correlation, unusual values, data quality, or a management summary."],
        follow_ups=build_question_examples(frame, selected_measures, selected_dimensions)[:3],
        question=question,
    )


def _answer_group_ranking(frame, measure, dimensions, aggregation, low, question, follow_ups):
    grouped = aggregate_data(frame, [measure], dimensions, aggregation).dropna(subset=[measure])
    if grouped.empty:
        return _result("insufficient", "group_ranking", "There are no valid grouped values to rank.", question=question)
    selected = grouped.sort_values(measure, ascending=low, kind="stable")
    best = selected.iloc[0]
    label = " / ".join(f"{column}: {best[column]}" for column in dimensions)
    grouped_sorted = grouped.sort_values(measure, ascending=False, kind="stable")
    facts = [{"label": f"{aggregation} {measure!s} for {label}", "value": _number(best[measure])}]
    if len(grouped_sorted) > 1:
        second = grouped_sorted.iloc[1] if not low else grouped_sorted.iloc[-2]
        second_label = " / ".join(f"{column}: {second[column]}" for column in dimensions)
        facts.append({"label": "Next comparison", "value": f"{second_label} · {_number(second[measure])}"})
    return _result(
        "answered", "group_ranking",
        f"{label} has the {'lowest' if low else 'highest'} grouped {aggregation.lower()} {measure!s}: {_number(best[measure])}.",
        facts,
        calculation=f"Grouped the filtered rows by {', '.join(map(str, dimensions))} and applied {aggregation.lower()} to valid numeric {measure!s} values.",
        limitations=["A ranking describes the selected filtered data and does not explain why groups differ."],
        follow_ups=follow_ups,
        question=question,
    )


def _answer_group_comparison(frame, measure, dimensions, aggregation, question, follow_ups):
    grouped = aggregate_data(frame, [measure], dimensions, aggregation).dropna(subset=[measure])
    if grouped.empty:
        return _result("insufficient", "group_comparison", "There are no valid grouped values to compare.", question=question)
    ordered = grouped.sort_values(measure, ascending=False, kind="stable")
    display = ordered.head(5)
    facts = []
    for _, row in display.iterrows():
        label = " / ".join(f"{column}: {row[column]}" for column in dimensions)
        facts.append({"label": label, "value": _number(row[measure])})
    high, low = float(ordered.iloc[0][measure]), float(ordered.iloc[-1][measure])
    spread = high - low
    return _result(
        "answered", "group_comparison",
        f"{aggregation} {measure!s} ranges from {_number(low)} to {_number(high)} across {len(ordered):,} groups.",
        facts,
        calculation=f"Grouped the filtered rows by {', '.join(map(str, dimensions))} and applied {aggregation.lower()} to valid numeric observations. The highest-to-lowest gap is {_number(spread)}.",
        limitations=["Group differences are descriptive and do not establish why performance varies."],
        follow_ups=follow_ups,
        question=question,
    )


def _answer_time_ranking(frame, measure, date_column, granularity, aggregation, low, question, follow_ups):
    grouped = aggregate_data(frame, [measure], [], aggregation, date_column, granularity).dropna(subset=[measure])
    if grouped.empty:
        return _result("insufficient", "time_ranking", "There are no valid time buckets to rank.", question=question)
    ordered = grouped.sort_values(measure, ascending=low, kind="stable")
    selected = ordered.iloc[0]
    label = selected["Date"].strftime("%Y-%m-%d")
    facts = [{"label": f"{granularity} beginning", "value": label}, {"label": f"{aggregation} {measure!s}", "value": _number(selected[measure])}]
    if len(ordered) > 1:
        second = ordered.iloc[1]
        facts.append({"label": "Next period", "value": f"{second['Date']:%Y-%m-%d} · {_number(second[measure])}"})
    return _result(
        "answered", "time_ranking",
        f"The {'lowest' if low else 'highest'} {aggregation.lower()} {measure!s} occurred in the period starting {label}: {_number(selected[measure])}.",
        facts,
        calculation=f"Parsed {date_column!s}, grouped by {granularity.lower()}, and ranked valid {aggregation.lower()} values.",
        limitations=["This ranking describes observed periods; it does not explain the cause of the difference."],
        follow_ups=follow_ups,
        question=question,
    )


def _answer_extreme(frame, measure, low, question, follow_ups):
    summary = numeric_summary(frame[measure].tolist())
    extreme = summary["minimum"] if low else summary["maximum"]
    if extreme is None:
        return _result("insufficient", "extreme", f"There are no valid numeric observations in {measure!s}.", question=question)
    return _result(
        "answered", "extreme",
        f"The {'lowest' if low else 'highest'} observed {measure!s} value is {_number(extreme)}.",
        [{"label": "Minimum" if low else "Maximum", "value": _number(extreme)}, {"label": "Valid observations", "value": summary["count"]}],
        calculation=f"Used the {'minimum' if low else 'maximum'} of {summary['count']} valid numeric observations in the filtered {measure!s} column.",
        limitations=["This is an observed value; the dataset does not establish its cause or whether it is representative."],
        follow_ups=follow_ups,
        question=question,
    )


def _answer_average_vs_highest(frame, measure, question, follow_ups):
    summary = numeric_summary(frame[measure].tolist())
    if not summary["count"]:
        return _result("insufficient", "average_vs_highest", f"There are no valid values in {measure!s}.", question=question)
    gap = float(summary["maximum"] - summary["mean"])
    relative = gap / abs(float(summary["maximum"])) * 100 if summary["maximum"] else None
    return _result(
        "answered", "average_vs_highest",
        f"The average {measure!s} is {_number(summary['mean'])}, compared with a highest value of {_number(summary['maximum'])}.",
        [
            {"label": "Average", "value": _number(summary["mean"])},
            {"label": "Highest value", "value": _number(summary["maximum"])},
            {"label": "Highest minus average", "value": _number(gap)},
            {"label": "Gap as share of highest", "value": f"{relative:.1f}%" if relative is not None else "Unavailable (highest value is zero)"},
        ],
        calculation=f"Compared the arithmetic mean and maximum across {summary['count']} valid numeric observations.",
        limitations=["The comparison is descriptive and does not explain the distribution or causes of variation."],
        follow_ups=follow_ups,
        question=question,
    )


def _answer_summary_metric(frame, measure, lowered, question, follow_ups):
    summary = numeric_summary(frame[measure].tolist())
    metric_key = "sum" if "sum" in lowered or "total" in lowered else "median" if "median" in lowered else "count" if "count" in lowered or "how many" in lowered else "mean"
    metric_name = {"sum": "Total", "mean": "Average", "median": "Median", "count": "Valid observations"}[metric_key]
    value = summary[metric_key]
    if value is None:
        return _result("insufficient", "summary_metric", f"There are no valid values to calculate {metric_name.lower()} for {measure!s}.", question=question)
    return _result("answered", "summary_metric", f"{metric_name} {measure!s} is {_number(value) if metric_key != 'count' else f'{value:,}'}.", [{"label": metric_name, "value": _number(value) if metric_key != "count" else value}], f"Calculated {metric_name.lower()} from {summary['count']} valid numeric observations in the current analysis rows; when app filters are active, only matching rows are included.", follow_ups=follow_ups, question=question)


def _answer_quality(frame, measure, question, follow_ups):
    quality, profiles = profile_dataset(frame)
    issues = []
    for _, row in quality.iterrows():
        column = str(row["Column"])
        missing = int(row["Missing"] or 0)
        nonnumeric = profiles[column]["nonnumeric_count"]
        if missing or nonnumeric:
            issues.append({"label": column, "value": f"{missing:,} missing; {nonnumeric:,} nonnumeric"})
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        issues.append({"label": "Duplicate rows", "value": f"{duplicates:,}"})
    available_measures, _, _ = _available_columns(frame)
    selected_measure = measure if measure in frame.columns and measure in available_measures else (available_measures[0] if available_measures else None)
    numeric = coerce_numeric(frame[selected_measure]).dropna() if selected_measure is not None else pd.Series(dtype=float)
    if selected_measure is not None and len(numeric) >= 4:
        q1, q3 = numeric.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        outliers = numeric[(numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)]
        issues.append({"label": f"Potential {selected_measure!s} outliers (1.5×IQR)", "value": f"{len(outliers):,} of {len(numeric):,} valid values"})
    if not issues:
        issues.append({"label": "Quality checks", "value": "No missing, nonnumeric, duplicate, or IQR outlier flags were found in the current filtered rows."})
    return _result(
        "answered", "quality",
        f"Data-quality checks found {sum(1 for item in issues if item['label'] != 'Quality checks'):,} reported issue type(s) in the current filtered data.",
        issues,
        calculation="Checked missing and nonnumeric values by column, exact duplicate rows, and potential measure outliers using the 1.5×IQR screening rule.",
        limitations=["IQR flags are screening signals, not proof that a value is erroneous.", "Only the currently filtered rows are checked."],
        follow_ups=follow_ups,
        question=question,
    )


def _answer_correlation(frame, pair, question, follow_ups):
    if len(pair) < 2:
        return _result("insufficient", "correlation", "Select or name two numeric measures to calculate a relationship.", limitations=["At least two numeric columns are required."], question=question)
    pair_frame = pd.DataFrame({str(column): coerce_numeric(frame[column]) for column in pair}).dropna()
    if len(pair_frame) < 2 or pair_frame.nunique().min() < 2:
        return _result("insufficient", "correlation", "There are not enough paired, varying observations to calculate correlation.", calculation="Pearson correlation requires at least two paired observations and variation in both measures.", question=question)
    correlation = float(pair_frame.iloc[:, 0].corr(pair_frame.iloc[:, 1]))
    return _result(
        "answered", "correlation",
        f"The Pearson correlation between {pair[0]!s} and {pair[1]!s} is {correlation:+.3f} across {len(pair_frame):,} paired observations.",
        [{"label": "Pearson correlation", "value": f"{correlation:+.3f}"}, {"label": "Paired observations", "value": len(pair_frame)}],
        calculation="Calculated Pearson's correlation using rows with valid numeric values in both measures.",
        limitations=["Correlation indicates association only; it does not show that one measure causes changes in the other."],
        follow_ups=follow_ups,
        question=question,
    )


def generate_dataset_insights(
    frame: pd.DataFrame,
    selected_measures: list[Any] | None = None,
    selected_dimensions: list[Any] | None = None,
    aggregation: str = "Sum",
    date_column: Any | None = None,
    date_granularity: str = "Month",
) -> list[dict[str, Any]]:
    """Create concise, evidence-backed findings from the current filtered rows."""
    if frame.empty:
        return [{"kind": "info", "text": "No rows are available after filtering; clear or adjust filters to generate findings.", "evidence": []}]
    measures, dimensions, dates = _available_columns(frame)
    chosen_measures = [column for column in (selected_measures or measures[:1]) if column in measures]
    chosen_dimensions = [column for column in (selected_dimensions or []) if column in dimensions]
    if not chosen_dimensions and dimensions:
        chosen_dimensions = dimensions[:1]
    chosen_date = date_column if date_column in dates else (dates[0] if dates else None)
    findings: list[dict[str, Any]] = []
    for measure in chosen_measures[:3]:
        summary = numeric_summary(frame[measure].tolist())
        if summary["count"]:
            findings.append({
                "kind": "metric",
                "text": f"{measure!s}: {_number(summary['sum'])} total across {summary['count']:,} valid observations; average {_number(summary['mean'])}.",
                "evidence": [{"label": "Total", "value": _number(summary["sum"])}, {"label": "Valid observations", "value": summary["count"]}],
            })
            for insight in build_business_insights(frame, str(measure), chosen_dimensions, chosen_date, date_granularity, aggregation):
                if insight["kind"] in {"success", "warning"} or (chosen_date and "changed by" in insight["text"]) or (not chosen_date and chosen_dimensions and "calendar-month order" in insight["text"]):
                    findings.append({"kind": insight["kind"], "text": insight["text"], "evidence": []})
            if chosen_dimensions:
                grouped_sums = aggregate_data(frame, [measure], chosen_dimensions, "Sum").dropna(subset=[measure])
                if len(grouped_sums) >= 3 and (grouped_sums[measure] >= 0).all():
                    denominator = float(grouped_sums[measure].sum())
                    if denominator > 0:
                        top_groups = grouped_sums.nlargest(3, measure)
                        concentration = float(top_groups[measure].sum()) / denominator * 100
                        if concentration >= 80:
                            labels = [" / ".join(f"{dimension}: {row[dimension]}" for dimension in chosen_dimensions) for _, row in top_groups.iterrows()]
                            findings.append({
                                "kind": "info",
                                "text": f"The top {len(top_groups)} groups account for {concentration:.1f}% of valid grouped {measure!s} total ({'; '.join(labels)}). This concentration flag uses Sum and a screening threshold of 80%.",
                                "evidence": [{"label": "Top-group share", "value": f"{concentration:.1f}%"}, {"label": "Groups included", "value": labels}],
                            })
    quality, profiles = profile_dataset(frame)
    quality_issues = []
    for _, row in quality.iterrows():
        column = str(row["Column"])
        missing = int(row["Missing"] or 0)
        nonnumeric = profiles[column]["nonnumeric_count"]
        if missing or nonnumeric:
            quality_issues.append(f"{column}: {missing:,} missing, {nonnumeric:,} nonnumeric")
    duplicate_count = int(frame.duplicated().sum())
    if duplicate_count:
        quality_issues.append(f"{duplicate_count:,} exact duplicate rows")
    if quality_issues:
        findings.append({"kind": "warning", "text": "Data-quality flags: " + "; ".join(quality_issues[:5]) + ".", "evidence": quality_issues[:5]})
    if not findings:
        findings.append({"kind": "info", "text": "The current data does not contain enough numeric or categorical information to support a performance finding.", "evidence": []})
    return findings[:8]


def generate_executive_summary(
    frame: pd.DataFrame,
    selected_measures: list[Any] | None = None,
    selected_dimensions: list[Any] | None = None,
    aggregation: str = "Sum",
    date_column: Any | None = None,
    date_granularity: str = "Month",
    question: str = "",
) -> dict[str, Any]:
    """Generate a deterministic management summary from calculated findings."""
    if frame.empty:
        return _result("insufficient", "executive_summary", "No rows remain in the current filtered dataset.", limitations=["Clear or adjust filters to summarize data."], question=question)
    findings = generate_dataset_insights(frame, selected_measures, selected_dimensions, aggregation, date_column, date_granularity)
    measures, _, _ = _available_columns(frame)
    facts = [
        {"label": "Filtered rows", "value": len(frame)},
        {"label": "Columns", "value": len(frame.columns)},
    ]
    for measure in (selected_measures or measures)[:3]:
        if measure in frame.columns:
            summary = numeric_summary(frame[measure].tolist())
            if summary["count"]:
                facts.extend([
                    {"label": f"{measure!s} total", "value": _number(summary["sum"])},
                    {"label": f"{measure!s} average", "value": _number(summary["mean"])},
                ])
    limitations = []
    if len(frame) < 4:
        limitations.append("The filtered dataset is small; distribution and outlier conclusions are limited.")
    limitations.append("Findings describe observed data and do not establish causes or predict future outcomes.")
    return _result(
        "answered", "executive_summary",
        f"The current filtered dataset contains {len(frame):,} rows and {len(frame.columns):,} columns. " + " ".join(item["text"] for item in findings[:4]),
        facts,
        calculation=f"Summary generated from {len(frame):,} current filtered rows using Python calculations and the selected analysis settings.",
        limitations=limitations,
        follow_ups=build_question_examples(frame, selected_measures, selected_dimensions)[:3],
        question=question or "Summarize the main findings for a management meeting.",
    )
