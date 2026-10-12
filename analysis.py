"""Data loading, validation, filtering, and deterministic business analysis."""

from __future__ import annotations

from calendar import month_abbr, month_name
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from math import isfinite
import re
from typing import Any, Iterable, Mapping

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


SUPPORTED_EXTENSIONS = {"csv", "xls", "xlsx"}
AGGREGATIONS = {"Sum": "sum", "Average": "mean", "Median": "median", "Minimum": "min", "Maximum": "max", "Count": "count"}
MONTH_NUMBERS = {
    label.casefold(): number
    for number in range(1, 13)
    for label in (month_name[number], month_abbr[number])
}
MONTH_NUMBERS["sept"] = 9


def month_number(value: Any) -> int | None:
    """Return calendar order for a full or abbreviated English month label."""
    if not isinstance(value, str):
        return None
    return MONTH_NUMBERS.get(value.strip().casefold())


def is_month_label_column(series: pd.Series) -> bool:
    """Whether all present values are month names or common abbreviations."""
    missing = _missing_mask(series)
    present = series.loc[~missing]
    return not present.empty and present.map(month_number).notna().all()


def get_excel_sheet_names(content: bytes, extension: str) -> list[str]:
    """Return workbook sheet names without retaining uploaded data globally."""
    if extension.lower().lstrip(".") not in {"xls", "xlsx"}:
        return []
    try:
        with pd.ExcelFile(BytesIO(content)) as workbook:
            if not workbook.sheet_names:
                raise ValueError("The Excel workbook does not contain any worksheets.")
            return workbook.sheet_names
    except ImportError as exc:
        engine = "xlrd" if extension.lower().endswith("xls") else "openpyxl"
        raise ValueError(f"Reading this Excel format requires the '{engine}' package. Install requirements.txt and retry.") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read this Excel workbook: {exc}") from exc


def load_dataset(content: bytes, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Load an uploaded table. This function does not cache or mutate source data."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a CSV, XLS, or XLSX file.")
    if not content:
        raise ValueError("The uploaded file is empty.")
    try:
        if extension == "csv":
            frame = pd.read_csv(BytesIO(content))
        else:
            names = get_excel_sheet_names(content, extension)
            selected_sheet = sheet_name if sheet_name in names else names[0]
            frame = pd.read_excel(BytesIO(content), sheet_name=selected_sheet)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV has no readable header or data rows.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"The CSV could not be parsed. Check its delimiter, quoting, and row lengths. Details: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV encoding is not supported. Save it as UTF-8 and try again.") from exc
    except ImportError as exc:
        engine = "xlrd" if extension == "xls" else "openpyxl"
        raise ValueError(f"Reading this Excel format requires the '{engine}' package. Install requirements.txt and retry.") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read '{filename}': {exc}") from exc

    if len(frame.columns) == 0:
        raise ValueError("The file does not contain a header row with columns.")
    usable_headers = [
        column for column in frame.columns
        if str(column).strip() and not str(column).startswith("Unnamed:")
    ]
    if not usable_headers:
        raise ValueError("The file has no usable column headers. Add a header row and try again.")
    return frame


def _missing_mask(series: pd.Series) -> pd.Series:
    missing = series.isna()
    blanks = series.map(lambda value: isinstance(value, str) and not value.strip()).astype(bool)
    return missing | blanks


def rows_with_missing_values(frame: pd.DataFrame) -> pd.Series:
    """Mark rows with any null, blank, or whitespace-only cell.

    This rule is shared by the optional analysis filter. Numeric zero and False
    are valid values and are not treated as missing.
    """
    if frame.empty or frame.shape[1] == 0:
        return pd.Series(False, index=frame.index, dtype=bool)
    return pd.concat(
        [_missing_mask(frame[column]).rename(column) for column in frame.columns], axis=1
    ).any(axis=1)


def exclude_rows_with_missing_values(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return a copy without incomplete rows and the number excluded."""
    missing_rows = rows_with_missing_values(frame)
    return frame.loc[~missing_rows].copy(), int(missing_rows.sum())


def _coerce_finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.map(lambda value: pd.isna(value) or isfinite(float(value)))
    return numeric.where(finite)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Return a numeric analysis view, coercing invalid and infinite values to missing."""
    return _coerce_finite_numeric(series)


def _parse_dates(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", utc=True)


def profile_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Summarize column types and quality without altering the supplied frame."""
    profiles: dict[str, dict[str, Any]] = {}
    report_rows = []
    for column in frame.columns:
        series = frame[column]
        missing = _missing_mask(series)
        numeric = _coerce_finite_numeric(series.mask(missing))
        numeric_count = int(numeric.notna().sum())
        source_is_numeric = is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)
        is_numeric = source_is_numeric or numeric_count > 0

        month_order = series.map(month_number).astype("Int64")
        month_count = int(month_order.notna().sum())
        nonmissing_count = int((~missing).sum())
        is_month = nonmissing_count > 0 and month_count == nonmissing_count
        dates = _parse_dates(series)
        date_count = int(dates.notna().sum())
        source_is_date = is_datetime64_any_dtype(series.dtype)
        column_words = set(re.findall(r"[a-z0-9]+", str(column).lower()))
        date_named = bool(column_words & {"date", "datetime", "timestamp", "time"})
        # Require a majority of nonblank values to parse, avoiding accidental dates in categories.
        is_date = not is_month and (source_is_date or (
            nonmissing_count > 0
            and (date_named or (not is_numeric and date_count / nonmissing_count >= 0.6))
        ))
        nonnumeric_count = max(0, nonmissing_count - numeric_count) if is_numeric else 0
        date_parse_failures = max(0, nonmissing_count - date_count) if is_date else 0
        role = "Numeric" if is_numeric else "Date/time" if is_date else "Boolean" if is_bool_dtype(series.dtype) else "Category / text"
        profiles[str(column)] = {
            "column": column,
            "role": role,
            "is_numeric": is_numeric,
            "is_date": is_date,
            "is_month": is_month,
            "numeric": numeric.astype(float),
            "dates": dates,
            "month_order": month_order,
            "missing_count": int(missing.sum()),
            "nonnumeric_count": nonnumeric_count,
            "date_parse_failures": date_parse_failures,
            "valid_date_count": date_count,
            "numeric_count": numeric_count,
        }
        report_rows.append(
            {
                "Column": str(column),
                "Detected type": str(series.dtype),
                "Suggested role": role,
                "Missing": int(missing.sum()),
                "Nonnumeric in numeric data": nonnumeric_count if is_numeric else pd.NA,
                "Date parse failures": date_parse_failures if is_date else pd.NA,
            }
        )
    quality = pd.DataFrame(report_rows, columns=["Column", "Detected type", "Suggested role", "Missing", "Nonnumeric in numeric data", "Date parse failures"])
    for column in ("Missing", "Nonnumeric in numeric data", "Date parse failures"):
        quality[column] = quality[column].astype("Int64")
    return quality, profiles


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    """Calculate common descriptive metrics with explicit valid/missing counts."""
    series = pd.Series(list(values), dtype="object")
    missing = _missing_mask(series)
    numeric = _coerce_finite_numeric(series.mask(missing))
    valid = numeric.dropna().astype(float)
    count = int(valid.count())
    return {
        "count": count,
        "missing_count": int(missing.sum()),
        "nonnumeric_count": int((~missing).sum() - count),
        "sum": float(valid.sum()) if count else None,
        "mean": float(valid.mean()) if count else None,
        "median": float(valid.median()) if count else None,
        "minimum": float(valid.min()) if count else None,
        "maximum": float(valid.max()) if count else None,
        "standard_deviation": float(valid.std(ddof=1)) if count > 1 else None,
    }


def apply_filters(
    frame: pd.DataFrame,
    category_filters: Mapping[str, Iterable[Any]] | None = None,
    numeric_ranges: Mapping[str, tuple[float, float]] | None = None,
    date_ranges: Mapping[str, tuple[Any, Any]] | None = None,
) -> pd.DataFrame:
    """Return a filtered copy; the caller's original dataframe remains unchanged."""
    result = frame.copy()
    for column, selected in (category_filters or {}).items():
        if column not in result.columns:
            continue
        selected_values = list(selected)
        include_missing = any(
            value is None
            or (isinstance(value, str) and not value.strip())
            or (not isinstance(value, str) and _scalar_missing(value))
            for value in selected_values
        )
        regular_values = [
            value for value in selected_values
            if not (value is None or (isinstance(value, str) and not value.strip()) or (not isinstance(value, str) and _scalar_missing(value)))
        ]
        mask = result[column].isin(regular_values)
        if include_missing:
            mask = mask | _missing_mask(result[column])
        result = result[mask]
    for column, bounds in (numeric_ranges or {}).items():
        if column not in result.columns:
            continue
        values = _coerce_finite_numeric(result[column])
        result = result[values.between(bounds[0], bounds[1], inclusive="both")]
    for column, bounds in (date_ranges or {}).items():
        if column not in result.columns:
            continue
        dates = _parse_dates(result[column])
        start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        result = result[dates.between(start, end, inclusive="both")]
    return result


def _scalar_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


_BUSINESS_ALIASES = {
    "revenue": ("revenue", "sales", "amount", "gross revenue", "net sales", "sales amount", "total sales"),
    "cost": ("total cost", "cost", "costs", "cogs", "total costs", "expense", "expenses"),
    "profit": ("profit", "net profit", "gross profit", "operating profit", "earnings"),
    "reported_margin": ("profit margin", "margin", "margin percent", "margin pct"),
    "units": ("units sold", "units", "quantity", "qty", "volume"),
    "product": ("product", "item", "sku", "service"),
    "category": ("category", "product category", "department", "segment"),
    "region": ("region", "territory", "market", "area"),
    "channel": ("sales channel", "channel", "order channel"),
    "identifier": ("order id", "transaction id", "invoice id", "record id", "id"),
}


def detect_business_columns(frame: pd.DataFrame) -> dict[str, Any]:
    """Map familiar business concepts to columns without assuming they exist."""
    normalized = {re.sub(r"[^a-z0-9]+", " ", str(column).casefold()).strip(): column for column in frame.columns}
    result: dict[str, Any] = {key: None for key in _BUSINESS_ALIASES}
    for role, aliases in _BUSINESS_ALIASES.items():
        for alias in aliases:
            key = re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
            if key in normalized:
                result[role] = normalized[key]
                break
    quality, profiles = profile_dataset(frame)
    date_candidates = [column for column in frame.columns if profiles[str(column)]["is_date"] and profiles[str(column)]["valid_date_count"]]
    result["date"] = next((column for column in date_candidates if re.search(r"date|time|timestamp", str(column).casefold())), date_candidates[0] if date_candidates else None)
    result["duplicate_rows"] = 0
    result["duplicate_ids"] = []
    identifier = result.get("identifier")
    if identifier is not None:
        values = frame[identifier]
        valid = ~_missing_mask(values)
        duplicate_mask = pd.Series(False, index=frame.index)
        duplicate_mask.loc[valid] = values.loc[valid].duplicated(keep="first")
        result["duplicate_rows"] = int(duplicate_mask.sum())
        result["duplicate_ids"] = [str(value) for value in values.loc[duplicate_mask].drop_duplicates().tolist()]
    # Prefer an explicitly named revenue field over a generic amount when both exist.
    return result


def remove_duplicate_identifier_rows(frame: pd.DataFrame, identifier: Any) -> pd.DataFrame:
    """Return a copy keeping the first nonblank identifier; preserve blank IDs separately."""
    if identifier not in frame.columns:
        raise ValueError(f"Unknown identifier column: {identifier!s}.")
    values = frame[identifier]
    valid = ~_missing_mask(values)
    duplicate_mask = pd.Series(False, index=frame.index)
    duplicate_mask.loc[valid] = values.loc[valid].duplicated(keep="first")
    return frame.loc[~duplicate_mask].copy()


def calculate_business_performance(
    frame: pd.DataFrame,
    *,
    identifier: Any | None = None,
    remove_duplicate_identifiers: bool = False,
) -> dict[str, Any]:
    """Calculate financial KPIs and segment tables from a supplied (usually filtered) frame.

    Duplicate identifiers are included by default. If explicitly requested, the first
    nonblank identifier occurrence is retained; blank identifiers remain separate rows.
    """
    roles = detect_business_columns(frame)
    identifier = identifier if identifier in frame.columns else roles["identifier"]
    duplicate_rows = 0
    duplicate_ids: list[str] = []
    analysis = frame.copy()
    if identifier is not None:
        values = analysis[identifier]
        valid = ~_missing_mask(values)
        duplicate_mask = pd.Series(False, index=analysis.index)
        duplicate_mask.loc[valid] = values.loc[valid].duplicated(keep="first")
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_ids = [str(value) for value in values.loc[duplicate_mask].drop_duplicates().tolist()]
        if remove_duplicate_identifiers:
            analysis = analysis.loc[~duplicate_mask].copy()
    roles["identifier"] = identifier
    roles["duplicate_rows"] = duplicate_rows
    roles["duplicate_ids"] = duplicate_ids
    roles["duplicate_policy"] = "first identifier occurrence retained" if remove_duplicate_identifiers else "all uploaded rows included"
    roles["duplicate_scope"] = "provided analysis frame"
    roles["input_rows"] = len(frame)
    roles["rows"] = len(analysis)
    metrics: dict[str, Any] = {}
    for role in ("revenue", "cost", "profit", "units"):
        column = roles[role]
        if column is None:
            metrics[role] = None
            continue
        values = _coerce_finite_numeric(analysis[column])
        valid = values.dropna()
        metrics[role] = {
            "column": str(column), "count": int(valid.count()),
            "missing_or_invalid": int(values.isna().sum()),
            "total": float(valid.sum()) if not valid.empty else None,
            "average": float(valid.mean()) if not valid.empty else None,
        }
    if roles["revenue"] is not None and roles["profit"] is not None:
        revenue = _coerce_finite_numeric(analysis[roles["revenue"]])
        profit = _coerce_finite_numeric(analysis[roles["profit"]])
        paired = revenue.notna() & profit.notna()
        denominator = float(revenue.loc[paired].sum()) if paired.any() else 0.0
        numerator = float(profit.loc[paired].sum()) if paired.any() else 0.0
        metrics["profit_margin"] = numerator / denominator if denominator != 0 else None
        metrics["margin_pair_count"] = int(paired.sum())
    else:
        metrics["profit_margin"] = None
        metrics["margin_pair_count"] = 0
    metrics["reported_margin_column"] = roles.get("reported_margin")

    breakdowns: dict[str, pd.DataFrame] = {}
    for role in ("product", "category", "region", "channel"):
        dimension = roles[role]
        if dimension is None:
            continue
        work = pd.DataFrame({"_dimension": analysis[dimension].astype("string").fillna("(Missing)")}, index=analysis.index)
        for metric_role in ("revenue", "cost", "profit", "units"):
            column = roles[metric_role]
            if column is not None:
                work[metric_role] = _coerce_finite_numeric(analysis[column])
        agg: dict[str, Any] = {column: (lambda s: s.sum(min_count=1)) for column in work.columns if column != "_dimension"}
        grouped = work.groupby("_dimension", dropna=False, sort=True).agg(agg).reset_index().rename(columns={"_dimension": str(dimension)})
        if "revenue" in grouped and "profit" in grouped:
            grouped["profit_margin"] = grouped.apply(
                lambda row: row["profit"] / row["revenue"] if pd.notna(row["profit"]) and pd.notna(row["revenue"]) and row["revenue"] != 0 else pd.NA,
                axis=1,
            )
        breakdowns[role] = grouped

    time_series = pd.DataFrame()
    date_column = roles["date"]
    if date_column is not None:
        parsed = _parse_dates(analysis[date_column])
        time = pd.DataFrame({"period": parsed.dt.tz_convert(None).dt.to_period("M").dt.start_time}, index=analysis.index)
        for role in ("revenue", "cost", "profit"):
            column = roles[role]
            if column is not None:
                time[role] = _coerce_finite_numeric(analysis[column])
        time = time.dropna(subset=["period"])
        if not time.empty:
            agg = {column: (lambda s: s.sum(min_count=1)) for column in time.columns if column != "period"}
            time_series = time.groupby("period", sort=True).agg(agg).reset_index()
            if "revenue" in time_series and "profit" in time_series:
                time_series["profit_margin"] = time_series.apply(
                    lambda row: row["profit"] / row["revenue"] if pd.notna(row["profit"]) and pd.notna(row["revenue"]) and row["revenue"] != 0 else pd.NA,
                    axis=1,
                )
    return {"frame": analysis, "columns": roles, "metrics": metrics, "breakdowns": breakdowns, "time_series": time_series}


def generate_business_insights(performance: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return evidence-backed, non-causal findings from calculated performance data."""
    columns, metrics = performance["columns"], performance["metrics"]
    findings: list[dict[str, str]] = []
    for role in ("product", "category", "region", "channel"):
        table = performance["breakdowns"].get(role)
        if table is None or table.empty:
            continue
        dimension = str(columns[role])
        for metric in ("profit", "revenue"):
            if metric not in table or table[metric].notna().sum() < 2:
                continue
            valid = table.dropna(subset=[metric]).sort_values(metric, kind="stable")
            low, high = valid.iloc[0], valid.iloc[-1]
            findings.append({
                "headline": f"{high[dimension]} leads {role} {metric}",
                "finding": f"{high[dimension]} has the highest {metric} at {high[metric]:,.2f}; {low[dimension]} is lowest at {low[metric]:,.2f}.",
                "comparison": f"Observed spread: {float(high[metric]) - float(low[metric]):,.2f} across {len(valid)} groups.",
                "implication": f"Use this ranking to prioritize a closer review of {role} performance; it does not explain why results differ.",
                "limitation": "Based on the current filtered rows and available valid values.",
            })
        if "profit" in table and (table["profit"].dropna() < 0).any():
            losses = table.loc[table["profit"] < 0].sort_values("profit")
            row = losses.iloc[0]
            findings.append({"headline": f"Negative-profit {role} identified", "finding": f"{row[dimension]} records {row['profit']:,.2f} profit.", "comparison": f"{len(losses)} of {len(table)} {role} groups have negative aggregate profit.", "implication": "Review the segment's underlying pricing and cost records before taking action.", "limitation": "The dataset shows the financial result, not its cause."})
        if "revenue" in table and "profit_margin" in table and table["profit_margin"].notna().sum() >= 2:
            margin_median = table["profit_margin"].dropna().median()
            candidates = table[(table["revenue"] > 0) & (table["profit_margin"] < margin_median)]
            if not candidates.empty:
                row = candidates.sort_values("revenue", ascending=False).iloc[0]
                findings.append({"headline": f"High-revenue, below-median-margin {role}", "finding": f"{row[dimension]} contributes {row['revenue']:,.2f} revenue at {row['profit_margin']:.1%} margin.", "comparison": f"Its margin is below the {margin_median:.1%} median across {table['profit_margin'].notna().sum()} groups.", "implication": "Review underlying records in this high-revenue segment to identify potential margin drivers.", "limitation": "Margin is aggregate profit divided by aggregate revenue; no causal driver is established."})
    time = performance["time_series"]
    if not time.empty and len(time) >= 2 and "revenue" in time:
        first, last = time.iloc[0], time.iloc[-1]
        for metric in ("revenue", "profit", "profit_margin"):
            if metric not in time or pd.isna(first[metric]) or pd.isna(last[metric]):
                continue
            delta = float(last[metric] - first[metric])
            if metric == "profit_margin":
                change_text = f"{delta * 100:+.1f} percentage points"
                first_value, last_value = f"{first[metric]:.1%}", f"{last[metric]:.1%}"
            else:
                pct = f"{delta / abs(float(first[metric])):+.1%}" if first[metric] != 0 else "percentage change unavailable (zero baseline)"
                change_text = f"{delta:+,.2f} ({pct})"
                first_value, last_value = f"{first[metric]:,.2f}", f"{last[metric]:,.2f}"
            findings.append({"headline": f"{metric.replace('_', ' ').title()} changed between the first and last observed months", "finding": f"{metric.replace('_', ' ').title()} moved from {first_value} ({first['period']:%b %Y}) to {last_value} ({last['period']:%b %Y}).", "comparison": f"Change: {change_text}; {len(time)} monthly periods are available.", "implication": "Compare intermediate periods and business context before treating the endpoint change as a trend.", "limitation": "This is an endpoint comparison, not a forecast or causal explanation."})
        if "revenue" in time and "profit" in time and first["revenue"] and first["profit"]:
            revenue_growth = (float(last["revenue"]) - float(first["revenue"])) / abs(float(first["revenue"]))
            profit_growth = (float(last["profit"]) - float(first["profit"])) / abs(float(first["profit"]))
            if abs(revenue_growth - profit_growth) >= 0.10:
                findings.append({"headline": "Revenue and profit changed at different rates", "finding": f"Endpoint revenue change was {revenue_growth:+.1%}; endpoint profit change was {profit_growth:+.1%}.", "comparison": f"Difference in growth rates: {(profit_growth - revenue_growth) * 100:+.1f} percentage points over {len(time)} monthly periods.", "implication": "Review cost and mix information to understand the divergence; these figures alone do not identify its cause.", "limitation": "Compares the first and last available periods only."})
        monthly_revenue = time.dropna(subset=["revenue"])
        for position in range(1, len(monthly_revenue)):
            previous, current = monthly_revenue.iloc[position - 1], monthly_revenue.iloc[position]
            if previous["revenue"] != 0:
                change = (float(current["revenue"]) - float(previous["revenue"])) / abs(float(previous["revenue"]))
                if abs(change) >= 0.5:
                    findings.append({"headline": "Large month-to-month revenue movement", "finding": f"Revenue changed {change:+.1%} from {previous['period']:%b %Y} to {current['period']:%b %Y}.", "comparison": "Screening threshold: an absolute change of at least 50% between adjacent observed months.", "implication": "Validate the periods and investigate relevant operational context before acting.", "limitation": "The threshold is a screening rule and does not establish an unusual cause."})
                    break
    for role in ("product", "category", "region", "channel"):
        table = performance["breakdowns"].get(role)
        if table is None or "revenue" not in table or len(table) < 3:
            continue
        dimension = str(columns[role])
        known_groups = table.loc[table[dimension].astype("string") != "(Missing)"]
        totals = known_groups["revenue"].dropna()
        top_count = min(3, len(totals))
        # Three groups are not a "small subset" when there are only three or
        # four known groups. Exclude the synthetic Missing bucket from the
        # dimension count and require the compared top groups to be less than
        # half of the known category set.
        is_small_subset = top_count > 0 and top_count < len(totals) / 2
        if is_small_subset and (totals >= 0).all() and totals.sum() > 0:
            top = known_groups.nlargest(top_count, "revenue")
            share = float(top["revenue"].sum() / totals.sum())
            if share >= 0.8:
                names = ", ".join(str(value) for value in top[dimension].tolist())
                findings.append({"headline": f"Revenue is concentrated in a few {role} groups", "finding": f"The top {len(top)} groups ({names}) account for {share:.1%} of revenue.", "comparison": f"Compared with the full {len(totals)} groups; screening threshold is 80%.", "implication": "Consider the exposure if these groups underperform, while validating the result against a longer period.", "limitation": "Concentration is descriptive and not itself evidence of risk or future performance."})
    if columns.get("identifier") is not None and columns.get("duplicate_rows"):
        examples = ", ".join(columns["duplicate_ids"][:5])
        if columns.get("duplicate_policy") == "first identifier occurrence retained":
            finding = (
                f"{columns['duplicate_rows']:,} later row(s) with repeated identifiers were found in "
                f"{columns.get('duplicate_scope', 'the analysis frame')}; examples: {examples}."
            )
            comparison = f"The first occurrence was retained and {columns['duplicate_rows']:,} later row(s) are excluded from the current analysis scope."
            implication = "The selected first-occurrence rule is already applied; reconsider it only if the business definition requires another treatment."
        else:
            finding = (
                f"{columns['duplicate_rows']:,} later row(s) repeat an identifier in "
                f"{columns.get('duplicate_scope', 'the analysis frame')}; examples: {examples}."
            )
            comparison = "All rows are included under the current policy, so both the first and later rows contribute to row-based totals."
            implication = "Treat this as an identifier-quality observation. A repeated ID alone does not establish an erroneous transaction."
        findings.append({
            "headline": "Repeated identifiers observed",
            "finding": finding,
            "comparison": comparison,
            "implication": implication,
            "limitation": "Identifier repetition is not by itself evidence that a transaction is invalid.",
        })
    for role in ("revenue", "cost", "profit"):
        detail = metrics.get(role)
        if detail and detail["missing_or_invalid"]:
            findings.append({"headline": f"Incomplete {role} values", "finding": f"{detail['missing_or_invalid']:,} of {columns['rows']:,} rows have missing or nonnumeric {role} values.", "comparison": f"{detail['count']:,} valid values contribute to the {role} total.", "implication": "Interpret this metric as a partial total and investigate the affected records.", "limitation": "Invalid or missing values are excluded from numeric calculations."})
    if not findings:
        findings.append({"headline": "Limited segment evidence available", "finding": "The available fields or observations do not support a useful segment comparison yet.", "comparison": f"{columns['rows']:,} filtered rows were analyzed.", "implication": "Add or select a relevant financial measure and dimension to enable comparisons.", "limitation": "No unsupported performance conclusions are generated."})
    return findings[:20]


def generate_management_summary(performance: Mapping[str, Any], insights: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Create deterministic management prose and up to three evidence-led actions."""
    insights = insights if insights is not None else generate_business_insights(performance)
    columns, metrics = performance["columns"], performance["metrics"]
    parts = [f"This analysis covers {columns['rows']:,} rows ({columns['duplicate_policy']})."]
    for role, label in (("revenue", "Revenue"), ("cost", "Costs"), ("profit", "Profit")):
        metric = metrics.get(role)
        if metric and metric["total"] is not None:
            parts.append(f"{label}: {metric['total']:,.2f} from {metric['count']:,} valid records.")
    if metrics.get("profit_margin") is not None:
        parts.append(f"Aggregate profit margin: {metrics['profit_margin']:.1%} (profit divided by revenue across rows with both values).")
    if metrics.get("revenue") and metrics["revenue"]["average"] is not None:
        parts.append(f"Average revenue per record with valid revenue: {metrics['revenue']['average']:,.2f}.")
    for item in insights:
        if "leads" in item["headline"] or "changed between" in item["headline"] or "concentrated" in item["headline"]:
            parts.append(item["finding"])
            if len(parts) >= 7:
                break
    for item in insights:
        if "Negative-profit" in item["headline"] or "Incomplete" in item["headline"] or "Repeated identifiers" in item["headline"]:
            parts.append("Risk / data note: " + item["finding"])
            if len(parts) >= 8:
                break
    recs: list[tuple[float, str, str]] = []
    best_by_kind: dict[str, tuple[float, str, str]] = {}
    total_revenue = metrics.get("revenue", {}).get("total") if metrics.get("revenue") else None
    revenue_scale = max(abs(float(total_revenue or 0)), 1.0)
    row_count = max(int(columns.get("rows", 0)), 1)

    def evidence_score(item: Mapping[str, str], kind: str) -> float:
        headline = item.get("headline", "").casefold()
        role = next((role for role in ("product", "category", "region", "channel") if role in headline), None)
        table = performance.get("breakdowns", {}).get(role) if role else None
        if kind == "loss" and table is not None and "profit" in table and table["profit"].notna().any():
            loss = abs(float(table["profit"].min()))
            return loss / revenue_scale
        if kind == "margin" and table is not None and {"revenue", "profit_margin"}.issubset(table.columns):
            margins = table["profit_margin"].dropna()
            if len(margins) >= 2:
                median_margin = float(margins.median())
                candidates = table[(table["revenue"] > 0) & (table["profit_margin"] < median_margin)]
                if not candidates.empty:
                    candidate = candidates.sort_values("revenue", ascending=False).iloc[0]
                    revenue_share = float(candidate["revenue"]) / revenue_scale
                    margin_gap = median_margin - float(candidate["profit_margin"])
                    return revenue_share * margin_gap
        if kind == "concentration" and table is not None and "revenue" in table:
            dimension_column = columns.get(role)
            known_groups = table.loc[table[dimension_column].astype("string") != "(Missing)"] if dimension_column in table else table
            values = known_groups["revenue"].dropna()
            top_count = min(3, len(values))
            if top_count > 0 and top_count < len(values) / 2 and (values >= 0).all() and values.sum() > 0:
                share = float(values.nlargest(top_count).sum() / values.sum())
                return max(0.0, share - 0.8)
        if kind == "region-gap" and table is not None and "profit" in table:
            dimension_column = columns.get("region")
            known_groups = table.loc[table[dimension_column].astype("string") != "(Missing)"] if dimension_column in table else table
            valid = known_groups.dropna(subset=["profit"])
            if len(valid) >= 2:
                gap = float(valid["profit"].max() - valid["profit"].min())
                total_profit = metrics.get("profit", {}).get("total") if metrics.get("profit") else None
                score = gap / max(abs(float(total_profit or 0)), 1.0)
                return score if score >= 0.10 else 0.0
        if kind == "product-gap" and table is not None and "profit" in table:
            values = table["profit"].dropna()
            if len(values) >= 2 and values.max() > values.min():
                total_profit = metrics.get("profit", {}).get("total") if metrics.get("profit") else None
                return float(values.max() - values.min()) / max(abs(float(total_profit or 0)), 1.0)
        if kind == "period-change":
            time = performance.get("time_series")
            if time is not None and len(time) >= 2:
                changes = []
                for metric in ("revenue", "profit"):
                    if metric in time:
                        first_value, last_value = time.iloc[0][metric], time.iloc[-1][metric]
                        if pd.notna(first_value) and pd.notna(last_value) and float(first_value) != 0:
                            changes.append(abs(float(last_value) - float(first_value)) / abs(float(first_value)))
                if changes:
                    change = max(changes)
                    return change if change >= 0.10 else 0.0
        if kind == "incomplete":
            financial = [metrics.get(role) for role in ("revenue", "cost", "profit")]
            return max((detail["missing_or_invalid"] / row_count for detail in financial if detail), default=0.0)
        return 0.0

    recommendation_rules = (
        ("loss", "Negative-profit", "Review the records behind this loss-making segment before taking action"),
        ("product-gap", "leads product profit", "Review the lower-profit product against the leading product"),
        ("region-gap", "leads region profit", "Compare profit and margin across the known regions"),
        ("margin", "below-median-margin", "Compare the revenue and cost mix in this below-median-margin segment"),
        ("concentration", "concentrated", "Review the exposure represented by this revenue concentration"),
        ("period-change", "changed between the first and last observed months", "Review the observed change across the reporting period"),
        ("incomplete", "Incomplete", "Validate or disclose incomplete financial data before decisions based on these totals"),
    )
    for item in insights:
        headline = item.get("headline", "")
        finding = item.get("finding", "")
        for kind, marker, action in recommendation_rules:
            if marker.casefold() not in headline.casefold():
                continue
            score = evidence_score(item, kind)
            if kind == "region-gap" and score > 0:
                table = performance.get("breakdowns", {}).get("region")
                region_column = columns.get("region")
                if table is not None and region_column in table:
                    known = table.loc[table[region_column].astype("string") != "(Missing)"].dropna(subset=["profit"])
                    if len(known) >= 2:
                        low = known.loc[known["profit"].idxmin()]
                        high = known.loc[known["profit"].idxmax()]
                        low_margin = f" and {low['profit_margin']:.1%} margin" if pd.notna(low.get("profit_margin")) else ""
                        high_margin = f" and {high['profit_margin']:.1%} margin" if pd.notna(high.get("profit_margin")) else ""
                        finding = (
                            f"{low[region_column]} recorded {low['profit']:,.2f} profit{low_margin}, "
                            f"compared with {high[region_column]} at {high['profit']:,.2f}{high_margin}."
                        )
            candidate = (score, kind, f"{action}: {finding}")
            if kind not in best_by_kind or score > best_by_kind[kind][0]:
                best_by_kind[kind] = candidate
            break
    recs = sorted(best_by_kind.values(), key=lambda item: item[0], reverse=True)[:3]
    caveats = []
    if columns.get("duplicate_rows") and not performance["frame"].empty:
        if columns.get("duplicate_policy") == "all uploaded rows included":
            caveats.append(f"{columns['duplicate_rows']:,} later repeated identifier row(s) are included in this view and contribute to row-based totals; repetition alone does not establish an invalid transaction.")
        else:
            caveats.append(f"{columns['duplicate_rows']:,} later repeated identifier row(s) detected in the source were excluded; first occurrences were retained.")
    incomplete = [role for role in ("revenue", "cost", "profit") if metrics.get(role) and metrics[role]["missing_or_invalid"]]
    if incomplete:
        caveats.append("Some financial values are missing or invalid: " + ", ".join(incomplete) + ".")
    if not metrics.get("revenue") or not metrics.get("profit"):
        caveats.append("Complete profitability metrics require both revenue and profit fields; absent values are not inferred.")
    if columns.get("reported_margin") and metrics.get("profit_margin") is None:
        caveats.append(f"A reported margin field ({columns['reported_margin']}) exists, but no aggregate margin is calculated without both revenue and profit totals.")
    prioritized = [f"Priority {index} — {text}" for index, (_, _, text) in enumerate(recs, start=1)]
    return {"summary": " ".join(parts), "recommendations": prioritized, "caveats": caveats, "insights": insights}


def aggregate_data(
    frame: pd.DataFrame,
    measures: list[str],
    dimensions: list[str] | None = None,
    aggregation: str = "Sum",
    date_column: str | None = None,
    date_granularity: str = "Month",
) -> pd.DataFrame:
    """Aggregate numeric measures by dimensions and an optional date bucket.

    Count counts valid numeric observations in each measure, so missing, nonnumeric,
    and infinite source values do not inflate the result.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation {aggregation!r}. Choose one of: {', '.join(AGGREGATIONS)}.")
    unknown_measures = [measure for measure in measures if measure not in frame.columns]
    if unknown_measures:
        raise ValueError(f"Unknown measure column(s): {', '.join(map(str, unknown_measures))}.")
    unknown_dimensions = [dimension for dimension in (dimensions or []) if dimension not in frame.columns]
    if unknown_dimensions:
        raise ValueError(f"Unknown dimension column(s): {', '.join(map(str, unknown_dimensions))}.")
    dimensions = list(dimensions or [])
    work = pd.DataFrame(index=frame.index)
    group_columns: list[str] = []
    if date_column and date_column in frame.columns:
        date_values = _parse_dates(frame[date_column])
        frequency = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}.get(date_granularity, "M")
        work["Date"] = date_values.dt.tz_convert(None).dt.to_period(frequency).dt.start_time
        group_columns.append("Date")
    dimension_names: dict[str, str] = {}
    for dimension in dimensions:
        if dimension not in frame.columns or dimension == date_column:
            continue
        key = f"Dimension {len(dimension_names) + 1}"
        work[key] = frame[dimension].astype("string").fillna("(Missing)")
        group_columns.append(key)
        dimension_names[key] = str(dimension)
    value_columns = []
    for index, measure in enumerate(measures):
        key = f"Measure {index + 1}"
        work[key] = _coerce_finite_numeric(frame[measure])
        value_columns.append(key)
    if not value_columns:
        return pd.DataFrame()
    reducer = AGGREGATIONS[aggregation]
    if not group_columns:
        output = {}
        for index, measure in enumerate(measures):
            values = work[value_columns[index]].dropna()
            if reducer == "count":
                result = int(values.count())
            elif values.empty:
                result = None
            elif reducer == "sum":
                result = values.sum()
            else:
                result = getattr(values, reducer)()
            output[measure] = [result]
        return pd.DataFrame(output)
    aggregation_function = (lambda values: values.sum(min_count=1)) if reducer == "sum" else reducer
    grouped = work.groupby(group_columns, dropna=False, sort=True)[value_columns].agg(aggregation_function).reset_index()
    rename = dimension_names
    rename.update({f"Measure {index + 1}": str(column) for index, column in enumerate(measures)})
    return grouped.rename(columns=rename)


def build_business_insights(
    frame: pd.DataFrame,
    measure: str,
    dimensions: list[str] | None = None,
    date_column: str | None = None,
    date_granularity: str = "Month",
    aggregation: str = "Sum",
) -> list[dict[str, str]]:
    """Generate findings only from observable calculations in the supplied data."""
    dimensions = list(dimensions or [])
    summary = numeric_summary(frame[measure].tolist()) if measure in frame.columns else numeric_summary([])
    insights: list[dict[str, str]] = []
    if not summary["count"]:
        return [{"kind": "info", "text": f"No valid numeric observations are available for {measure!r}; insights need at least one value."}]

    if summary["minimum"] == summary["maximum"]:
        insights.append({"kind": "info", "text": f"All {summary['count']} valid {measure!r} observation(s) have the same value: {summary['minimum']:,.2f}."})
    else:
        insights.append({"kind": "info", "text": f"The largest observed {measure!r} is {summary['maximum']:,.2f}; the smallest is {summary['minimum']:,.2f}."})
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        insights.append({"kind": "warning", "text": f"The current filtered data contains {duplicates} duplicate row(s). They are retained and included in calculations."})

    if dimensions:
        grouped = aggregate_data(frame, [measure], dimensions, aggregation)
        value_name = str(measure)
        category_names = [name for name in grouped.columns if name != value_name]
        if not grouped.empty and value_name in grouped:
            grouped = grouped.dropna(subset=[value_name])
            ordered = grouped.sort_values(value_name, ascending=False, kind="stable")
            if not ordered.empty:
                high = ordered.iloc[0]
                low = ordered.iloc[-1]
                def label(row: pd.Series) -> str:
                    return " / ".join(f"{name}={row[name]}" for name in category_names)
                insights.append({"kind": "success", "text": f"Highest grouped {aggregation.lower()}: {label(high)} ({high[value_name]:,.2f}); lowest grouped {aggregation.lower()}: {label(low)} ({low[value_name]:,.2f})."})
                if len(ordered) > 1:
                    gap = float(high[value_name]) - float(low[value_name])
                    insights.append({"kind": "info", "text": f"The difference between the highest and lowest grouped {aggregation.lower()} values is {gap:,.2f}."})

    if date_column and date_column in frame.columns:
        trend = aggregate_data(frame, [measure], [], aggregation, date_column, date_granularity)
        if "Date" in trend.columns:
            # For temporal comparison, total across categories at each selected time bucket.
            value_name = str(measure)
            by_date = trend.dropna(subset=[value_name]).sort_values("Date")
            if len(by_date) >= 2:
                first, last = by_date.iloc[0], by_date.iloc[-1]
                change = float(last[value_name]) - float(first[value_name])
                if float(first[value_name]) == 0:
                    percent_text = "percentage change unavailable because the first value is zero"
                else:
                    percent = change / abs(float(first[value_name])) * 100
                    percent_text = f"a {percent:+.1f}% change"
                insights.append({"kind": "info", "text": f"From {first['Date']:%Y-%m-%d} to {last['Date']:%Y-%m-%d}, grouped {aggregation.lower()} {measure!r} changed by {change:+,.2f} ({percent_text}). Assumption: records are compared in parsed date order using {date_granularity.lower()} buckets."})
            else:
                insights.append({"kind": "info", "text": "A first-to-last change is not shown: at least two distinct valid date buckets are required."})
    else:
        month_dimension = next((dimension for dimension in dimensions
                                if dimension in frame.columns and is_month_label_column(frame[dimension])), None)
        if month_dimension is not None:
            month_frame = pd.DataFrame({
                "_month_order": frame[month_dimension].map(month_number),
                "_value": _coerce_finite_numeric(frame[measure]),
            }).dropna(subset=["_month_order", "_value"])
            reducer = AGGREGATIONS.get(aggregation, "sum")
            if reducer == "sum":
                monthly = month_frame.groupby("_month_order", sort=True)["_value"].agg(lambda values: values.sum(min_count=1))
            else:
                monthly = month_frame.groupby("_month_order", sort=True)["_value"].agg(reducer)
            if len(monthly) >= 2:
                first_value, last_value = float(monthly.iloc[0]), float(monthly.iloc[-1])
                change = last_value - first_value
                percent_text = (
                    "percentage change unavailable because the first value is zero"
                    if first_value == 0 else f"a {change / abs(first_value) * 100:+.1f}% change"
                )
                first_label, last_label = month_name[int(monthly.index[0])], month_name[int(monthly.index[-1])]
                insights.append({"kind": "info", "text": f"From {first_label} to {last_label} in calendar-month order, {aggregation.lower()} {measure!r} changed by {change:+,.2f} ({percent_text}). Assumption: month labels have no year, so repeated month names are combined into one calendar-month cycle."})
            else:
                insights.append({"kind": "info", "text": "A month-based first-to-last change is not shown: at least two distinct valid month categories with numeric values are required."})
        else:
            insights.append({"kind": "info", "text": "A first-to-last change is not calculated because no date field was selected and no selected month-name dimension was available. Row order alone is not treated as a meaningful time order."})

    numeric = _coerce_finite_numeric(frame[measure]).dropna()
    if len(numeric) >= 4:
        q1, q3 = numeric.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        outlier_count = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())
        insights.append({"kind": "warning" if outlier_count else "info", "text": f"Potential outliers by the 1.5×IQR rule: {outlier_count} of {len(numeric)} valid values. This is a screening flag, not proof of an error."})
    else:
        insights.append({"kind": "info", "text": f"Outlier screening is limited: {len(numeric)} valid values; at least 4 are used for the IQR check."})

    if summary["missing_count"] or summary["nonnumeric_count"]:
        insights.append({"kind": "warning", "text": f"Data quality for {measure!r}: {summary['missing_count']} missing and {summary['nonnumeric_count']} nonnumeric value(s) are excluded from its numeric calculations."})
    return insights


def build_html_report(
    frame: pd.DataFrame,
    quality: pd.DataFrame,
    summaries: Mapping[str, Mapping[str, Any]],
    insights: list[dict[str, str]],
    figure: Any | None = None,
    title: str = "Business analysis report",
    grouped_data: pd.DataFrame | None = None,
    aggregation: str | None = None,
    *,
    dataset_name: str = "Uploaded dataset",
    scope: str = "Current filtered analysis",
    management_summary: Mapping[str, Any] | None = None,
    performance: Mapping[str, Any] | None = None,
) -> str:
    """Create a standalone HTML report, including an interactive chart when supplied."""
    sensitive_tokens = ("customer", "client", "consumer", "email", "phone", "address", "contact", "user", "person", "name")
    sensitive_grouping = grouped_data is not None and any(
        any(token in re.sub(r"[^a-z0-9]+", " ", str(column).casefold()) for token in sensitive_tokens)
        for column in grouped_data.columns
    )
    metric_rows = []
    for measure, metrics in summaries.items():
        metric_rows.append("<h3>" + escape(str(measure)) + "</h3><table>" + "".join(
            f"<tr><th>{escape(label.replace('_', ' ').title())}</th><td>{escape('—' if value is None else f'{value:,.2f}' if isinstance(value, (float, int)) else str(value))}</td></tr>"
            for label, value in metrics.items()
        ) + "</table>")
    insight_items = "".join(f"<li><strong>{escape(item['kind'].title())}:</strong> {escape(item['text'])}</li>" for item in insights) if not sensitive_grouping else "<li>Selected customer-level grouped findings are omitted from the shareable report to reduce exposure of sensitive data.</li>"
    chart_html = figure.to_html(full_html=False, include_plotlyjs=True) if figure is not None and not sensitive_grouping else "<p>No chart was selected.</p>" if figure is None else "<p>The selected chart is omitted because it groups at a customer or personal-data level.</p>"
    grouped_section = ""
    if sensitive_grouping:
        grouped_section = "<h2>Grouped comparison</h2><p>Omitted from this shareable report because the selected dimension may identify customers or people.</p>"
    elif grouped_data is not None and not grouped_data.empty:
        label = f"Grouped comparison ({escape(aggregation)})" if aggregation else "Grouped comparison"
        grouped_section = f"<h2>{label}</h2>{grouped_data.to_html(index=False, escape=True, border=0, justify='left', float_format=lambda value: f'{value:,.2f}')}"
    report_sections = ""
    if management_summary is not None:
        recommendations = "".join(f"<li>{escape(str(item))}</li>" for item in management_summary.get("recommendations", []))
        caveats = "".join(f"<li>{escape(str(item))}</li>" for item in management_summary.get("caveats", []))
        report_sections += f"<h2>Management summary</h2><p>{escape(str(management_summary.get('summary', '')))}</p>"
        if recommendations:
            report_sections += f"<h3>Prioritized recommendations</h3><ol>{recommendations}</ol>"
        if caveats:
            report_sections += f"<h3>Limitations and caveats</h3><ul>{caveats}</ul>"
    if performance is not None:
        performance_metric_rows = []
        metrics = performance.get("metrics", {})
        for role, label in (("revenue", "Total revenue"), ("cost", "Total cost"), ("profit", "Total profit"), ("profit_margin", "Aggregate profit margin"), ("units", "Units sold")):
            metric = metrics.get(role)
            value = metric.get("total") if isinstance(metric, Mapping) else metric
            if value is not None:
                formatted = f"{value:,.2%}" if role == "profit_margin" else f"{value:,.2f}"
                performance_metric_rows.append(f"<tr><th>{escape(label)}</th><td>{escape(formatted)}</td></tr>")
        performance_metric_rows.append(f"<tr><th>Records analyzed</th><td>{int(performance['columns']['rows']):,}</td></tr>")
        report_sections += "<h2>Business performance</h2><table>" + "".join(performance_metric_rows) + "</table>"
        report_sections += f"<p>Duplicate policy: {escape(str(performance['columns']['duplicate_policy']))}. "
        report_sections += f"{int(performance['columns']['duplicate_rows']):,} later repeated identifier row(s) detected in the source dataset.</p>"
        for role, table in performance.get("breakdowns", {}).items():
            if table is not None and not table.empty:
                report_sections += f"<h3>Profitability breakdown by {escape(role.title())}</h3>"
                report_sections += table.to_html(index=False, escape=True, border=0, justify="left", float_format=lambda value: f"{value:,.2f}", formatters={"profit_margin": lambda value: f"{value:.1%}"} if "profit_margin" in table else None)
        time_table = performance.get("time_series")
        if time_table is not None and not time_table.empty:
            report_sections += "<h3>Performance over time (monthly)</h3>" + time_table.to_html(index=False, escape=True, border=0, justify="left", float_format=lambda value: f"{value:,.2f}", formatters={"profit_margin": lambda value: f"{value:.1%}"} if "profit_margin" in time_table else None)
        business_insights = performance.get("business_insights", [])
        if business_insights:
            report_sections += "<h2>Business insights</h2><div class='insights'>" + "".join(
                "<article><h3>" + escape(item["headline"]) + "</h3><p><strong>Finding:</strong> " + escape(item["finding"]) +
                "</p><p><strong>Comparison:</strong> " + escape(item["comparison"]) + "</p><p><strong>Implication:</strong> " +
                escape(item["implication"]) + "</p><p class='meta'><strong>Limit:</strong> " + escape(item["limitation"]) + "</p></article>"
                for item in business_insights
            ) + "</div>"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>body{{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172033;line-height:1.55}}h1,h2{{color:#183b56}}h2{{margin-top:2rem}}article{{border:1px solid #d9e2ec;padding:1rem;margin:1rem 0;border-radius:.5rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}td,th{{border:1px solid #d9e2ec;padding:.55rem;text-align:left;vertical-align:top}}th{{background:#f0f4f8}}li{{margin:.5rem 0}}.meta{{color:#52606d}}@media print{{body{{max-width:none}}}}</style></head>
<body><h1>{escape(title)}</h1><p class="meta">Dataset: {escape(dataset_name)} · {escape(scope)} · Generated from the current filtered analysis at {generated}<br>{len(frame):,} rows · {len(frame.columns):,} columns</p>
{report_sections}<h2>Existing analysis overview</h2><h3>Selected measure summaries</h3>{''.join(metric_rows)}<h3>Selected analysis findings</h3><ul>{insight_items}</ul>
{grouped_section}<h2>Data quality</h2>{quality.to_html(index=False, escape=True, border=0, justify='left')}<h2>Visualization</h2>{chart_html}
<h2>Methodology</h2><p>Financial totals sum finite numeric values. Profit margin is aggregate profit divided by aggregate revenue over rows with both values available. Group-level margins use the same ratio of group totals. Missing and nonnumeric values are excluded from the affected metric; zero revenue yields an unavailable margin. Uploaded rows are not modified. Duplicate identifiers are included unless the report explicitly states that first-occurrence deduplication was selected.</p>
</body></html>"""


def dataframe_to_excel(
    frame: pd.DataFrame,
    quality: pd.DataFrame | None = None,
    grouped_data: pd.DataFrame | None = None,
) -> bytes:
    """Serialize filtered source data and optional analysis tables in memory."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Filtered data")
        if quality is not None:
            quality.to_excel(writer, index=False, sheet_name="Data quality")
        if grouped_data is not None and not grouped_data.empty:
            grouped_data.to_excel(writer, index=False, sheet_name="Grouped analysis")
        from openpyxl.styles import Alignment, Font, PatternFill
        from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

        for sheet_name, worksheet in writer.sheets.items():
            source = frame if sheet_name == "Filtered data" else quality if sheet_name == "Data quality" else grouped_data
            if source is None:
                continue
            for column_index, column in enumerate(source.columns, start=1):
                series = source[column]
                numeric = is_numeric_dtype(series.dtype)
                date = is_datetime64_any_dtype(series.dtype)
                alignment = "left"
                header = worksheet.cell(1, column_index)
                header.fill = PatternFill("solid", fgColor="E8EEF3")
                header.font = Font(bold=True, color="17324D")
                header.alignment = Alignment(horizontal=alignment, vertical="center")
                for row_index in range(2, len(source) + 2):
                    cell = worksheet.cell(row_index, column_index)
                    cell.alignment = Alignment(horizontal=alignment, vertical="center")
                    if date and cell.value is not None:
                        cell.number_format = "yyyy-mm-dd"
                    elif numeric and cell.value is not None:
                        cell.number_format = "#,##0" if is_integer_dtype(series.dtype) else "#,##0.00"
                content_width = series.astype("string").str.len().max()
                content_width = int(content_width) if pd.notna(content_width) else 0
                width = min(max(len(str(column)) + 2, content_width + 2), 48)
                worksheet.column_dimensions[worksheet.cell(1, column_index).column_letter].width = width
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
    return output.getvalue()

