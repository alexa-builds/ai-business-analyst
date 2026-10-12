"""Interactive Streamlit interface for the business data analyst."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    AGGREGATIONS,
    SUPPORTED_EXTENSIONS,
    aggregate_data,
    apply_filters,
    build_business_insights,
    build_html_report,
    coerce_numeric,
    dataframe_to_excel,
    calculate_business_performance,
    detect_business_columns,
    exclude_rows_with_missing_values,
    generate_business_insights,
    generate_management_summary,
    get_excel_sheet_names,
    load_dataset,
    month_number,
    numeric_summary,
    profile_dataset,
    remove_duplicate_identifier_rows,
)
from business_analyst import (
    answer_business_question,
    build_question_examples,
    generate_executive_summary,
)
from charts import build_category_bar_chart, build_category_pie_chart, build_distribution_histogram, style_figure
from ollama_provider import OllamaUnavailable, UnsafeInterpretation, generate_interpretation, list_local_models
from presentation import business_table_for_display, dataframe_column_config


st.set_page_config(page_title="Business Data Analyst", page_icon=":material/monitoring:", layout="wide")
st.title("Business Data Analyst", icon=":material/monitoring:")
st.caption("A focused workspace to validate business data, compare performance, and share clear findings.")


def format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def render_metric_row(items: list[dict[str, str]]) -> None:
    """Render aligned, responsive KPI cards using Streamlit's native layout."""
    with st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="top", gap="small", wrap=True):
        for item in items:
            st.metric(
                item["label"],
                item["value"],
                delta_description=item.get("description"),
                width="stretch",
            )


def category_filter_widget(frame: pd.DataFrame, profiles: dict[str, dict], widget_epoch: int) -> dict[str, list[object]]:
    filters: dict[str, list[object]] = {}
    eligible = []
    for column in frame.columns:
        profile = profiles[str(column)]
        values = frame.loc[~frame[column].isna(), column].drop_duplicates().tolist()
        if frame[column].map(lambda value: isinstance(value, str) and not value.strip()).any():
            values = [value for value in values if not (isinstance(value, str) and not value.strip())]
        if not profile["is_numeric"] and not profile["is_date"] and len(values) <= 80:
            eligible.append((column, values))
    if not eligible:
        st.sidebar.caption("No low-cardinality category columns are available for filtering.")
        return filters
    with st.sidebar.expander("Category filters", expanded=False):
        for column, values in eligible:
            has_missing = frame[column].isna().any() or frame[column].map(lambda value: isinstance(value, str) and not value.strip()).any()
            options = values + ([None] if has_missing else [])
            # A keyed Streamlit widget may retain a selection across reruns even
            # after its options change. Tie its identity to both the active
            # option domain and an explicit reset/scope epoch so the browser
            # creates a fresh widget when either changes.
            option_signature = "\x1f".join(f"{type(value).__name__}:{value!r}" for value in options)
            option_version = sha256(option_signature.encode("utf-8")).hexdigest()[:10]
            selected = st.multiselect(
                str(column),
                options=options,
                default=options,
                format_func=lambda value: "(Missing)" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value),
                key=f"category-filter-{column}-{widget_epoch}-{option_version}",
            )
            filters[str(column)] = selected
    return filters


def clear_category_filter_state() -> None:
    """Create fresh category widgets whenever a row-scope control changes."""
    for state_key in list(st.session_state):
        if state_key.startswith("category-filter-"):
            st.session_state.pop(state_key, None)
    st.session_state["_category_filter_epoch"] = int(st.session_state.get("_category_filter_epoch", 0)) + 1


def numeric_filter_widget(frame: pd.DataFrame, profiles: dict[str, dict]) -> dict[str, tuple[float, float]]:
    filters: dict[str, tuple[float, float]] = {}
    columns = [column for column in frame.columns if profiles[str(column)]["is_numeric"]]
    with st.sidebar.expander("Numeric ranges", expanded=False):
        for column in columns:
            numeric = coerce_numeric(frame[column]).dropna()
            if numeric.empty or numeric.min() == numeric.max():
                continue
            low, high = float(numeric.min()), float(numeric.max())
            filters[str(column)] = st.slider(
                str(column), min_value=low, max_value=high, value=(low, high), key=f"numeric-filter-{column}"
            )
        if columns and not filters:
            st.caption("Range controls need at least two distinct numeric values.")
    return filters


def date_filter_widget(frame: pd.DataFrame, profiles: dict[str, dict]) -> dict[str, tuple[date, date]]:
    filters: dict[str, tuple[date, date]] = {}
    columns = [
        column for column in frame.columns
        if profiles[str(column)]["is_date"] and profiles[str(column)]["valid_date_count"] > 0
    ]
    with st.sidebar.expander("Date ranges", expanded=False):
        for column in columns:
            parsed = profiles[str(column)]["dates"].dropna()
            if parsed.empty:
                continue
            min_date, max_date = parsed.min().date(), parsed.max().date()
            if min_date == max_date:
                continue
            selected = st.date_input(
                str(column), value=(min_date, max_date), min_value=min_date, max_value=max_date,
                key=f"date-filter-{column}",
            )
            if isinstance(selected, tuple) and len(selected) == 2:
                filters[str(column)] = selected
    return filters


def make_category_labels(frame: pd.DataFrame, dimensions: list[str]) -> pd.Series:
    if not dimensions:
        return pd.Series("All records", index=frame.index, dtype="string")
    pieces = [frame[column].astype("string").fillna("(Missing)") for column in dimensions]
    labels = pieces[0]
    for piece in pieces[1:]:
        labels = labels + " · " + piece
    return labels


def render_downloads(frame: pd.DataFrame, quality: pd.DataFrame, summaries: dict, insights: list, figure=None, grouped_data=None, aggregation=None, *, dataset_name="Uploaded dataset", scope="Current filtered analysis", management_summary=None, performance=None) -> None:
    csv_bytes = frame.to_csv(index=False).encode("utf-8-sig")
    excel_bytes = dataframe_to_excel(frame, quality, grouped_data)
    report_html = build_html_report(frame, quality, summaries, insights, figure, grouped_data=grouped_data, aggregation=aggregation, dataset_name=dataset_name, scope=scope, management_summary=management_summary, performance=performance)
    has_grouped_data = grouped_data is not None and not grouped_data.empty
    with st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center", gap="small", wrap=True):
        st.download_button("Download filtered CSV", csv_bytes, file_name="business-analysis-filtered.csv", mime="text/csv", icon=":material/download:", width="content")
        st.download_button("Download filtered Excel", excel_bytes, file_name="business-analysis-filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:", width="content")
        st.download_button("Download management HTML report", report_html.encode("utf-8"), file_name="business-management-report.html", mime="text/html", icon=":material/description:", width="content")
        if has_grouped_data:
            grouped_csv = grouped_data.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download grouped CSV", grouped_csv, file_name=f"business-analysis-{str(aggregation).lower()}-grouped.csv", mime="text/csv", icon=":material/table_view:", width="content")


def render_ai_answer(result: dict) -> None:
    st.subheader("Python-calculated answer")
    st.markdown(f"**Main finding**  {result['headline']}")
    if result["facts"]:
        render_metric_row([
            {"label": str(fact["label"]), "value": str(fact["value"])}
            for fact in result["facts"][:6]
        ])
    if result.get("calculation"):
        st.markdown("**How this was calculated**")
        st.write(result["calculation"])
    if result.get("limitations"):
        st.markdown("**Limitations**")
        for limitation in result["limitations"]:
            st.markdown(f"- {limitation}")
    if result.get("model_interpretation"):
        st.markdown("**Optional local AI interpretation**")
        st.write(result["model_interpretation"])
        st.caption("This qualitative wording is generated locally. Python-calculated figures above remain the factual source of truth.")
    elif result.get("model_error"):
        st.caption(result["model_error"])
    if result.get("follow_ups"):
        st.markdown("**Suggested follow-up questions**")
        for follow_up in result["follow_ups"]:
            st.markdown(f"- {follow_up}")


def render_ai_analyst(
    frame: pd.DataFrame,
    upload_signature: str,
    sheet_name: str | None,
    category_filters: dict,
    numeric_ranges: dict,
    date_ranges: dict,
    selected_measures: list,
    selected_dimensions: list,
    aggregation: str,
    date_column: str | None,
    date_granularity: str,
    source_data: pd.DataFrame | None = None,
    analysis_scope: str = "all uploaded rows",
    exclude_missing_values: bool = False,
) -> None:
    """Render calculated findings, question answers, and optional local wording."""
    st.header("AI Business Analyst", icon=":material/psychology:")
    st.caption("Ask a question about the current filtered rows. Python calculates the answer; an optional local model can add a short qualitative explanation.")
    st.caption(f"Analysis scope: {analysis_scope}. Answers use the same rows as the overview, exploration, and report; duplicate ID counts inspect the uploaded source to report source-level repeats.")

    examples = build_question_examples(frame, selected_measures, selected_dimensions)
    if examples:
        example_options = ["Choose an example question"] + examples
        chosen_example = st.selectbox("Example questions", example_options, key="ai-example-question")
        previously_loaded = st.session_state.get("_ai_loaded_example")
        if chosen_example != example_options[0] and chosen_example != previously_loaded:
            st.session_state["ai-question"] = chosen_example
            st.session_state["_ai_loaded_example"] = chosen_example
        elif chosen_example == example_options[0]:
            st.session_state.pop("_ai_loaded_example", None)

    use_ollama = st.checkbox(
        "Add optional interpretation from a local Ollama model",
        key="ai-use-local-model",
        help="Only loopback Ollama models already installed on this computer are offered. No model is downloaded by this app.",
    )
    local_models = []
    local_model_error = None
    if use_ollama:
        try:
            local_models = list_local_models()
        except OllamaUnavailable as exc:
            local_model_error = str(exc)
        if local_models:
            selected_model = st.selectbox("Installed local model", local_models, key="ai-local-model")
        else:
            selected_model = None
            if local_model_error:
                st.info(f"{local_model_error} Start Ollama locally if you want AI wording; Python calculations remain available.")
            else:
                st.caption("No eligible local models were found. Install and start Ollama separately, then add a model locally to enable this option.")
    else:
        selected_model = None

    with st.form("ai-question-form"):
        question = st.text_input(
            "Ask about this dataset",
            key="ai-question",
            placeholder="For example: Which group has the highest total?",
        )
        submitted = st.form_submit_button("Analyze question", type="primary", icon=":material/search:", key="ai-question-submit")

    def context_signature(prompt: str) -> str:
        context = {
            "upload": upload_signature,
            "sheet": sheet_name,
            "filters": [category_filters, numeric_ranges, date_ranges],
            "measures": selected_measures,
            "dimensions": selected_dimensions,
            "aggregation": aggregation,
            "date_column": date_column,
            "date_granularity": date_granularity,
            "question": prompt,
            "scope": analysis_scope,
        }
        return sha256(json.dumps(context, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def save_result(result: dict, kind: str, prompt: str) -> None:
        if use_ollama and selected_model and result["status"] == "answered":
            try:
                with st.spinner("Preparing a local qualitative interpretation…"):
                    result["model_interpretation"] = generate_interpretation(selected_model, result)
            except UnsafeInterpretation:
                result["model_error"] = "The local model response was omitted because it introduced a number or causal claim outside the validated Python findings."
            except OllamaUnavailable as exc:
                result["model_error"] = f"{exc} The Python-calculated answer is still available."
        st.session_state["_ai_result"] = result
        st.session_state["_ai_request_kind"] = kind
        st.session_state["_ai_context_signature"] = context_signature(prompt)

    if submitted:
        answer = answer_business_question(
            frame,
            question,
            selected_measures,
            selected_dimensions,
            aggregation,
            date_column,
            date_granularity,
            source_frame=source_data,
            category_filters=category_filters,
            numeric_ranges=numeric_ranges,
            date_ranges=date_ranges,
            exclude_missing_values=exclude_missing_values,
        )
        save_result(answer, "question", question)

    if st.button("Generate executive summary", key="ai-executive-summary", icon=":material/summarize:"):
        summary_question = "Summarize the main findings for a management meeting."
        summary = generate_executive_summary(
            frame,
            selected_measures,
            selected_dimensions,
            aggregation,
            date_column,
            date_granularity,
            summary_question,
        )
        save_result(summary, "summary", summary_question)

    stored_result = st.session_state.get("_ai_result")
    stored_kind = st.session_state.get("_ai_request_kind")
    active_prompt = "Summarize the main findings for a management meeting." if stored_kind == "summary" else question if examples else ""
    if stored_result:
        if st.session_state.get("_ai_context_signature") == context_signature(active_prompt):
            render_ai_answer(stored_result)
        else:
            st.caption("The previous answer used different filters or analysis settings. Run it again to refresh the result.")


st.subheader("Start with a dataset", anchor=False)
st.caption("Upload a CSV or Excel workbook. Your file stays in this local app session.")
uploaded = st.file_uploader("Choose a data file", type=sorted(SUPPORTED_EXTENSIONS), key="data_upload", help="Supported formats: CSV, XLS, and XLSX.")
if uploaded is None:
    for state_key in ("_upload_signature", "_workbook_sheets", "_workbook_signature", "_dataset_signature", "_uploaded_dataframe", "_ai_result", "_ai_request_kind", "_ai_context_signature", "_ai_loaded_example", "ai-question", "exclude-missing-values", "analysis-deduplicate-ids"):
        st.session_state.pop(state_key, None)
    st.info("Choose a CSV, XLS, or XLSX file to see the overview, data quality, and analysis.", icon=":material/upload_file:")
    st.stop()

content = uploaded.getvalue()
extension = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
if extension not in SUPPORTED_EXTENSIONS:
    st.error("Unsupported file type. Upload a CSV, XLS, or XLSX file.")
    st.stop()

upload_signature = f"{uploaded.name}:{sha256(content).hexdigest()}"
if st.session_state.get("_upload_signature") != upload_signature:
    for state_key in ("_workbook_sheets", "_workbook_signature", "_dataset_signature", "_uploaded_dataframe", "_ai_result", "_ai_request_kind", "_ai_context_signature", "_ai_loaded_example", "ai-question"):
        st.session_state.pop(state_key, None)
    for state_key in list(st.session_state):
        if state_key.startswith(("category-filter-", "numeric-filter-", "date-filter-")):
            st.session_state.pop(state_key, None)
    clear_category_filter_state()
    st.session_state["analysis-deduplicate-ids"] = False
    st.session_state["exclude-missing-values"] = False
    st.session_state["_upload_signature"] = upload_signature

sheet_name = None
if extension in {"xls", "xlsx"}:
    if st.session_state.get("_workbook_signature") == upload_signature:
        sheets = st.session_state["_workbook_sheets"]
    else:
        try:
            sheets = get_excel_sheet_names(content, extension)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["_workbook_signature"] = upload_signature
        st.session_state["_workbook_sheets"] = sheets
    if len(sheets) > 1:
        sheet_name = st.selectbox("Worksheet", sheets, key=f"worksheet-{upload_signature}")
    elif sheets:
        sheet_name = sheets[0]

dataset_signature = (upload_signature, sheet_name)
if st.session_state.get("_dataset_signature") != dataset_signature:
    for state_key in (
        "selected-measures", "group-dimensions", "analysis-date-column", "date-granularity",
        "group-aggregation", "chart-type", "chart-measure", "scatter-x", "scatter-y",
        "performance-breakdown", "ai-example-question", "_ai_loaded_example", "_ai_result",
        "_ai_request_kind", "_ai_context_signature", "ai-question",
    ):
        st.session_state.pop(state_key, None)
    for state_key in list(st.session_state):
        if state_key.startswith(("category-filter-", "numeric-filter-", "date-filter-")):
            st.session_state.pop(state_key, None)
    clear_category_filter_state()
    st.session_state["analysis-deduplicate-ids"] = False
    st.session_state["exclude-missing-values"] = False
if st.session_state.get("_dataset_signature") == dataset_signature:
    original_data = st.session_state["_uploaded_dataframe"]
else:
    try:
        original_data = load_dataset(content, uploaded.name, sheet_name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["_dataset_signature"] = dataset_signature
    st.session_state["_uploaded_dataframe"] = original_data

if len(original_data.columns) == 0:
    st.error("The selected sheet does not contain any columns.")
    st.stop()

full_quality, full_profiles = profile_dataset(original_data)
duplicate_count = int(original_data.duplicated().sum())
business_columns = detect_business_columns(original_data)
identifier_column = business_columns.get("identifier")
identifier_duplicate_count = int(business_columns.get("duplicate_rows", 0))
numeric_columns = [column for column in original_data.columns if full_profiles[str(column)]["is_numeric"]]
date_columns = [column for column in original_data.columns if full_profiles[str(column)]["is_date"] and full_profiles[str(column)]["valid_date_count"] > 0]

def clear_analysis_filters() -> None:
    for state_key in list(st.session_state):
        if state_key.startswith(("category-filter-", "numeric-filter-", "date-filter-")):
            st.session_state.pop(state_key, None)
    st.session_state["analysis-deduplicate-ids"] = False
    st.session_state["exclude-missing-values"] = False
    st.session_state["_category_filter_epoch"] = int(st.session_state.get("_category_filter_epoch", 0)) + 1
    for state_key in ("_ai_result", "_ai_request_kind", "_ai_context_signature", "_ai_loaded_example"):
        st.session_state.pop(state_key, None)


st.sidebar.header("Filters")
st.sidebar.button("Reset filters", key="reset-analysis-filters", on_click=clear_analysis_filters, width="stretch")
exclude_missing = st.sidebar.checkbox(
    "Exclude rows with missing values",
    key="exclude-missing-values",
    on_change=clear_category_filter_state,
    help="When enabled, exclude a row if any cell is null, blank, or whitespace-only. Zero is retained. The uploaded source remains unchanged and visible in Data & Quality.",
)
if identifier_column is not None and identifier_duplicate_count:
    remove_duplicate_ids = st.sidebar.checkbox(
        f"Keep first {identifier_column} occurrence",
        key="analysis-deduplicate-ids",
        on_change=clear_category_filter_state,
        help="Off by default: all rows are included. When enabled, the first nonblank ID row is retained before applying the filters. The source file is never changed.",
    )
else:
    remove_duplicate_ids = False
deduplicated_source = remove_duplicate_identifier_rows(original_data, identifier_column) if remove_duplicate_ids and identifier_column is not None else original_data.copy()
duplicate_rows_excluded = len(original_data) - len(deduplicated_source)
if exclude_missing:
    analysis_source, missing_rows_excluded = exclude_rows_with_missing_values(deduplicated_source)
else:
    analysis_source, missing_rows_excluded = deduplicated_source, 0
category_filters = category_filter_widget(
    analysis_source, full_profiles, int(st.session_state.get("_category_filter_epoch", 0))
)
numeric_ranges = numeric_filter_widget(original_data, full_profiles)
date_ranges = date_filter_widget(original_data, full_profiles)
duplicate_policy = "first identifier occurrence retained" if remove_duplicate_ids else "all uploaded rows included"
filtered_data = apply_filters(analysis_source, category_filters, numeric_ranges, date_ranges)
missing_policy = "any null, blank, or whitespace-only cell excluded" if exclude_missing else "missing-value exclusion off"
st.sidebar.caption(f"{len(filtered_data):,} of {len(original_data):,} uploaded rows analyzed; {duplicate_rows_excluded:,} duplicate row(s) and {missing_rows_excluded:,} missing-value row(s) excluded. Duplicate policy: {duplicate_policy}; {missing_policy}.")
active_filter_names = []
for column, selected_values in category_filters.items():
    all_values = analysis_source.loc[~analysis_source[column].isna(), column].drop_duplicates().tolist()
    has_blank = analysis_source[column].map(lambda value: isinstance(value, str) and not value.strip()).any()
    if has_blank:
        all_values = [value for value in all_values if not (isinstance(value, str) and not value.strip())]
    has_missing = analysis_source[column].isna().any() or has_blank
    available_options = all_values + ([None] if has_missing else [])
    if len(selected_values) < len(available_options):
        active_filter_names.append(f"{column}: {len(selected_values)} selected")
for column, (low, high) in numeric_ranges.items():
    values = coerce_numeric(original_data[column]).dropna()
    if not values.empty and (low > float(values.min()) or high < float(values.max())):
        active_filter_names.append(f"{column}: {low:,.3g}–{high:,.3g}")
for column, (start, end) in date_ranges.items():
    parsed_dates = full_profiles[str(column)]["dates"].dropna()
    if not parsed_dates.empty and (start > parsed_dates.min().date() or end < parsed_dates.max().date()):
        active_filter_names.append(f"{column}: {start}–{end}")
st.sidebar.caption("Active filters: " + ("; ".join(active_filter_names) if active_filter_names else "None"))
applied_row_rules = []
if exclude_missing:
    applied_row_rules.append(f"Missing values: {missing_rows_excluded:,} row(s) excluded")
if remove_duplicate_ids:
    applied_row_rules.append(f"Duplicate IDs: {duplicate_rows_excluded:,} later row(s) excluded")
st.sidebar.caption("Applied row rules: " + ("; ".join(applied_row_rules) if applied_row_rules else "No rows removed by row rules."))

filtered_quality, filtered_profiles = profile_dataset(filtered_data)
performance = calculate_business_performance(filtered_data, identifier=identifier_column)
if remove_duplicate_ids and identifier_column is not None:
    performance["columns"]["duplicate_rows"] = identifier_duplicate_count
    performance["columns"]["duplicate_ids"] = business_columns.get("duplicate_ids", [])
    performance["columns"]["input_rows"] = len(original_data)
    performance["columns"]["duplicate_policy"] = duplicate_policy
    performance["columns"]["duplicate_scope"] = "uploaded source before filters"
elif identifier_column is None:
    performance["columns"]["duplicate_policy"] = duplicate_policy
business_insights = generate_business_insights(performance)
management_summary = generate_management_summary(performance, business_insights)
scope_description = (
    f"{len(filtered_data):,} filtered rows from {len(original_data):,} uploaded rows; "
    f"{duplicate_policy} ({duplicate_rows_excluded:,} duplicate rows excluded); "
    f"{missing_policy} ({missing_rows_excluded:,} rows excluded)"
)

selected_measures: list = []
group_dimensions: list = []
aggregation = "Sum"
granularity = "Month"
date_column_value = None
summaries: dict = {}
grouped_results = None
figure = None
insights: list = []
primary_measure = None

overview_tab, data_tab, explore_tab, performance_tab, analyst_tab, report_tab = st.tabs([
    "Overview", "Data & Quality", "Explore Data", "Business Performance", "AI Analyst", "Management Report"
])

with overview_tab:
    st.subheader("Current analysis")
    overview_items = [{"label": "Rows analyzed", "value": f"{len(filtered_data):,}"}]
    for role, label in (("revenue", "Revenue"), ("profit", "Profit"), ("profit_margin", "Profit margin")):
        metric = performance["metrics"].get(role)
        value = metric.get("total") if isinstance(metric, dict) else metric
        if value is not None:
            overview_items.append({"label": label, "value": f"{value:,.1%}" if role == "profit_margin" else f"{value:,.2f}"})
    render_metric_row(overview_items)
    st.caption(f"Scope: {scope_description}. All sections, question answers, and exports use the same filtered analysis rows.")
    if filtered_data.empty:
        st.warning("No rows match these filters. Use Reset filters or adjust the filter values.")
    elif business_insights:
        st.markdown(f"**{business_insights[0]['headline']}** — {business_insights[0]['finding']}")
    total_missing = int(filtered_quality["Missing"].fillna(0).sum()) if "Missing" in filtered_quality else 0
    st.caption(f"Data check: {total_missing:,} missing cells across the filtered rows; {duplicate_count:,} exact duplicate source row(s).")

with data_tab:
    st.subheader("Uploaded data")
    st.caption(f"Showing up to 100 source rows from {uploaded.name}. The source is not mutated by analysis or duplicate handling.")
    st.dataframe(original_data.head(100), hide_index=True, column_config=dataframe_column_config(original_data, full_profiles), alt="Preview of uploaded data")
    if len(original_data) > 100:
        st.caption(f"Preview limited to 100 of {len(original_data):,} source rows.")
    with st.expander("Full-file quality profile"):
        st.dataframe(full_quality, hide_index=True, column_config=dataframe_column_config(full_quality, full_profiles), alt="Types, missing values, and parse issues for uploaded data")
    with st.expander("Filtered analysis quality profile"):
        st.dataframe(filtered_quality, hide_index=True, column_config=dataframe_column_config(filtered_quality, filtered_profiles), alt="Types, missing values, and parse issues for filtered analysis rows")
    if duplicate_count or identifier_duplicate_count:
        st.info(f"Source audit: {duplicate_count:,} exact duplicate row(s); {identifier_duplicate_count:,} later repeated identifier row(s) in {identifier_column!s}. The selected policy is {duplicate_policy}.")

with explore_tab:
    st.subheader("Explore filtered data")
    if filtered_data.empty:
        st.info("No rows match the shared filters. Reset or adjust the filters to explore data.")
    elif not numeric_columns:
        st.info("No numeric measures were detected. Data preview, quality checks, AI question support, and exports remain available.")
    else:
        selected_measures = st.multiselect("Numeric measures", numeric_columns, default=[numeric_columns[0]], key="selected-measures")
        remaining_columns = [
            column for column in filtered_data.columns
            if column not in selected_measures and column not in date_columns and not full_profiles[str(column)]["is_numeric"]
        ]
        with st.container(horizontal_alignment="left", vertical_alignment="top", gap="small"):
            semantic_dimensions = [business_columns.get(role) for role in ("product", "category", "region", "channel")]
            default_dimensions = [
                column for column in semantic_dimensions
                if column in remaining_columns
            ][:1]
            if not default_dimensions:
                low_cardinality = [
                    column for column in remaining_columns
                    if column != identifier_column and original_data[column].nunique(dropna=True) <= 80
                ]
                default_dimensions = low_cardinality[:1]
            group_dimensions = st.multiselect("Group by categories (optional)", remaining_columns, default=default_dimensions, key="group-dimensions")
            date_choices = ["None"] + date_columns
            prior_date = st.session_state.get("analysis-date-column", "None")
            date_default = prior_date if prior_date in date_choices else "None"
            date_column = st.selectbox("Date field (optional)", date_choices, index=date_choices.index(date_default), key="analysis-date-column")
            granularity = st.selectbox("Date grouping", ["Day", "Week", "Month", "Quarter", "Year"], index=2, disabled=date_column == "None", key="date-granularity")
            aggregation = st.selectbox("Grouped calculation", list(AGGREGATIONS), key="group-aggregation", help="Count means valid numeric observations; other options aggregate valid numeric values.")
        if not selected_measures:
            st.info("Select at least one measure to display summary statistics and charts.")
        else:
            summaries = {str(measure): numeric_summary(filtered_data[measure].tolist()) for measure in selected_measures}
            primary_measure = selected_measures[0]
            primary_summary = summaries[str(primary_measure)]
            render_metric_row([
                {"label": "Valid values", "value": f"{primary_summary['count']:,}"},
                {"label": "Total", "value": format_value(primary_summary["sum"])},
                {"label": "Average", "value": format_value(primary_summary["mean"])},
                {"label": "Median", "value": format_value(primary_summary["median"])},
            ])
            summary_table = pd.DataFrame.from_dict(summaries, orient="index")
            summary_table.index.name = "Measure"
            summary_table = summary_table.reset_index().rename(columns={
                "count": "Count", "sum": "Sum", "mean": "Mean", "median": "Median", "minimum": "Minimum",
                "maximum": "Maximum", "standard_deviation": "Standard deviation", "missing_count": "Missing", "nonnumeric_count": "Nonnumeric",
            })
            with st.expander("Detailed measure statistics"):
                st.dataframe(summary_table, hide_index=True, column_config=dataframe_column_config(summary_table), alt="Descriptive statistics for selected measures")
            dimensions_for_chart = list(group_dimensions)
            month_dimensions = [column for column in dimensions_for_chart if full_profiles[str(column)]["is_month"]]
            date_column_value = date_column if date_column != "None" else None
            if dimensions_for_chart or date_column_value:
                grouped_results = aggregate_data(filtered_data, list(selected_measures), dimensions_for_chart, aggregation, date_column_value, granularity)
            chart_options = ["Histogram"]
            if dimensions_for_chart:
                chart_options.extend(["Bar", "Pie", "Box"])
            if date_column_value or month_dimensions:
                chart_options.append("Line")
            if len(numeric_columns) >= 2:
                chart_options.append("Scatter")
            default_chart = "Bar" if dimensions_for_chart else "Line" if date_column_value else "Histogram"
            if st.session_state.get("chart-type") not in chart_options:
                st.session_state.pop("chart-type", None)
            chart_type = st.selectbox("Chart type", chart_options, index=chart_options.index(default_chart), key="chart-type")
            chart_measure = st.selectbox("Chart measure", selected_measures, key="chart-measure")
            if chart_type == "Histogram" and dimensions_for_chart:
                st.caption("Histogram bins show observation counts; the grouped calculation applies to the separate grouped table.")
            chart_measure_count = numeric_summary(filtered_data[chart_measure].tolist())["count"]
            if chart_measure_count == 0 and chart_type != "Scatter":
                st.info(f"No valid observations are available for {chart_measure!r}.")
            elif chart_type == "Histogram":
                figure = build_distribution_histogram(filtered_data, str(chart_measure))
            elif chart_type == "Bar":
                figure = build_category_bar_chart(grouped_results, str(chart_measure), list(dimensions_for_chart), aggregation)
            elif chart_type == "Pie" and dimensions_for_chart:
                # A share chart always represents sums, independently of the
                # separate grouped table's selected calculation.
                pie_groups = aggregate_data(filtered_data, [str(chart_measure)], list(dimensions_for_chart), "Sum")
                figure, pie_note = build_category_pie_chart(pie_groups, str(chart_measure), list(dimensions_for_chart))
                if pie_note:
                    st.info(pie_note)
                st.caption(f"Category shares use summed {chart_measure} values from the current filtered rows; the grouped table continues to use {aggregation.lower()}.")
            elif chart_type == "Line" and date_column_value:
                grouped = grouped_results
                if grouped is not None and not grouped.empty and "Date" in grouped.columns:
                    color_col = None
                    category_names = [str(column) for column in dimensions_for_chart if str(column) in grouped.columns]
                    if category_names:
                        grouped["Series"] = make_category_labels(grouped, category_names)
                        color_col = "Series"
                    figure = style_figure(px.line(grouped, x="Date", y=str(chart_measure), color=color_col, markers=True, title=f"{aggregation} {chart_measure} over time", labels={"Date": str(date_column_value), str(chart_measure): f"{aggregation} {chart_measure}"}))
            elif chart_type == "Line" and month_dimensions:
                month_dimension = month_dimensions[0]
                grouped = grouped_results
                if grouped is not None and not grouped.empty:
                    grouped["_month_order"] = grouped[month_dimension].map(month_number)
                    grouped = grouped.dropna(subset=["_month_order", str(chart_measure)]).sort_values("_month_order", kind="stable")
                    category_names = [str(column) for column in dimensions_for_chart if str(column) != str(month_dimension)]
                    if category_names:
                        grouped["Series"] = make_category_labels(grouped, category_names)
                    month_labels = grouped[month_dimension].drop_duplicates().tolist()
                    figure = style_figure(px.line(grouped, x=month_dimension, y=str(chart_measure), color="Series" if category_names else None, markers=True, title=f"{aggregation} {chart_measure} by month", labels={str(month_dimension): str(month_dimension), str(chart_measure): f"{aggregation} {chart_measure}"}))
                    figure.update_xaxes(categoryorder="array", categoryarray=month_labels)
            elif chart_type == "Box" and dimensions_for_chart:
                chart_frame = filtered_data.copy()
                chart_frame["Category"] = make_category_labels(chart_frame, dimensions_for_chart)
                chart_frame["_analysis_value"] = coerce_numeric(chart_frame[chart_measure])
                figure = style_figure(px.box(chart_frame, x="Category", y="_analysis_value", points="outliers", title=f"{chart_measure} distribution by category", labels={"Category": "Category", "_analysis_value": str(chart_measure)}))
            elif chart_type == "Scatter":
                with st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="top", gap="small", wrap=True):
                    x_measure = st.selectbox("X measure", numeric_columns, key="scatter-x")
                    y_options = [column for column in numeric_columns if column != x_measure]
                    y_measure = st.selectbox("Y measure", y_options, key="scatter-y")
                chart_frame = filtered_data.copy()
                chart_frame["_analysis_x"] = coerce_numeric(chart_frame[x_measure])
                chart_frame["_analysis_y"] = coerce_numeric(chart_frame[y_measure])
                chart_frame = chart_frame.dropna(subset=["_analysis_x", "_analysis_y"])
                if not chart_frame.empty:
                    color = str(dimensions_for_chart[0]) if dimensions_for_chart else None
                    figure = style_figure(px.scatter(chart_frame, x="_analysis_x", y="_analysis_y", color=color, title=f"{x_measure} compared with {y_measure}", labels={"_analysis_x": str(x_measure), "_analysis_y": str(y_measure)}))
            if figure is None:
                st.info("The selected chart does not have enough valid data to display.")
            else:
                st.plotly_chart(figure, key="analysis-chart", width="stretch")
            if grouped_results is not None:
                valid_groups = grouped_results.dropna(subset=[str(primary_measure)])
                st.caption(f"Grouped {aggregation.lower()} uses {len(filtered_data):,} current analysis rows. Count excludes missing and invalid measure values.")
                st.dataframe(grouped_results.head(500), hide_index=True, column_config=dataframe_column_config(grouped_results), alt="Selected measure grouped by the active dimensions")

with performance_tab:
    st.subheader("Financial performance")
    st.caption(f"All calculations use {scope_description}. Margins use group profit divided by group revenue, not an average of row margins.")
    performance_items = [{"label": "Records", "value": f"{len(filtered_data):,}"}]
    for role, label in (("revenue", "Revenue"), ("cost", "Costs"), ("profit", "Profit"), ("profit_margin", "Profit margin")):
        metric = performance["metrics"].get(role)
        value = metric.get("total") if isinstance(metric, dict) else metric
        if value is not None:
            performance_items.append({"label": label, "value": f"{value:,.1%}" if role == "profit_margin" else f"{value:,.2f}"})
    render_metric_row(performance_items)
    details = []
    for role, label in (("revenue", "Average revenue / valid record"), ("profit", "Average profit / valid record"), ("units", "Units sold")):
        metric = performance["metrics"].get(role)
        value = metric.get("average") if isinstance(metric, dict) and role != "units" else metric.get("total") if isinstance(metric, dict) else None
        if value is not None:
            details.append(f"{label}: {value:,.2f}")
    if details:
        st.caption(" · ".join(details))
    if not any(performance["metrics"].get(role) for role in ("revenue", "cost", "profit")):
        st.info("No recognized financial fields are available. The app does not infer missing revenue, costs, or profit.")
    breakdown_options = [role for role in ("product", "category", "region", "channel") if role in performance["breakdowns"]]
    if breakdown_options:
        breakdown_role = st.selectbox("Performance breakdown", breakdown_options, format_func=lambda role: str(business_columns.get(role, role)).replace("_", " ").title(), key="performance-breakdown")
        breakdown = performance["breakdowns"][breakdown_role]
        breakdown_display = business_table_for_display(breakdown)
        st.dataframe(breakdown_display, hide_index=True, column_config=dataframe_column_config(breakdown_display), alt=f"Business performance by {breakdown_role}")
    elif not filtered_data.empty:
        st.info("No product, category, region, or channel dimensions were detected for a segment breakdown.")
    if not performance["time_series"].empty:
        with st.expander("Monthly performance details"):
            time_display = business_table_for_display(performance["time_series"])
            st.dataframe(time_display, hide_index=True, column_config=dataframe_column_config(time_display), alt="Monthly financial performance")
    st.subheader("Business insights")
    for item in business_insights[:4]:
        with st.container(border=True):
            st.markdown(f"**{item['headline']}**")
            st.write(item["finding"])
            st.caption(f"{item['comparison']} {item['limitation']}")
            st.write(f"Implication: {item['implication']}")
    if len(business_insights) > 4:
        with st.expander(f"More findings ({len(business_insights) - 4})"):
            for item in business_insights[4:]:
                st.markdown(f"**{item['headline']}** — {item['finding']}")
                st.caption(f"{item['comparison']} {item['limitation']}")

with analyst_tab:
    render_ai_analyst(
        filtered_data, upload_signature, sheet_name, category_filters, numeric_ranges, date_ranges,
        selected_measures, group_dimensions, aggregation, date_column_value, granularity, original_data, duplicate_policy, exclude_missing,
    )

with report_tab:
    st.subheader("Management summary")
    st.caption(f"Scope: {scope_description}. Report metrics, tables, chart, AI answers, and downloads use the same current analysis rows.")
    st.write(management_summary["summary"])
    if management_summary["recommendations"]:
        st.markdown("**Prioritized recommendations**")
        for recommendation in management_summary["recommendations"]:
            st.markdown(f"- {recommendation}")
    if management_summary["caveats"]:
        st.caption("Caveats: " + " ".join(management_summary["caveats"]))
    st.subheader("Download report and data")
    st.caption("CSV and Excel contain the same filtered analysis rows used by the other sections. The HTML report embeds the selected chart and excludes customer-level grouped content.")
    render_downloads(
        filtered_data, filtered_quality, summaries, insights, figure, grouped_results, aggregation,
        dataset_name=uploaded.name,
        scope=scope_description,
        management_summary=management_summary,
        performance={**performance, "business_insights": business_insights},
    )
