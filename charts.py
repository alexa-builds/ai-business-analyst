"""Plotly chart builders shared by the app and focused unit tests."""

from __future__ import annotations

from math import ceil, sqrt

import pandas as pd
import plotly.express as px

from analysis import coerce_numeric


COLOR_SEQUENCE = ["#137C78", "#345B75", "#6A8B86", "#B28A52", "#7A6E9B", "#A85C52"]


def style_figure(figure):
    """Apply the shared, low-contrast chart style."""
    figure.update_layout(
        template="plotly_white",
        colorway=COLOR_SEQUENCE,
        font={"family": "Arial, sans-serif", "color": "#183247", "size": 13},
        title={"x": 0, "xanchor": "left", "font": {"size": 19}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin={"l": 16, "r": 16, "t": 88, "b": 28},
        legend={
            "title": None,
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
        },
    )
    figure.update_annotations(align="left")
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(gridcolor="#E7EDF1", zeroline=False)
    return figure


def build_distribution_histogram(frame: pd.DataFrame, measure: str):
    """Build an observation-count histogram; grouped calculation is not applied."""
    values = coerce_numeric(frame[measure]).dropna()
    if values.empty:
        return None
    chart_frame = pd.DataFrame({"_value": values})
    bins = min(40, max(1, ceil(sqrt(len(values)))))
    figure = px.histogram(
        chart_frame,
        x="_value",
        nbins=bins,
        histfunc="count",
        title=f"Distribution of {measure}",
        labels={"_value": f"{measure} value", "count": "Number of observations"},
        color_discrete_sequence=[COLOR_SEQUENCE[0]],
        hover_data={"_value": ":,.2f"},
    )
    figure.update_xaxes(title_text=f"{measure} value")
    figure.update_yaxes(title_text="Number of observations", rangemode="tozero")
    figure.update_traces(hovertemplate=f"{measure} value: %{{x:,.2f}}<br>Number of observations: %{{y}}<extra></extra>")
    return style_figure(figure)


def build_category_bar_chart(
    grouped: pd.DataFrame,
    measure: str,
    dimensions: list[str],
    aggregation: str,
):
    """Build a category bar chart from already-aggregated data."""
    if grouped.empty or measure not in grouped.columns:
        return None
    category_names = [column for column in dimensions if column in grouped.columns]
    chart_frame = grouped.copy()
    if category_names:
        pieces = [chart_frame[column].astype("string").fillna("(Missing)") for column in category_names]
        labels = pieces[0]
        for piece in pieces[1:]:
            labels = labels + " · " + piece
        chart_frame["_category"] = labels
    else:
        chart_frame["_category"] = "All records"
    figure = px.bar(
        chart_frame,
        x="_category",
        y=measure,
        title=f"{aggregation} {measure} by category",
        labels={"_category": "Category", measure: f"{aggregation} {measure}"},
        hover_data=category_names,
        color_discrete_sequence=[COLOR_SEQUENCE[0]],
    )
    figure.update_traces(hovertemplate="%{x}<br>" + f"{aggregation} {measure}: %{{y:,.2f}}<extra></extra>")
    return style_figure(figure)


def build_category_pie_chart(
    grouped: pd.DataFrame,
    measure: str,
    dimensions: list[str],
    *,
    max_slices: int = 12,
):
    """Build a sum-based category share chart, or return a reason it is unsuitable.

    Values must be nonnegative and have a positive total. When there are many
    groups, the largest ``max_slices - 1`` remain separate and the rest are
    combined into an exact ``Other`` total.
    """
    if not dimensions:
        return None, "Choose at least one category dimension. A pie chart needs groups to compare."
    if grouped.empty or measure not in grouped.columns:
        return None, "There are no valid grouped values to show. Try a bar chart or another measure."
    category_names = [column for column in dimensions if column in grouped.columns]
    if not category_names:
        return None, "The selected category columns are unavailable in the grouped results. Try a bar chart."
    values = coerce_numeric(grouped[measure])
    if values.isna().any():
        valid = values.notna()
        grouped = grouped.loc[valid].copy()
        values = values.loc[valid]
    if values.empty:
        return None, "There are no valid numeric values for a category share. Try a bar chart."
    if (values < 0).any():
        return None, "Pie charts cannot represent negative contributions without misleading shares. Use a bar chart."
    if float(values.sum()) <= 0:
        return None, "The selected measure has a zero total, so category percentages are undefined. Use a bar chart."

    pieces = [grouped[column].astype("string").fillna("(Missing)") for column in category_names]
    labels = pieces[0]
    for piece in pieces[1:]:
        labels = labels + " · " + piece
    chart_frame = pd.DataFrame({"_category": labels, "_value": values.astype(float)}).groupby(
        "_category", as_index=False, sort=False
    )["_value"].sum()
    chart_frame = chart_frame.loc[chart_frame["_value"] > 0].sort_values("_value", ascending=False, kind="stable")
    if chart_frame.empty:
        return None, "All category contributions are zero, so percentages are undefined. Use a bar chart."
    if max_slices < 2:
        max_slices = 2
    if len(chart_frame) > max_slices:
        visible = chart_frame.iloc[: max_slices - 1].copy()
        other_total = float(chart_frame.iloc[max_slices - 1:]["_value"].sum())
        visible.loc[len(visible)] = {"_category": "Other", "_value": other_total}
        chart_frame = visible
    figure = px.pie(
        chart_frame,
        names="_category",
        values="_value",
        title=f"Share of total {measure} by category (Sum)",
        color_discrete_sequence=COLOR_SEQUENCE,
        custom_data=["_value"],
    )
    figure.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}<br>Total: %{value:,.2f}<br>Share: %{percent}<extra></extra>",
        sort=False,
    )
    message = f"Top {max_slices - 1} categories are shown individually; remaining categories are combined into Other." if len(grouped) > max_slices else None
    return style_figure(figure), message
