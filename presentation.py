"""Streamlit presentation helpers that keep source values and types intact."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype


def dataframe_column_config(frame: pd.DataFrame, profiles: dict | None = None) -> dict:
    """Configure readable formats and left alignment without converting values."""
    config = {}
    for column in frame.columns:
        series = frame[column]
        if is_datetime64_any_dtype(series.dtype):
            config[column] = st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm", alignment="left")
        elif is_numeric_dtype(series.dtype):
            number_format = "%,.0f" if is_integer_dtype(series.dtype) else "%,.2f"
            config[column] = st.column_config.NumberColumn(format=number_format, alignment="left")
        elif profiles and str(column) in profiles and profiles[str(column)]["is_numeric"]:
            # Numeric-looking source text stays textual in the preview.
            config[column] = st.column_config.TextColumn(alignment="left")
        else:
            config[column] = st.column_config.TextColumn(alignment="left")
    return config


def business_table_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation copy with fractional margin ratios shown as percent points."""
    displayed = frame.copy()
    rename = {}
    for column in displayed.columns:
        normalized = str(column).strip().casefold().replace(" ", "_")
        if normalized in {"profit_margin", "margin_ratio"} and is_numeric_dtype(displayed[column].dtype):
            displayed[column] = displayed[column] * 100
            rename[column] = "Profit margin (%)"
    return displayed.rename(columns=rename)
