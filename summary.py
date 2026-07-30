"""
summary.py
Builds a comprehensive Dataset Summary before deeper statistical analysis.
"""

import pandas as pd
import numpy as np


def dataset_summary(df: pd.DataFrame) -> dict:
    """
    Returns a general summary of the dataset:
    - descriptive statistics for numeric columns
    - description of categorical/text columns
    - top values per categorical column
    """
    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(include=["object", "category", "bool"])

    numeric_summary = numeric_df.describe().T.round(3) if not numeric_df.empty else pd.DataFrame()

    categorical_summary = {}
    for col in categorical_df.columns:
        top = df[col].value_counts(dropna=True).head(5)
        categorical_summary[col] = top

    return {
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "n_numeric": numeric_df.shape[1],
        "n_categorical": categorical_df.shape[1],
    }


def build_text_profile(df: pd.DataFrame, inspect: dict, summary: dict, max_cols: int = 25) -> str:
    """
    Builds a short text profile of the dataset to send to the local LLM as context
    (aggregated statistics only, not the raw data, to save tokens and protect privacy).
    """
    lines = []
    lines.append(f"Row count: {inspect['rows']} | Column count: {inspect['cols']}")
    lines.append(f"Duplicate rows: {inspect['duplicates']}")
    lines.append(f"Numeric columns: {', '.join(inspect['numeric_cols'][:max_cols]) or 'None'}")
    lines.append(f"Categorical/text columns: {', '.join(inspect['categorical_cols'][:max_cols]) or 'None'}")

    if not summary["numeric_summary"].empty:
        lines.append("\nStatistical summary of numeric columns:")
        lines.append(summary["numeric_summary"].to_string())

    if summary["categorical_summary"]:
        lines.append("\nTop values in categorical columns:")
        for col, top in list(summary["categorical_summary"].items())[:max_cols]:
            lines.append(f"- {col}: " + ", ".join(f"{idx} ({val})" for idx, val in top.items()))

    missing_cols = inspect["dtypes_table"]
    high_missing = missing_cols[missing_cols["Missing %"] > 0]
    if not high_missing.empty:
        lines.append("\nColumns with missing values:")
        lines.append(high_missing[["Column", "Missing %"]].to_string(index=False))

    return "\n".join(lines)
