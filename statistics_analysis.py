"""
statistics_analysis.py
Handles Statistical Analysis: correlations, outliers, and basic distribution tests.
"""

import pandas as pd
import numpy as np
from scipy import stats


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Detects outliers using the IQR (Interquartile Range) method for each numeric column.
    Returns a table with: column, outlier count, percentage, and normal lower/upper bounds.
    """
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        rows.append({
            "Column": col,
            "Outlier Count": len(outliers),
            "Percentage %": round(len(outliers) / len(series) * 100, 2) if len(series) else 0,
            "Normal Lower Bound": round(lower, 3),
            "Normal Upper Bound": round(upper, 3),
        })
    return pd.DataFrame(rows)


def top_correlations(df: pd.DataFrame, numeric_cols: list, threshold: float = 0.5) -> pd.DataFrame:
    """
    Returns the strongest correlations between numeric columns (above threshold, positive or negative).
    """
    if len(numeric_cols) < 2:
        return pd.DataFrame()

    corr = df[numeric_cols].corr(numeric_only=True)
    pairs = []
    seen = set()
    for col1 in corr.columns:
        for col2 in corr.columns:
            if col1 == col2 or (col2, col1) in seen:
                continue
            seen.add((col1, col2))
            value = corr.loc[col1, col2]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"Column A": col1, "Column B": col2, "Correlation": round(value, 3)})

    result = pd.DataFrame(pairs)
    if not result.empty:
        result = result.reindex(result["Correlation"].abs().sort_values(ascending=False).index)
    return result


def normality_check(df: pd.DataFrame, numeric_cols: list, sample_limit: int = 5000) -> pd.DataFrame:
    """
    Shapiro-Wilk test to check whether each numeric column's distribution is approximately normal.
    (Uses a random sample if the data is too large, since the test has a sample size limit)
    """
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        sample = series.sample(min(len(series), sample_limit), random_state=42) if len(series) > sample_limit else series
        try:
            stat, p_value = stats.shapiro(sample)
            rows.append({
                "Column": col,
                "p-value": round(p_value, 5),
                "Approximately Normal?": "Yes" if p_value > 0.05 else "No",
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def numeric_vs_category_summary(df: pd.DataFrame, numeric_col: str, categorical_col: str) -> pd.DataFrame:
    """Mean/median of a numeric column grouped by a category - useful for business comparisons (e.g. average sales by region)"""
    grouped = df.groupby(categorical_col)[numeric_col].agg(["mean", "median", "sum", "count"]).round(2)
    grouped = grouped.sort_values("sum", ascending=False)
    grouped.columns = ["Mean", "Median", "Total", "Row Count"]
    return grouped
