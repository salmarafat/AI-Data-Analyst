"""
visualizations.py
Generates Automatic Visualizations based on column types.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def numeric_histograms(df: pd.DataFrame, numeric_cols: list, max_charts: int = 6):
    """Returns a list of (column_name, figure) for the top numeric columns"""
    figs = []
    for col in numeric_cols[:max_charts]:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution: {col}", marginal="box")
        fig.update_layout(bargap=0.05)
        figs.append((col, fig))
    return figs


def categorical_bar_charts(df: pd.DataFrame, categorical_cols: list, max_charts: int = 6, top_n: int = 10):
    """Returns a list of (column_name, figure) for the top categorical columns (bar chart of top_n values)"""
    figs = []
    for col in categorical_cols[:max_charts]:
        counts = df[col].value_counts(dropna=True).head(top_n).reset_index()
        counts.columns = [col, "Count"]
        fig = px.bar(counts, x=col, y="Count", title=f"Most Frequent Values: {col}")
        figs.append((col, fig))
    return figs


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    """Returns a figure for the correlation matrix of numeric columns (if there are 2+ numeric columns)"""
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True).round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Matrix (Numeric Columns)",
    )
    return fig


def time_series_chart(df: pd.DataFrame, date_col: str, numeric_col: str):
    """Returns a line chart for a numeric column over time, if a date column exists"""
    try:
        temp = df[[date_col, numeric_col]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col]).sort_values(date_col)
        if temp.empty:
            return None
        fig = px.line(temp, x=date_col, y=numeric_col, title=f"{numeric_col} Over Time ({date_col})")
        return fig
    except Exception:
        return None


def boxplots_for_outliers(df: pd.DataFrame, numeric_cols: list, max_charts: int = 6):
    """Boxplots for visually detecting outliers"""
    figs = []
    for col in numeric_cols[:max_charts]:
        fig = px.box(df, y=col, title=f"Outlier Detection: {col}", points="outliers")
        figs.append((col, fig))
    return figs
