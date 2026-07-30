
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


FREQ_MAP = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "MS",
    "Quarterly": "QS",
    "Yearly": "YS",
}


def build_time_series(df: pd.DataFrame, date_col: str, value_col: str, freq: str, agg: str = "sum") -> pd.Series:
    """
    Converts a date column + a numeric column into a regularly resampled time series
    at the requested frequency.
    """
    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if temp.empty:
        return pd.Series(dtype=float)

    temp = temp.set_index(date_col)
    series = temp[value_col].resample(freq).agg(agg)
    series = series.fillna(0)
    return series


def analyze_trend(series: pd.Series) -> dict:
    """
    Computes the trend of a time series using a simple linear regression (slope) against the
    time index, and returns: direction (up/down/stable), total percent change, and moving average.
    """
    if len(series) < 2:
        return {"direction": "Not enough data", "slope": 0, "pct_change_total": 0, "moving_avg": series}

    x = np.arange(len(series))
    y = series.values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    mean_y = y.mean() if y.mean() != 0 else 1
    relative_slope = slope / abs(mean_y)

    if relative_slope > 0.01:
        direction = "📈 Upward trend"
    elif relative_slope < -0.01:
        direction = "📉 Downward trend"
    else:
        direction = "➖ Roughly stable"

    pct_change_total = ((y[-1] - y[0]) / abs(y[0]) * 100) if y[0] != 0 else 0

    window = max(2, min(7, len(series) // 3))
    moving_avg = series.rolling(window=window, min_periods=1).mean()

    return {
        "direction": direction,
        "slope": round(float(slope), 4),
        "pct_change_total": round(float(pct_change_total), 2),
        "moving_avg": moving_avg,
        "window": window,
    }


def forecast_series(series: pd.Series, periods: int = 6) -> pd.Series:
    """
    Forecasts future values for the time series.
    - If statsmodels is available and there are enough points (>= 4): uses Holt's
      Exponential Smoothing (additive trend).
    - Otherwise: falls back to a simple linear regression (numpy polyfit).
    """
    if len(series) < 2:
        return pd.Series(dtype=float)

    if HAS_STATSMODELS and len(series) >= 4:
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit()
            forecast = fit.forecast(periods)
            return forecast
        except Exception:
            pass  # fall through to fallback below

    # Fallback: simple linear regression
    x = np.arange(len(series))
    y = series.values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(series), len(series) + periods)
    future_y = slope * future_x + intercept

    if isinstance(series.index, pd.DatetimeIndex) and len(series.index) >= 2:
        step = series.index[-1] - series.index[-2]
        future_index = [series.index[-1] + step * (i + 1) for i in range(periods)]
    else:
        future_index = range(len(series), len(series) + periods)

    return pd.Series(future_y, index=future_index)


def trend_and_forecast_chart(series: pd.Series, moving_avg: pd.Series, forecast: pd.Series, value_col: str) -> go.Figure:
    """Plots the original series + moving average + future forecast in a single chart"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers", name="Actual Values"))

    if moving_avg is not None and len(moving_avg) > 0:
        fig.add_trace(go.Scatter(x=moving_avg.index, y=moving_avg.values, mode="lines",
                                  name="Moving Average", line=dict(dash="dot")))

    if forecast is not None and len(forecast) > 0:
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers",
                                  name="Forecast", line=dict(color="orange", dash="dash")))

    fig.update_layout(title=f"Trend & Forecast: {value_col}", xaxis_title="Period", yaxis_title=value_col)
    return fig


def rank_based_trend(df: pd.DataFrame, numeric_col: str) -> dict:
    """
    When there's no date column: computes an approximate trend based on row order
    (useful when the data is actually ordered chronologically but has no explicit date column).
    """
    series = df[numeric_col].dropna().reset_index(drop=True)
    return analyze_trend(series)
