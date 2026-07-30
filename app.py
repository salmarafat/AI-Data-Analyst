

import streamlit as st
import pandas as pd

from data_loader import load_file, inspect_data
from summary import dataset_summary, build_text_profile
from visualizations import (
    numeric_histograms, categorical_bar_charts, correlation_heatmap, boxplots_for_outliers
)
from statistics_analysis import detect_outliers_iqr, top_correlations, normality_check
from forecasting import (
    FREQ_MAP, build_time_series, analyze_trend, forecast_series, trend_and_forecast_chart,
    rank_based_trend, HAS_STATSMODELS
)
from llm_client import ensure_ollama_running, list_local_models
from insights import generate_business_insights, generate_recommendations
from chatbot import answer_question

st.set_page_config(page_title="AI Data Analyst", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: local model settings + file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    if "ollama_status" not in st.session_state:
        with st.spinner("Checking / starting Ollama..."):
            st.session_state["ollama_status"] = ensure_ollama_running()
    ollama_status = st.session_state["ollama_status"]

    if ollama_status in ("running", "started"):
        ollama_ok = True
        label = "✅ Ollama is running" if ollama_status == "running" else "✅ Ollama started automatically"
        st.success(label)
        available_models = list_local_models()
    elif ollama_status == "not_installed":
        ollama_ok = False
        st.error(" Ollama isn't installed (or not on PATH). Install it from https://ollama.com/download")
        available_models = []
    else:  # timeout
        ollama_ok = False
        st.warning("⏳ Ollama is starting but isn't ready yet. Give it a few seconds, then click below.")
        available_models = []

    if not ollama_ok and st.button("🔄 Retry connecting to Ollama"):
        st.session_state.pop("ollama_status", None)
        st.rerun()

    if available_models:
        selected_model = st.selectbox("Choose local model", available_models)
    else:
        selected_model = st.text_input("Model name (e.g. llama3.1)", value="llama3.1")
        st.caption("If no models appear, pull one with: `ollama pull llama3.1`")

    st.divider()
    uploaded_file = st.file_uploader("📁 Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

st.title(" AI Data Analyst")
st.caption("Upload your company's data file and get inspection, visualizations, statistical analysis, and smart business insights — all running locally, with no data sent to any external server.")

if uploaded_file is None:
    st.info("⬅️ Start by uploading a CSV or Excel file from the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Read File
# ---------------------------------------------------------------------------
try:
    df = load_file(uploaded_file)
except Exception as e:
    st.error(f"An error occurred while reading the file: {e}")
    st.stop()

if df.empty:
    st.warning("The file is empty or contains no valid data.")
    st.stop()

# Store the data in session_state so the chatbot can use it without reloading
st.session_state["df"] = df

inspect = inspect_data(df)
summary = dataset_summary(df)

tabs = st.tabs([
    "🔍 Data Inspection", "📈 Visualizations", "🧮 Statistical Analysis",
    "📉 Trends & Forecasting", "💡 Business Insights & Recommendations", "💬 AI Analyst Chatbot"
])

# ---------------------------------------------------------------------------
# TAB 1: Data Inspection + Dataset Summary
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{inspect['rows']:,}")
    c2.metric("Columns", inspect["cols"])
    c3.metric("Duplicate Rows", inspect["duplicates"])
    c4.metric("Memory Usage (MB)", inspect["memory_usage_mb"])

    st.subheader("Column Types & Missing Values")
    st.dataframe(inspect["dtypes_table"], use_container_width=True)

    st.subheader("Data Sample")
    st.dataframe(inspect["sample"], use_container_width=True)

    if not summary["numeric_summary"].empty:
        st.subheader("Statistical Summary of Numeric Columns")
        st.dataframe(summary["numeric_summary"], use_container_width=True)

    if summary["categorical_summary"]:
        st.subheader("Top Values in Categorical Columns")
        cols = st.columns(min(3, len(summary["categorical_summary"])) or 1)
        for i, (col, top) in enumerate(summary["categorical_summary"].items()):
            with cols[i % len(cols)]:
                st.caption(col)
                st.bar_chart(top)

# ---------------------------------------------------------------------------
# TAB 2: Automatic Visualizations
# ---------------------------------------------------------------------------
with tabs[1]:
    numeric_cols = inspect["numeric_cols"]
    categorical_cols = inspect["categorical_cols"]

    if numeric_cols:
        st.subheader("Distribution of Numeric Columns")
        for col, fig in numeric_histograms(df, numeric_cols):
            st.plotly_chart(fig, use_container_width=True)

    if categorical_cols:
        st.subheader("Categorical Columns")
        for col, fig in categorical_bar_charts(df, categorical_cols):
            st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:
        st.subheader("Correlation Matrix")
        heatmap = correlation_heatmap(df, numeric_cols)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)

    if not numeric_cols and not categorical_cols:
        st.info("Not enough columns to generate automatic visualizations.")

# ---------------------------------------------------------------------------
# TAB 3: Statistical Analysis
# ---------------------------------------------------------------------------
with tabs[2]:
    numeric_cols = inspect["numeric_cols"]

    if numeric_cols:
        st.subheader("Outlier Detection - IQR Method")
        outliers_table = detect_outliers_iqr(df, numeric_cols)
        st.dataframe(outliers_table, use_container_width=True)

        st.subheader("Boxplots for Outliers")
        for col, fig in boxplots_for_outliers(df, numeric_cols):
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Strongest Correlations (|r| ≥ 0.5)")
        corr_table = top_correlations(df, numeric_cols, threshold=0.5)
        if corr_table.empty:
            st.info("No strong correlations found between numeric columns.")
        else:
            st.dataframe(corr_table, use_container_width=True)

        with st.expander("Normality Test (Shapiro-Wilk)"):
            st.dataframe(normality_check(df, numeric_cols), use_container_width=True)
    else:
        st.info("Not enough numeric columns to run statistical analysis.")

# ---------------------------------------------------------------------------
# TAB 4: Trend Analysis + Forecasting
# ---------------------------------------------------------------------------
with tabs[3]:
    numeric_cols = inspect["numeric_cols"]
    date_candidates = list(dict.fromkeys(inspect["datetime_cols"] + inspect["possible_date_cols"]))

    if not numeric_cols:
        st.info("Not enough numeric columns to run trend analysis or forecasting.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            value_col = st.selectbox("Choose the numeric column to analyze", numeric_cols)

        if date_candidates:
            with col2:
                date_col = st.selectbox("Choose a date column", ["-- No date column --"] + date_candidates)
        else:
            date_col = "-- No date column --"
            st.caption("ℹ️ No clear date column was found. We'll use row order as an approximate time index (no precise time-based forecast).")

        if date_col != "-- No date column --":
            c1, c2, c3 = st.columns(3)
            with c1:
                agg_freq_label = st.selectbox("Time aggregation", list(FREQ_MAP.keys()), index=2)
            with c2:
                agg_func = st.selectbox("Aggregation method", ["sum", "mean", "max", "min"], index=0)
            with c3:
                horizon = st.number_input("Number of future periods to forecast", min_value=1, max_value=24, value=6)

            series = build_time_series(df, date_col, value_col, FREQ_MAP[agg_freq_label], agg=agg_func)

            if series.empty or len(series) < 2:
                st.warning("Not enough data to build a time series with the chosen aggregation. Try a different frequency.")
            else:
                trend = analyze_trend(series)
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall Trend", trend["direction"])
                m2.metric("Total % Change", f"{trend['pct_change_total']}%")
                m3.metric("Series Points", len(series))

                forecast = forecast_series(series, periods=int(horizon))
                fig = trend_and_forecast_chart(series, trend["moving_avg"], forecast, value_col)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader(f"📅 Forecasted Values for the Next {horizon} Periods")
                forecast_df = forecast.reset_index()
                forecast_df.columns = ["Period", f"Forecast - {value_col}"]
                st.dataframe(forecast_df, use_container_width=True)

                if (not HAS_STATSMODELS) or len(series) < 4:
                    st.caption("ℹ️ A simple linear regression was used for the forecast (Holt's Exponential Smoothing unavailable or too few series points).")
        else:
            row_series = df[value_col].dropna().reset_index(drop=True)
            trend = rank_based_trend(df, value_col)
            m1, m2 = st.columns(2)
            m1.metric("Overall Trend (approx, by row order)", trend["direction"])
            m2.metric("% Change from First to Last Row", f"{trend['pct_change_total']}%")

            fig = trend_and_forecast_chart(row_series, trend["moving_avg"], pd.Series(dtype=float), value_col)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 5: AI Business Insights + Recommendations
# ---------------------------------------------------------------------------
with tabs[4]:
    if not ollama_ok:
        st.warning("You need to start Ollama first to use AI-generated insights and recommendations.")
    else:
        profile_text = build_text_profile(df, inspect, summary)

        with st.expander("View the text profile sent to the model (optional)"):
            st.text(profile_text)

        if st.button("🧠 Generate Business Insights & Recommendations", type="primary"):
            with st.spinner("The model is analyzing the data and extracting insights..."):
                insights_text = generate_business_insights(profile_text, model=selected_model)
                st.session_state["insights_text"] = insights_text

            st.subheader("💡 Business Insights")
            st.markdown(insights_text)

            with st.spinner("The model is building recommendations..."):
                recs_text = generate_recommendations(profile_text, insights_text, model=selected_model)

            st.subheader("✅ Recommendations")
            st.markdown(recs_text)
        elif "insights_text" in st.session_state:
            st.subheader("💡 Business Insights")
            st.markdown(st.session_state["insights_text"])

# ---------------------------------------------------------------------------
# TAB 6: AI Data Analyst Chatbot
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("💬 Ask questions about your data in natural language")
    
    if not ollama_ok:
        st.warning("You need to start Ollama first to use the chatbot.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("fig") is not None:
                    st.plotly_chart(msg["fig"], use_container_width=True)
                if msg.get("table") is not None:
                    st.dataframe(msg["table"], use_container_width=True)

        question = st.chat_input("Type your question about the data here...")

        if question:
            st.session_state["chat_history"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing the question and running the code..."):
                    output = answer_question(question, df, model=selected_model)

                if output["error"]:
                    st.error(output["error"])
                    with st.expander("View the generated code"):
                        st.code(output["code"], language="python")
                    st.session_state["chat_history"].append({"role": "assistant", "content": output["error"]})
                else:
                    if output["explanation"]:
                        st.markdown(output["explanation"])

                    table_to_store = None
                    if isinstance(output["result"], (pd.DataFrame, pd.Series)):
                        st.dataframe(output["result"], use_container_width=True)
                        table_to_store = output["result"]
                    elif output["result"] is not None:
                        st.write(output["result"])

                    if output["fig"] is not None:
                        st.plotly_chart(output["fig"], use_container_width=True)

                    with st.expander("View the code that was actually executed"):
                        st.code(output["code"], language="python")

                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": output["explanation"] or "Analysis complete.",
                        "fig": output["fig"],
                        "table": table_to_store,
                    })
