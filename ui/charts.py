import streamlit as st
import plotly.express as px
import pandas as pd


def render_charts(report_data: dict):
    st.header("Dashboard")

    if not report_data:
        st.info("Run the ETL pipeline to see charts and summary metrics.")
        return

    summary = report_data.get("summary", {})
    metrics = report_data.get("metrics", {})
    history = report_data.get("history", pd.DataFrame())

    st.subheader("Summary")
    st.write(summary)

    if not history.empty:
        st.subheader("Sales trend")
        fig = px.line(history, x="date", y="value", title="Sample metric over time")
        st.plotly_chart(fig, use_container_width=True)

    if metrics:
        st.subheader("Key metrics")
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, value)
