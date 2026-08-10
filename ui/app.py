import streamlit as st

from pipelines.reporting import build_dashboard
from ui.charts import render_charts


def run_app():
    st.set_page_config(page_title="DataChef ETL Demo", layout="wide")
    st.title("DataChef ETL Demo")
    st.markdown(
        "This demo ingests raw data, transforms it, and generates a dashboard with Plotly charts."
    )

    if st.button("Run ETL pipeline"):
        with st.spinner("Running ETL pipeline..."):
            report_data = build_dashboard()

        st.success("ETL pipeline completed.")
        render_charts(report_data)

    st.sidebar.header("Navigation")
    st.sidebar.write("Use the button above to execute a sample ETL workflow.")
