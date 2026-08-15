import os
import sys

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import re
import pandas as pd
import streamlit as st

# Logo DATAChef
LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "img",
    "Gemini_Generated_Image_adhg9madhg9madhg.png",
)

# Importaciones de los Agentes del Equipo
from crew.dashboard_agent.dashboard_agent import build_dashboard_spec
from crew.dashboard_agent.exporters import to_powerbi, to_tableau
from crew.transformation_agent.transformation_agent import (
    execute_transformation_with_sandbox,
    save_pipeline_script,
)
from ui.charts import render_charts
from ui.ingestion_view import render_ingestion


def _to_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Prep mínima hacia la capa 'gold': tipa fechas y números."""
    df = df.copy()
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    text_cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    for col in text_cols:
        limpio = df[col].astype(str).str.replace(r"[$,%\s]", "", regex=True)
        convertido = pd.to_numeric(limpio, errors="coerce")
        if convertido.notna().mean() >= 0.8:
            df[col] = convertido
    return df


def _init_state():
    defaults = {
        "stage": 0,
        "uploaded_file": None,
        "raw_df": None,
        "gold_df": None,
        "generated_code": None,
        "dashboard_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _demo_dataframe():
    return pd.DataFrame(
        {
            "order_id": [101, 102, 103, 103, 104],
            "customer_id": ["C-001", None, "C-003", "C-003", "C-005"],
            "region": ["North", "North", "South", "South", "West"],
            "order_date": [
                "2026-01-03",
                "2026-01-05",
                "bad-date",
                "2026-01-08",
                "2026-01-10",
            ],
            "amount": ["$120.5", "$80.0", "$330.0", "$330.0", "$95.0"],
            "status": [
                "completed",
                "completed",
                "pending",
                "pending",
                "completed",
            ],
        }
    )


def _render_stage_nav():
    stages = [
        "1. Upload",
        "2. Diagnose",
        "3. Transform",
        "4. Dashboard",
    ]

    st.sidebar.title("Workflow")
    for index, label in enumerate(stages):
        if st.sidebar.button(
            label, key=f"nav_{index}", use_container_width=True
        ):
            st.session_state["stage"] = index

    st.sidebar.markdown("---")
    st.sidebar.caption("DataChef Agentic Platform")


def _render_upload_stage():
    st.header("Stage 1: Ingest and diagnose")
    st.subheader("Upload a raw dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or JSON file", type=["csv", "json"]
    )
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Load demo dataset", use_container_width=True):
            st.session_state["raw_df"] = _demo_dataframe()
            st.session_state["uploaded_file"] = "demo_orders.csv"
            st.success("Demo dataset loaded.")

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file.name
        if uploaded_file.name.endswith(".csv"):
            st.session_state["raw_df"] = pd.read_csv(uploaded_file)
        else:
            st.session_state["raw_df"] = pd.read_json(uploaded_file)

    if st.session_state["raw_df"] is not None:
        st.success(f"Loaded: {st.session_state['uploaded_file']}")
        st.dataframe(
            st.session_state["raw_df"].head(10), use_container_width=True
        )

        st.markdown("### Dataset overview")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", len(st.session_state["raw_df"]))
        col_b.metric("Columns", st.session_state["raw_df"].shape[1])
        col_c.metric(
            "Data types", str(st.session_state["raw_df"].dtypes.nunique())
        )

        if st.button("Analyze data", type="primary"):
            st.session_state["stage"] = 1
            st.rerun()


def _render_diagnose_stage():
    st.header("Stage 2: Ingestion & diagnosis")
    st.markdown(
        "The ingestion agent profiles the raw data, scores its health, and "
        "suggests SQL, indexes and quality alerts. Ask it anything below."
    )
    if st.session_state["raw_df"] is None:
        st.warning("Load a dataset first in the upload stage.")
        return

    df = _to_gold(st.session_state["raw_df"])
    table_name = st.session_state.get("uploaded_file") or "ingested_data"
    render_ingestion(df, table_name=table_name)

    st.markdown("---")
    if st.button("Accept diagnosis and continue", type="primary"):
        st.session_state["stage"] = 2
        st.rerun()


def _render_transform_stage():
    st.header("Stage 3: Transformation and processing")
    st.markdown(
        "The **Action Agent** executes transformations directly in a **Local Sandbox** "
        "with automatic **Self-Healing** capabilities upon encountering syntax errors."
    )

    if st.session_state["raw_df"] is None:
        st.warning("Load a dataset first in Stage 1.")
        return

    st.subheader("🤖 Ask Action Agent to transform your data")
    user_prompt = st.text_area(
        "Enter natural language instruction:",
        value="Clean missing customer_id values with 'UNKNOWN', drop duplicates in order_id, remove $ symbols in amount and parse order_date as date.",
        height=100,
    )

    if st.button("🚀 Run Transformation Agent", type="primary"):
        with st.spinner(
            "Executing Pandas code in Local Sandbox (with Self-Healing)..."
        ):
            try:
                df_transformed, generated_code = (
                    execute_transformation_with_sandbox(
                        st.session_state["raw_df"], user_prompt
                    )
                )

                st.session_state["gold_df"] = df_transformed
                st.session_state["generated_code"] = generated_code
                st.success("✅ Transformation executed successfully!")

            except Exception as e:
                st.error(f"❌ Transformation failed: {e}")

    if (
        "gold_df" in st.session_state
        and st.session_state["gold_df"] is not None
    ):
        st.subheader("Preview of Transformed (Gold Layer) Data")
        st.dataframe(st.session_state["gold_df"], use_container_width=True)

        st.subheader("Generated Python Script")
        st.code(st.session_state["generated_code"], language="python")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "📄 Export Reusable Pipeline (.py)", use_container_width=True
            ):
                path = save_pipeline_script(
                    st.session_state["generated_code"],
                    "pipeline_transformacion.py",
                )
                st.success(f"Pipeline saved to `{path}`!")

        with col2:
            if st.button(
                "📊 Build Dashboard", type="primary", use_container_width=True
            ):
                st.session_state["stage"] = 3
                st.session_state["dashboard_ready"] = True
                st.rerun()


def _render_dashboard_stage():
    st.header("Stage 4: Dashboard and insights")
    st.markdown(
        "The dashboard agent turns the cleaned (gold) data into KPIs, charts and business insights."
    )

    if not st.session_state["dashboard_ready"]:
        st.info(
            "Complete the previous stages to generate the dashboard. The agent is ready once the transformed data is approved."
        )
        return

    df = st.session_state.get("gold_df")
    if df is None:
        df = st.session_state.get("raw_df")

    if df is None or df.empty:
        st.warning("No data available. Load a dataset in Stage 1 first.")
        return

    df = _to_gold(df)

    goal = st.selectbox(
        "What would you like to analyze?",
        [
            "Overview (auto)",
            "Revenue by region",
            "Order performance by month",
            "Customer retention trend",
        ],
    )
    audience = st.selectbox(
        "Audience", ["Executive team", "Operations team", "Analyst team"]
    )

    spec = build_dashboard_spec(df)
    st.caption(
        f"Analyzed {spec['meta']['rows']} rows x {spec['meta']['columns']} cols "
        f"| Goal: {goal} | Audience: {audience}"
    )

    render_charts({"spec": spec, "data": df})

    table = re.sub(
        r"\W+",
        "_",
        str(st.session_state.get("uploaded_file") or "DataChef"),
    ).strip("_")

    with st.expander("📤  Export to Power BI (DAX measures)"):
        pkg = to_powerbi(spec, table_name=table)
        st.code("\n".join(pkg["dax_measures"]), language="dax")
        if pkg["relationships"]:
            st.caption("Suggested relationships:")
            for r in pkg["relationships"]:
                st.markdown(
                    f"- `{r['from']}` → `{r['to']}` ({r['cardinality']}, {r['cross_filter']})"
                )

    with st.expander("📤  Export to Tableau"):
        res = to_tableau(spec)
        st.info(f"Status: {res['status']} — {res['reason']}")


def run_app():
    st.set_page_config(page_title="DataChef", page_icon=LOGO_PATH, layout="wide")

    st.logo(LOGO_PATH, size="large")

    _init_state()
    _render_stage_nav()

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image(LOGO_PATH, width="stretch")
        st.caption(
            "Agentic workflow for ingestion, transformation, and dashboarding"
        )

    stage = st.session_state["stage"]
    if stage == 0:
        _render_upload_stage()
    elif stage == 1:
        _render_diagnose_stage()
    elif stage == 2:
        _render_transform_stage()
    else:
        _render_dashboard_stage()


if __name__ == "__main__":
    run_app()