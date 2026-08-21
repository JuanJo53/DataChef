import io
import os
import re
import sys

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from ui.dashboard_chat import render_dashboard_chat
from ui.ingestion_view import render_ingestion
from ui.styles import _apply_custom_styles


def _to_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Prep mínima hacia la capa 'gold': tipa fechas y números."""

    df = df.copy()

    # ==========================================
    # DATE COLUMNS
    # ==========================================

    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                dayfirst=True,
            )

    # ==========================================
    # TEXT -> NUMERIC CANDIDATES
    # ==========================================

    text_cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]

    for col in text_cols:
        limpio = (
            df[col]
            .astype(str)
            .str.replace(
                r"[$,%\s]",
                "",
                regex=True,
            )
        )

        convertido = pd.to_numeric(
            limpio,
            errors="coerce",
        )

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

    current_stage = st.session_state["stage"]

    for index, label in enumerate(stages):

        button_type = (
            "primary"
            if index == current_stage
            else "secondary"
        )

        if st.sidebar.button(
            label,
            key=f"nav_{index}",
            use_container_width=True,
            type=button_type,
        ):

            st.session_state["stage"] = index

            st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "DataChef Agentic Platform"
    )


def _render_upload_stage():

    st.header("Stage 1: Ingest and diagnose")
    st.subheader("Upload a raw dataset")

    # =====================================================
    # UPLOAD AREA
    # =====================================================

    uploaded_file = st.file_uploader(
        "Choose a CSV or JSON file",
        type=["csv", "json"],
        label_visibility="collapsed",
    )

    st.markdown(
        "<div style='height: 100px;'></div>",
        unsafe_allow_html=True,
    )

    # =====================================================
    # DEMO DATASET BUTTON
    # =====================================================

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button(
            "Load demo dataset",
            use_container_width=True,
        ):
            st.session_state["raw_df"] = _demo_dataframe()
            st.session_state["uploaded_file"] = "demo_orders.csv"
            st.success("Demo dataset loaded.")

    # =====================================================
    # READ FILE
    # =====================================================

    if uploaded_file is not None:

        st.session_state["uploaded_file"] = uploaded_file.name

        if uploaded_file.name.endswith(".csv"):
            st.session_state["raw_df"] = pd.read_csv(uploaded_file)

        else:
            st.session_state["raw_df"] = pd.read_json(uploaded_file)

    # =====================================================
    # DATASET LOADED
    # =====================================================

    if st.session_state["raw_df"] is not None:

        df = st.session_state["raw_df"]

        st.markdown(
            "<div style='height: 20px;'></div>",
            unsafe_allow_html=True,
        )

        st.success(
            f"Loaded: {st.session_state['uploaded_file']}"
        )

        # =================================================
        # DATA PREVIEW
        # =================================================

        st.markdown("### Data preview")

        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=300,
        )

        st.markdown(
            "<div style='height: 15px;'></div>",
            unsafe_allow_html=True,
        )

        # =================================================
        # OVERVIEW
        # =================================================

        st.markdown("### Dataset overview")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric(
                "Rows",
                len(df),
            )

        with col_b:
            st.metric(
                "Columns",
                df.shape[1],
            )

        with col_c:
            st.metric(
                "Data types",
                df.dtypes.nunique(),
            )

        st.markdown(
            "<div style='height: 18px;'></div>",
            unsafe_allow_html=True,
        )

        # =================================================
        # ANALYZE BUTTON
        # =================================================

        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            if st.button(
                "Analyze data →",
                type="primary",
                use_container_width=True,
            ):
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

        # =====================================================================
        # BLOQUE DE DESCARGA DE DATOS GOLD MULTIFORMATO
        # =====================================================================
        st.subheader("📥 Download Transformed Data (Gold Layer)")
        
        format_choice = st.selectbox(
            "Select export format:",
            options=["CSV", "Parquet", "JSON", "Excel"],
            key="download_format_selector",
        )

        df_export = st.session_state["gold_df"]

        # Generar datos y parámetros según la opción seleccionada actualmente
        if format_choice == "CSV":
            file_data = df_export.to_csv(index=False).encode("utf-8")
            file_name = "data_gold.csv"
            mime_type = "text/csv"

        elif format_choice == "Parquet":
            buffer = io.BytesIO()
            df_export.to_parquet(buffer, index=False)
            file_data = buffer.getvalue()
            file_name = "data_gold.parquet"
            mime_type = "application/octet-stream"

        elif format_choice == "JSON":
            file_data = df_export.to_json(orient="records", indent=2).encode("utf-8")
            file_name = "data_gold.json"
            mime_type = "application/json"

        elif format_choice == "Excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_export.to_excel(writer, index=False)
            file_data = buffer.getvalue()
            file_name = "data_gold.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        # Usar key dinámico en el botón de descarga para forzar la actualización en Streamlit
        st.download_button(
            label=f"⬇️ Download {file_name}",
            data=file_data,
            file_name=file_name,
            mime=mime_type,
            key=f"btn_download_{format_choice.lower()}",
            use_container_width=True,
        )