import os
import re

import streamlit as st
import pandas as pd

# Logo DATAChef (imagen que el equipo agrego en ui/img/). Ruta absoluta para
# que funcione sin importar desde donde se ejecute la app.
LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "img",
    "Gemini_Generated_Image_adhg9madhg9madhg.png",
)

from crew.dashboard_agent.dashboard_agent import build_dashboard_spec
from crew.dashboard_agent.exporters import to_powerbi, to_tableau
from ui.charts import render_charts


def _to_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Prep minima hacia la capa 'gold': tipa fechas y numeros. En el pipeline
    final esto lo entregaria ya listo el transformation agent (aqui es respaldo).

    - Columnas con 'date' en el nombre -> datetime.
    - Columnas de texto que en realidad son numeros ('$1,200', '85%') -> numero.
      Esto evita el caso 'no salen graficos' cuando el CSV trae numeros como texto.
    """
    df = df.copy()
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    text_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    for col in text_cols:
        limpio = df[col].astype(str).str.replace(r"[$,%\s]", "", regex=True)
        convertido = pd.to_numeric(limpio, errors="coerce")
        # Solo se convierte si la mayoria (>=80%) son numeros de verdad;
        # asi un 'customer_id' tipo 'C-001' se queda como texto.
        if convertido.notna().mean() >= 0.8:
            df[col] = convertido
    return df


def _init_state():
    defaults = {
        "stage": 0,
        "uploaded_file": None,
        "raw_df": None,
        "issue_actions": {
            "missing_values": "Fill with UNKNOWN",
            "duplicates": "Remove duplicates",
            "invalid_dates": "Standardize format",
            "outliers": "Flag for review",
        },
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
            "order_date": ["2024-01-03", "2024-01-05", "bad-date", "2024-01-08", "2024-01-10"],
            "amount": [120.5, 80.0, 330.0, 330.0, 95.0],
            "status": ["completed", "completed", "pending", "pending", "completed"],
        }
    )


def _diagnostic_issues():
    return [
        {
            "id": "missing_values",
            "title": "Missing customer IDs",
            "severity": "High",
            "count": 1,
            "detail": "1 row has a null customer_id value. This may break joins and customer-level aggregation.",
        },
        {
            "id": "duplicates",
            "title": "Duplicate orders detected",
            "severity": "Medium",
            "count": 2,
            "detail": "Multiple rows share the same order_id. Duplicate records should be checked before reporting.",
        },
        {
            "id": "invalid_dates",
            "title": "Invalid date format",
            "severity": "High",
            "count": 1,
            "detail": "One order_date value does not match the expected ISO date format and could fail time-based analytics.",
        },
        {
            "id": "outliers",
            "title": "Unusual amount spike",
            "severity": "Medium",
            "count": 1,
            "detail": "One order amount is notably higher than the rest of the dataset and may indicate a data issue or premium sale.",
        },
    ]


def _render_stage_nav():
    stages = [
        "1. Upload",
        "2. Diagnose",
        "3. Transform",
        "4. Dashboard",
    ]

    st.sidebar.title("Workflow")
    for index, label in enumerate(stages):
        if st.sidebar.button(label, key=f"nav_{index}", use_container_width=True):
            st.session_state["stage"] = index

    st.sidebar.markdown("---")
    st.sidebar.caption("Hackathon UI mockup")


def _render_upload_stage():
    st.header("Stage 1: Ingest and diagnose")
    st.subheader("Upload a raw dataset")

    uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])
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
        st.dataframe(st.session_state["raw_df"].head(10), use_container_width=True)

        st.markdown("### Dataset overview")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", len(st.session_state["raw_df"]))
        col_b.metric("Columns", st.session_state["raw_df"].shape[1])
        col_c.metric("Data types", str(st.session_state["raw_df"].dtypes.nunique()))

        if st.button("Analyze data", type="primary"):
            st.session_state["stage"] = 1
            st.rerun()


def _render_diagnose_stage():
    st.header("Stage 2: Diagnosis report")
    st.markdown("The ingestion agent scans for quality issues before any actual transformation starts.")

    if st.session_state["raw_df"] is None:
        st.warning("Load a dataset first in the upload stage.")
        return

    issues = _diagnostic_issues()
    for issue in issues:
        with st.container():
            st.markdown(f"### {issue['title']}")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(issue["detail"])
            with col2:
                st.badge(issue["severity"], color="orange")
                st.write(f"Affected rows: {issue['count']}")

            options = [
                "Drop rows",
                "Fill with UNKNOWN",
                "Standardize format",
                "Flag for review",
                "Keep as-is",
            ]
            selected = st.radio(
                "Recommended action",
                options,
                index=0,
                key=f"action_{issue['id']}",
                horizontal=True,
            )
            st.session_state["issue_actions"][issue["id"]] = selected
            st.markdown("---")

    if st.button("Accept diagnosis and continue", type="primary"):
        st.session_state["stage"] = 2
        st.rerun()


def _render_transform_stage():
    st.header("Stage 3: Transformation and processing")
    st.markdown("The transformation agent uses the diagnosis decisions to create a reusable, explainable cleaning pipeline.")

    if st.session_state["raw_df"] is None:
        st.warning("Load a dataset first.")
        return

    st.subheader("Decision summary")
    for issue in _diagnostic_issues():
        st.write(f"- {issue['title']}: {st.session_state['issue_actions'].get(issue['id'], 'Drop rows')}")

    st.subheader("Preview of transformed output")
    transformed_preview = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104],
            "customer_id": ["C-001", "UNKNOWN", "C-003", "C-005"],
            "region": ["North", "North", "South", "West"],
            "order_date": ["2024-01-03", "2024-01-05", "2024-01-08", "2024-01-10"],
            "amount": [120.5, 80.0, 330.0, 95.0],
            "status": ["completed", "completed", "pending", "completed"],
        }
    )
    st.dataframe(transformed_preview, use_container_width=True)

    st.subheader("Generated pipeline script")
    pipeline_code = '''import pandas as pd


def clean_orders(df):
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    df["customer_id"] = df["customer_id"].fillna("UNKNOWN")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"]) 
    return df
'''
    st.code(pipeline_code, language="python")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save pipeline", use_container_width=True):
            st.success("Pipeline mock saved to data/processed/")
    with col2:
        if st.button("Build dashboard", type="primary", use_container_width=True):
            st.session_state["stage"] = 3
            st.session_state["dashboard_ready"] = True
            st.rerun()


def _render_dashboard_stage():
    st.header("Stage 4: Dashboard and insights")
    st.markdown("The dashboard agent turns the cleaned (gold) data into KPIs, charts and business insights.")

    if not st.session_state["dashboard_ready"]:
        st.info("Complete the previous stages to generate the dashboard. The agent is ready once the transformed data is approved.")
        return

    # 1. Conseguir el DataFrame a analizar.
    #    Prioridad: la capa gold que dejen los agentes previos ('gold_df');
    #    si aun no existe, se usa el dataset crudo cargado en el Paso 1.
    df = st.session_state.get("gold_df")
    if df is None:
        df = st.session_state.get("raw_df")
    if df is None or df.empty:
        st.warning("No data available. Load a dataset in Stage 1 first.")
        return

    # 2. Prep ligera hacia gold (tipar fechas). Respaldo por si el
    #    transformation agent aun no entrega los datos ya tipados.
    df = _to_gold(df)

    # 3. Contexto de negocio (se conserva la UX del mockup del equipo).
    goal = st.selectbox(
        "What would you like to analyze?",
        [
            "Overview (auto)",
            "Revenue by region",
            "Order performance by month",
            "Customer retention trend",
        ],
    )
    audience = st.selectbox("Audience", ["Executive team", "Operations team", "Analyst team"])

    # 4. Ejecutar el AGENTE DE DASHBOARDS -> produce el spec declarativo.
    spec = build_dashboard_spec(df)
    st.caption(
        f"Analyzed {spec['meta']['rows']} rows x {spec['meta']['columns']} cols "
        f"| Goal: {goal} | Audience: {audience}"
    )

    # 5. Dibujar el dashboard real con la capa de charts (Plotly).
    render_charts({"spec": spec, "data": df})

    # 6. Exportadores: mismo spec -> Power BI / Tableau (sin reescribir el agente).
    table = re.sub(r"\W+", "_", str(st.session_state.get("uploaded_file") or "DataChef")).strip("_")
    with st.expander("📤  Export to Power BI (DAX measures)"):
        pkg = to_powerbi(spec, table_name=table)
        st.code("\n".join(pkg["dax_measures"]), language="dax")
        if pkg["relationships"]:
            st.caption("Suggested relationships:")
            for r in pkg["relationships"]:
                st.markdown(f"- `{r['from']}` → `{r['to']}` ({r['cardinality']}, {r['cross_filter']})")
        st.caption("Live push hook ready for `powerbi-modeling-mcp`.")

    with st.expander("📤  Export to Tableau"):
        res = to_tableau(spec)  # sin api_key todavia
        st.info(f"Status: {res['status']} — {res['reason']}")
        st.caption("Add a Tableau API key in to_tableau(spec, api_key=...) to connect later.")

    st.button("Export report", type="primary")


def run_app():
    st.set_page_config(page_title="DataChef", page_icon=LOGO_PATH, layout="wide")

    # Logo de marca: arriba a la izquierda y en el sidebar.
    st.logo(LOGO_PATH, size="large")

    _init_state()
    _render_stage_nav()

    # Encabezado: logo grande centrado.
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image(LOGO_PATH, width="stretch")
        st.caption("Agentic workflow for ingestion, transformation, and dashboarding")

    stage = st.session_state["stage"]
    if stage == 0:
        _render_upload_stage()
    elif stage == 1:
        _render_diagnose_stage()
    elif stage == 2:
        _render_transform_stage()
    else:
        _render_dashboard_stage()
