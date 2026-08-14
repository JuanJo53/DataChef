import streamlit as st
import plotly.express as px
import pandas as pd


# ---------------------------------------------------------------------
# Helpers de formato
# ---------------------------------------------------------------------
def _format_value(value, fmt: str) -> str:
    """Da formato humano a los valores de los KPIs."""
    try:
        if fmt == "currency":
            return f"${value:,.0f}"
        if fmt == "int":
            return f"{int(value):,}"
        return f"{value:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _aggregate(df: pd.DataFrame, chart: dict) -> pd.DataFrame:
    """Aplica el group-by declarado en la receta del grafico.

    Ejemplo: {"x": "region", "y": "revenue", "agg": "sum"}
    -> df.groupby("region")["revenue"].sum()
    Si agg == "count" (o no hay 'y'), cuenta filas por categoria.
    """
    x = chart["x"]
    y = chart.get("y")
    agg = chart.get("agg", "sum")

    # Conteo por categoria (cuando no hay medida numerica que sumar).
    if agg == "count" or y is None:
        data = df[x].value_counts(dropna=True).reset_index()
        data.columns = [x, "count"]
        if "top_n" in chart:
            data = data.head(chart["top_n"])
        return data

    data = df.groupby(x)[y].agg(agg).reset_index()

    # Ordenar: por fecha si es linea, por valor (desc) si es barra/pastel.
    if chart["type"] == "line":
        data = data.sort_values(x)
    else:
        data = data.sort_values(y, ascending=False)
        if "top_n" in chart:
            data = data.head(chart["top_n"])
    return data


# ---------------------------------------------------------------------
# Render principal: consume el "spec" del dashboard_agent
# ---------------------------------------------------------------------
def render_charts(report_data: dict):
    spec = report_data.get("spec", {})
    df = report_data.get("data")

    # Estado inicial / errores.
    if not spec:
        st.info("Run the ETL pipeline to see charts and summary metrics.")
        return
    if spec.get("error"):
        st.error(spec["error"])
        return
    if df is None or df.empty:
        st.warning("No data available to render the dashboard.")
        return

    # Titulo + badge del motor de insights (reglas vs Gemini).
    engine = spec.get("engine", "rule-based")
    st.header(spec.get("title", "Dashboard"))
    st.caption(f"Insights engine: {engine}")

    # 1) KPIs en una fila de tarjetas.
    kpis = spec.get("kpis", [])
    if kpis:
        cols = st.columns(len(kpis))
        for col, kpi in zip(cols, kpis):
            col.metric(kpi["label"], _format_value(kpi["value"], kpi["format"]))

    # 2) Graficos: se dibujan interpretando cada receta declarativa.
    charts = spec.get("charts", [])
    if not charts:
        # Estado vacio explicito: dice POR QUE y muestra un resumen del dato.
        st.info(
            "Couldn't auto-detect chartable columns (no numeric measures or "
            "groupable dimensions were found). Here's a data summary instead."
        )
        st.dataframe(df.head(20), use_container_width=True)
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            st.write(numeric.describe())

    for chart in charts:
        st.subheader(chart["title"])
        if chart["type"] == "histogram":
            fig = px.histogram(df, x=chart["x"])
        else:
            data = _aggregate(df, chart)
            # La columna Y es "count" en graficos de conteo, o la medida real.
            ycol = "count" if (chart.get("agg") == "count" or chart.get("y") is None) else chart["y"]
            if chart["type"] == "line":
                fig = px.line(data, x=chart["x"], y=ycol, markers=True)
            elif chart["type"] == "bar":
                fig = px.bar(data, x=chart["x"], y=ycol)
            elif chart["type"] == "pie":
                fig = px.pie(data, names=chart["x"], values=ycol)
            else:
                continue
        st.plotly_chart(fig, use_container_width=True)

    # 3) Insights en lenguaje natural.
    insights = spec.get("insights", [])
    if insights:
        st.subheader("Insights")
        for frase in insights:
            st.markdown(f"- {frase}")
