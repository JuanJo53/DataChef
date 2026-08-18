import streamlit as st
import plotly.express as px
import pandas as pd


# ---------------------------------------------------------------------
# Helpers de formato
# ---------------------------------------------------------------------
def _shades(hex_color: str, n: int) -> list:
    """Genera n tonos (del color base hacia claro) para las rebanadas del pastel."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    out = []
    n = max(int(n), 1)
    for i in range(n):
        t = 0.0 if n == 1 else (i / n) * 0.6  # hasta 60% hacia el blanco
        rr, gg, bb = int(r + (255 - r) * t), int(g + (255 - g) * t), int(b + (255 - b) * t)
        out.append(f"#{rr:02x}{gg:02x}{bb:02x}")
    return out


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

    # Controles de personalizacion (color + tipo de grafico). Solo afectan como
    # se DIBUJA; nunca cambian el spec (el contrato con el pipeline se respeta).
    color = "#22D3EE"
    if charts:
        with st.expander("🎨 Customize dashboard"):
            color = st.color_picker("Chart color", value="#22D3EE", key="dash_color")
            st.caption("Pick a color, and switch each chart's type with the dropdown next to its title.")

    type_options = ["bar", "line", "area", "pie"]
    for i, chart in enumerate(charts):
        if chart["type"] == "histogram":
            st.subheader(chart["title"])
            fig = px.histogram(df, x=chart["x"], color_discrete_sequence=[color])
            st.plotly_chart(fig, use_container_width=True, key=f"dash_chart_{i}")
            continue

        # Titulo + selector de tipo de grafico (por grafico).
        head, ctrl = st.columns([4, 1])
        head.subheader(chart["title"])
        default_type = chart["type"] if chart["type"] in type_options else "bar"
        ctype = ctrl.selectbox(
            "Type", type_options, index=type_options.index(default_type),
            key=f"dash_ctype_{i}", label_visibility="collapsed",
        )

        data = _aggregate(df, chart)
        # La columna Y es "count" en graficos de conteo, o la medida real.
        ycol = "count" if (chart.get("agg") == "count" or chart.get("y") is None) else chart["y"]
        if ctype == "line":
            fig = px.line(data, x=chart["x"], y=ycol, markers=True, color_discrete_sequence=[color])
        elif ctype == "area":
            fig = px.area(data, x=chart["x"], y=ycol, color_discrete_sequence=[color])
        elif ctype == "pie":
            fig = px.pie(data, names=chart["x"], values=ycol, color_discrete_sequence=_shades(color, len(data)))
        else:  # bar
            fig = px.bar(data, x=chart["x"], y=ycol, color_discrete_sequence=[color])
        st.plotly_chart(fig, use_container_width=True, key=f"dash_chart_{i}")

    # 3) Insights en lenguaje natural.
    insights = spec.get("insights", [])
    if insights:
        st.subheader("Insights")
        for frase in insights:
            st.markdown(f"- {frase}")
