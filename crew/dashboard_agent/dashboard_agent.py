"""
Dashboard Agent - DATAChef
==========================

Este agente es la ultima etapa del pipeline (capa GOLD -> dashboard).

Filosofia de diseño (importante para explicarlo):
1. SPEC-FIRST: el agente NO dibuja graficos. Solo produce una "receta"
   declarativa (un diccionario) que describe QUE mostrar: KPIs, graficos e
   insights. Quien dibuja es la capa de UI (ui/charts.py con Plotly). Gracias a
   esto, la MISMA receta puede exportarse mañana a Power BI o Tableau sin tocar
   este agente.
2. SCHEMA-FLEXIBLE: no asume nombres de columnas fijos ("revenue", "region").
   Detecta el ROL de cada columna (fecha / medida numerica / dimension
   categorica). Asi funciona con el CSV de ejemplo hoy y con la capa gold real
   del equipo de transformacion mañana, aunque cambien los nombres.
3. LLM OPCIONAL: los insights se generan con reglas (siempre funciona, sin
   API key). Si hay una key de Gemini disponible, se enriquecen con lenguaje
   natural. Nunca truena por falta de key: hace fallback a las reglas.

Funcion publica principal:  build_dashboard_spec(df) -> dict
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


# =====================================================================
# 1. DETECCION DE ROLES DE COLUMNAS  (el corazon "schema-flexible")
# =====================================================================
def detect_column_roles(df: pd.DataFrame) -> dict[str, Any]:
    """Clasifica cada columna en un rol: fecha, medida o dimension.

    - date:       la primera columna de tipo fecha (eje temporal).
    - measures:   columnas numericas -> sirven para KPIs y ejes 'Y'.
    - dimensions: columnas de texto de baja cardinalidad (<=30 valores
                  distintos) -> sirven para agrupar (ejes 'X' / categorias).
    """
    date_col: str | None = None
    measures: list[str] = []
    dimensions: list[str] = []

    for col in df.columns:
        serie = df[col]

        # a) Fecha: ya viene como datetime (transformation.py la convierte).
        if pd.api.types.is_datetime64_any_dtype(serie):
            if date_col is None:
                date_col = col
            continue

        # b) Medida: columna numerica que NO sea un identificador.
        #    Sumar un 'order_id' no tiene sentido, asi que se ignora.
        if pd.api.types.is_numeric_dtype(serie):
            if not _looks_like_id(col):
                measures.append(col)
            continue

        # c) Dimension: texto con pocas categorias (util para agrupar).
        #    Si tuviera cientos de valores unicos (ej. un ID) no sirve como eje.
        if serie.nunique(dropna=True) <= 30:
            dimensions.append(col)

    return {"date": date_col, "measures": measures, "dimensions": dimensions}


def _pick_main_measure(measures: list[str]) -> str | None:
    """Elige la medida 'principal' (la que mas le importa al negocio).

    Prioriza nombres tipicos de negocio; si no hay match, usa la primera.
    """
    if not measures:
        return None
    prioridad = ("revenue", "sales", "amount", "total", "price", "value", "units")
    for palabra in prioridad:
        for m in measures:
            if palabra in m.lower():
                return m
    return measures[0]


def _money_like(name: str) -> bool:
    """True si el nombre de la columna huele a dinero (para formatear como $)."""
    return any(p in name.lower() for p in ("revenue", "sales", "amount", "price", "cost", "value"))


def _looks_like_id(name: str) -> bool:
    """True si el nombre parece un identificador (no sirve como medida a sumar)."""
    n = name.lower()
    return n == "id" or n.endswith("_id")


# =====================================================================
# 2. KPIs  (las tarjetas grandes de numeros)
# =====================================================================
def build_kpis(df: pd.DataFrame, roles: dict) -> list[dict]:
    """Construye las tarjetas KPI a partir de las medidas detectadas."""
    kpis: list[dict] = [
        {"label": "Total records", "value": int(len(df)), "format": "int"},
    ]

    # Un KPI de SUMA por cada medida (maximo 3 para no saturar la fila).
    for measure in roles["measures"][:3]:
        kpis.append(
            {
                "label": f"Total {measure}",
                "value": float(df[measure].sum()),
                "format": "currency" if _money_like(measure) else "number",
            }
        )
    return kpis


# =====================================================================
# 3. GRAFICOS  (definicion DECLARATIVA, sin dibujar todavia)
# =====================================================================
def build_chart_specs(df: pd.DataFrame, roles: dict) -> list[dict]:
    """Devuelve una lista de 'recetas' de graficos.

    Cada receta dice: tipo, columna X, columna Y, y como agregar (sum).
    La UI las interpreta y las dibuja con Plotly.
    """
    charts: list[dict] = []
    main = _pick_main_measure(roles["measures"])

    # Sin medidas numericas: al menos contar filas por la primera dimension,
    # para que SIEMPRE se muestre algo util (no una pantalla vacia).
    if main is None:
        if roles["dimensions"]:
            dim = roles["dimensions"][0]
            charts.append({
                "type": "bar", "x": dim, "y": None, "agg": "count",
                "top_n": 10, "title": f"Row count by {dim}",
            })
        return charts

    # a) Tendencia en el tiempo (si existe una columna de fecha).
    if roles["date"]:
        charts.append(
            {
                "type": "line",
                "x": roles["date"],
                "y": main,
                "agg": "sum",
                "title": f"{main} over time",
            }
        )

    # b) Comparativo por cada dimension (barras), hasta 2 dimensiones.
    for dim in roles["dimensions"][:2]:
        charts.append(
            {
                "type": "bar",
                "x": dim,
                "y": main,
                "agg": "sum",
                "top_n": 10,
                "title": f"{main} by {dim}",
            }
        )

    # c) Composicion (pastel) de la primera dimension, si tiene pocas categorias.
    if roles["dimensions"]:
        dim0 = roles["dimensions"][0]
        if df[dim0].nunique() <= 8:
            charts.append(
                {
                    "type": "pie",
                    "x": dim0,
                    "y": main,
                    "agg": "sum",
                    "title": f"{main} share by {dim0}",
                }
            )

    # d) Si hay medida pero nada por que agrupar (ni fecha ni dimensiones),
    #    mostrar al menos la distribucion de la medida (histograma).
    if not roles["date"] and not roles["dimensions"]:
        charts.append({"type": "histogram", "x": main, "title": f"Distribution of {main}"})

    return charts


# =====================================================================
# 4. INSIGHTS por REGLAS  (siempre funciona, sin LLM)
# =====================================================================
def build_rule_based_insights(df: pd.DataFrame, roles: dict) -> list[str]:
    """Genera frases de negocio con pura estadistica de pandas."""
    insights: list[str] = []
    main = _pick_main_measure(roles["measures"])
    if main is None:
        return ["No numeric measures found to analyze."]

    total = df[main].sum()
    insights.append(f"Total {main} across all records is {total:,.2f}.")

    # Categoria lider y su participacion (%).
    if roles["dimensions"]:
        dim = roles["dimensions"][0]
        por_cat = df.groupby(dim)[main].sum().sort_values(ascending=False)
        lider = por_cat.index[0]
        share = (por_cat.iloc[0] / total * 100) if total else 0
        insights.append(f"'{lider}' leads {dim} with {share:.0f}% of total {main}.")
        if len(por_cat) > 1:
            insights.append(f"'{por_cat.index[-1]}' is the lowest {dim} by {main}.")

    # Tendencia: primer periodo vs ultimo periodo.
    if roles["date"]:
        serie = df.groupby(roles["date"])[main].sum().sort_index()
        if len(serie) >= 2 and serie.iloc[0]:
            cambio = (serie.iloc[-1] - serie.iloc[0]) / serie.iloc[0] * 100
            rumbo = "up" if cambio >= 0 else "down"
            insights.append(f"{main} trended {rumbo} {abs(cambio):.0f}% from first to last period.")

    return insights


# =====================================================================
# 5. INSIGHTS con LLM (Gemini) - OPCIONAL
# =====================================================================
def build_llm_insights(kpis: list[dict], rule_insights: list[str]) -> list[str] | None:
    """Enriquece los insights con Gemini. Devuelve None si no es posible.

    Reusa el mismo patron que el transformation_agent del equipo:
    langchain_google_genai + variable de entorno GOOGLE_API_KEY.
    Cualquier fallo (sin key, sin libreria, error de red) -> None -> fallback.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # flash = rapido y barato para texto corto
            temperature=0.3,
            google_api_key=api_key,
        )

        contexto = "KPIs:\n" + "\n".join(
            f"- {k['label']}: {k['value']}" for k in kpis
        )
        contexto += "\n\nStats:\n" + "\n".join(f"- {s}" for s in rule_insights)

        prompt = (
            "You are a data analyst. Based on these KPIs and stats, write exactly "
            "3 short, business-focused insights (one sentence each, no numbering, "
            "no preamble):\n\n" + contexto
        )
        respuesta = llm.invoke(prompt).content
        # Parsear a lista limpia (una frase por linea).
        lineas = [l.strip("-• \t") for l in respuesta.splitlines() if l.strip()]
        return lineas[:3] or None
    except Exception:
        # Nunca romper el dashboard por el LLM.
        return None


# =====================================================================
# 6. ORQUESTADOR  (la unica funcion que el resto del proyecto llama)
# =====================================================================
def build_dashboard_spec(df: pd.DataFrame, use_llm: bool = True) -> dict:
    """Construye la especificacion completa del dashboard a partir de la
    capa gold (df ya limpio y transformado).

    Devuelve un dict declarativo con: title, engine, meta, kpis, charts,
    insights. Ese dict es el 'contrato' que consume la UI (y mañana Power BI).
    """
    roles = detect_column_roles(df)
    kpis = build_kpis(df, roles)
    charts = build_chart_specs(df, roles)
    insights = build_rule_based_insights(df, roles)

    engine = "rule-based"
    if use_llm:
        llm_insights = build_llm_insights(kpis, insights)
        if llm_insights:
            insights = llm_insights
            engine = "gemini"

    return {
        "title": "Sales Dashboard",
        "engine": engine,
        "meta": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "roles": roles,
        "kpis": kpis,
        "charts": charts,
        "insights": insights,
    }
