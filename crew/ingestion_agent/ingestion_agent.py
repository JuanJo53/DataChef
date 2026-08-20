"""
Ingestion Agent - DATAChef
==========================

Primera etapa del pipeline: ingesta + diagnostico de datos crudos, con vision
de Data Engineer / DBA.

Que produce (todo desde un DataFrame crudo):
1. PERFIL por columna: tipo, nulos, unicos, tipo SQL sugerido, si es PII, si es
   candidata a llave primaria.
2. SALUD del dato (health): un score 0-100 + tarjetas (cards) de completitud,
   duplicados, columnas con problemas y riesgo PII.
3. ISSUES: lista de problemas detectados con severidad y accion sugerida.
4. SQL: script CREATE TABLE (T-SQL / SQL Server) con tipos inferidos.
5. INDICES: recomendaciones (PK clustered, FKs, columnas de fecha).
6. ALERTAS: reglas de monitoreo de calidad sugeridas (nulls, freshness, etc.).
7. CHAT: responde preguntas sobre el dataset (reglas + Gemini opcional).

Filosofia (igual que el dashboard_agent):
- SPEC-FIRST: devuelve un dict declarativo; la UI lo dibuja.
- SCHEMA-FLEXIBLE: no asume nombres de columnas.
- LLM OPCIONAL: todo funciona offline con reglas; Gemini solo enriquece.

Funciones publicas:  build_ingestion_report(df)  y  answer_question(df, pregunta)
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd


# Patrones para detectar PII (informacion personal) por valor.
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
# Palabras en el nombre de columna que sugieren PII.
_PII_WORDS = ("email", "mail", "phone", "tel", "ssn", "dni", "passport",
              "name", "nombre", "address", "direccion", "birth", "dob", "card")


# =====================================================================
# Helpers
# =====================================================================
def _looks_like_id(name: str) -> bool:
    n = name.lower()
    return n == "id" or n.endswith("_id") or n.endswith("id")


def _is_text_series(serie: pd.Series) -> bool:
    """True si la columna es de texto (ni numero, ni fecha, ni booleano).
    Robusto en pandas 2 y 3 (donde el texto puede ser dtype 'object' o 'str')."""
    return not (
        pd.api.types.is_numeric_dtype(serie)
        or pd.api.types.is_datetime64_any_dtype(serie)
        or pd.api.types.is_bool_dtype(serie)
    )


def _sql_type(serie: pd.Series) -> str:
    """Infiere un tipo de SQL Server (T-SQL) a partir de la columna de pandas."""
    if pd.api.types.is_bool_dtype(serie):
        return "BIT"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "DATETIME2"
    if pd.api.types.is_integer_dtype(serie):
        maximo = serie.abs().max()
        return "INT" if pd.notna(maximo) and maximo < 2_147_483_647 else "BIGINT"
    if pd.api.types.is_float_dtype(serie):
        return "DECIMAL(18,2)"
    # Texto: se dimensiona el NVARCHAR segun el largo maximo observado.
    largos = serie.dropna().astype(str).str.len()
    maximo = int(largos.max()) if not largos.empty else 0
    for cota in (50, 100, 255, 500, 1000, 4000):
        if maximo <= cota:
            return f"NVARCHAR({cota})"
    return "NVARCHAR(MAX)"


def _is_pii(name: str, serie: pd.Series) -> bool:
    """PII por nombre de columna o por valores (emails / telefonos)."""
    if any(w in name.lower() for w in _PII_WORDS):
        return True
    # El chequeo por valor solo aplica a texto: evita falsos positivos donde
    # una fecha ('2024-01-03') o un numero se confunde con un telefono.
    if not _is_text_series(serie):
        return False
    muestra = serie.dropna().astype(str).head(20)
    if muestra.empty:
        return False
    hits = sum(bool(_EMAIL_RE.search(v) or _PHONE_RE.search(v)) for v in muestra)
    return hits >= max(3, len(muestra) // 2)


# =====================================================================
# 1. PERFIL por columna
# =====================================================================
def profile_columns(df: pd.DataFrame) -> list[dict]:
    total = len(df)
    perfil: list[dict] = []
    for col in df.columns:
        serie = df[col]
        nulos = int(serie.isna().sum())
        unicos = int(serie.nunique(dropna=True))
        perfil.append(
            {
                "name": col,
                "dtype": str(serie.dtype),
                "sql_type": _sql_type(serie),
                "nulls": nulos,
                "null_pct": round(nulos / total * 100, 1) if total else 0.0,
                "unique": unicos,
                "is_pk_candidate": (nulos == 0 and unicos == total and total > 0),
                "is_pii": _is_pii(col, serie),
            }
        )
    return perfil


# =====================================================================
# 2. SALUD del dato
# =====================================================================
def compute_health(df: pd.DataFrame, perfil: list[dict]) -> dict:
    total_celdas = df.size or 1
    nulos_totales = sum(c["nulls"] for c in perfil)
    completitud = (1 - nulos_totales / total_celdas) * 100

    dup = int(df.duplicated().sum())
    unicidad = (1 - dup / len(df)) * 100 if len(df) else 100

    cols_con_problemas = sum(1 for c in perfil if c["null_pct"] > 0)
    cols_pii = [c["name"] for c in perfil if c["is_pii"]]

    # Score ponderado (completitud + unicidad).
    score = round(0.6 * completitud + 0.4 * unicidad)
    grado = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"

    return {
        "score": score,
        "grade": grado,
        "completeness_pct": round(completitud, 1),
        "uniqueness_pct": round(unicidad, 1),
        "duplicate_rows": dup,
        "columns_with_issues": cols_con_problemas,
        "pii_columns": cols_pii,
    }


def build_cards(df: pd.DataFrame, health: dict) -> list[dict]:
    """Tarjetas resumidas para la UI. status: good / warn / bad."""
    def estado(cond_ok, cond_warn):
        return "good" if cond_ok else "warn" if cond_warn else "bad"

    return [
        {"label": "Health score", "value": f"{health['score']}/100 ({health['grade']})",
         "status": estado(health["score"] >= 90, health["score"] >= 60),
         "detail": "Weighted completeness + uniqueness."},
        {"label": "Completeness", "value": f"{health['completeness_pct']}%",
         "status": estado(health["completeness_pct"] >= 98, health["completeness_pct"] >= 90),
         "detail": "Share of non-null cells."},
        {"label": "Duplicate rows", "value": health["duplicate_rows"],
         "status": estado(health["duplicate_rows"] == 0, health["duplicate_rows"] <= 2),
         "detail": "Fully duplicated rows."},
        {"label": "Cols with nulls", "value": health["columns_with_issues"],
         "status": estado(health["columns_with_issues"] == 0, health["columns_with_issues"] <= 2),
         "detail": "Columns containing missing values."},
        {"label": "PII columns", "value": len(health["pii_columns"]),
         "status": estado(len(health["pii_columns"]) == 0, True),
         "detail": ", ".join(health["pii_columns"]) or "No obvious PII detected."},
    ]


# =====================================================================
# 3. ISSUES detectados
# =====================================================================
def detect_issues(df: pd.DataFrame, perfil: list[dict]) -> list[dict]:
    issues: list[dict] = []

    for c in perfil:
        if c["null_pct"] > 0:
            sev = "High" if c["null_pct"] >= 20 else "Medium" if c["null_pct"] >= 5 else "Low"
            issues.append({
                "id": f"nulls_{c['name']}", "title": f"Missing values in '{c['name']}'",
                "severity": sev, "count": c["nulls"],
                "detail": f"{c['null_pct']}% of rows are null.",
                "suggested_action": "Fill with default / drop rows",
            })
        if c["is_pii"]:
            issues.append({
                "id": f"pii_{c['name']}", "title": f"Possible PII in '{c['name']}'",
                "severity": "High", "count": c["unique"],
                "detail": "Column looks like personal data (name/email/phone).",
                "suggested_action": "Mask / anonymize before storing",
            })

    dup = int(df.duplicated().sum())
    if dup:
        issues.append({
            "id": "dup_rows", "title": "Duplicate rows detected",
            "severity": "Medium" if dup <= 2 else "High", "count": dup,
            "detail": "Identical rows appear more than once.",
            "suggested_action": "Drop duplicates",
        })

    # Texto que en realidad es numerico (deberia castearse).
    for col in [c for c in df.columns if _is_text_series(df[c])]:
        muestra = df[col].dropna().astype(str).head(30)
        if not muestra.empty:
            num = sum(bool(re.fullmatch(r"-?\d+(\.\d+)?", v.strip())) for v in muestra)
            if num / len(muestra) >= 0.8:
                issues.append({
                    "id": f"cast_{col}", "title": f"'{col}' looks numeric but is text",
                    "severity": "Low", "count": len(muestra),
                    "detail": "Most values are numbers stored as strings.",
                    "suggested_action": f"Cast '{col}' to a numeric type",
                })
    return issues


# =====================================================================
# 4 & 5. SQL (CREATE TABLE) + INDICES
# =====================================================================
def suggest_pk(perfil: list[dict]) -> str | None:
    """Elige una llave primaria: columna unica y sin nulos (prioriza *_id)."""
    candidatas = [c["name"] for c in perfil if c["is_pk_candidate"]]
    if not candidatas:
        return None
    for c in candidatas:
        if _looks_like_id(c):
            return c
    return candidatas[0]


def suggest_indexes(df: pd.DataFrame, perfil: list[dict], pk: str | None) -> list[dict]:
    idx: list[dict] = []
    if pk:
        idx.append({"name": f"PK_{{table}}", "type": "CLUSTERED PRIMARY KEY",
                    "columns": [pk], "rationale": "Unique, non-null key column."})
    for c in perfil:
        col = c["name"]
        if col == pk:
            continue
        if _looks_like_id(col):
            idx.append({"name": f"IX_{{table}}_{col}", "type": "NONCLUSTERED",
                        "columns": [col], "rationale": "Foreign-key-like column (joins)."})
        elif "DATETIME" in c["sql_type"]:
            idx.append({"name": f"IX_{{table}}_{col}", "type": "NONCLUSTERED",
                        "columns": [col], "rationale": "Date column (time-range filters)."})
    return idx


def build_sql_ddl(df: pd.DataFrame, perfil: list[dict], table_name: str,
                  pk: str | None, indexes: list[dict]) -> dict:
    lineas = []
    for c in perfil:
        nulabilidad = "NOT NULL" if c["nulls"] == 0 else "NULL"
        lineas.append(f"    [{c['name']}] {c['sql_type']} {nulabilidad}")
    cuerpo = ",\n".join(lineas)
    if pk:
        cuerpo += f",\n    CONSTRAINT [PK_{table_name}] PRIMARY KEY CLUSTERED ([{pk}])"

    create = f"CREATE TABLE [{table_name}] (\n{cuerpo}\n);"

    index_ddl = []
    for i in indexes:
        if "PRIMARY KEY" in i["type"]:
            continue  # la PK ya va dentro del CREATE TABLE
        nombre = i["name"].replace("{table}", table_name)
        cols = ", ".join(f"[{c}]" for c in i["columns"])
        index_ddl.append(f"CREATE NONCLUSTERED INDEX [{nombre}] ON [{table_name}] ({cols});")

    return {"create_table": create, "index_ddl": index_ddl}


# =====================================================================
# 6. ALERTAS de calidad sugeridas
# =====================================================================
def suggest_alerts(df: pd.DataFrame, perfil: list[dict], pk: str | None) -> list[dict]:
    alerts: list[dict] = []

    for c in perfil:
        if c["null_pct"] > 0:
            alerts.append({
                "name": f"Null rate: {c['name']}", "type": "Completeness",
                "severity": "High" if c["null_pct"] >= 20 else "Medium",
                "condition": f"null_rate([{c['name']}]) > {max(1, round(c['null_pct']))}%",
                "rationale": "Warn if missing values grow beyond the current baseline.",
            })

    if pk:
        alerts.append({
            "name": f"Duplicate key: {pk}", "type": "Uniqueness", "severity": "High",
            "condition": f"COUNT(*) > COUNT(DISTINCT [{pk}])",
            "rationale": "The primary key must stay unique across loads.",
        })

    fechas = [c["name"] for c in perfil if "DATETIME" in c["sql_type"]]
    if fechas:
        alerts.append({
            "name": f"Freshness: {fechas[0]}", "type": "Timeliness", "severity": "Medium",
            "condition": f"MAX([{fechas[0]}]) < TODAY - 2 days",
            "rationale": "Warn if no new data has arrived recently.",
        })

    alerts.append({
        "name": "Row volume drop", "type": "Volume", "severity": "Medium",
        "condition": "daily_row_count < 0.5 * 7day_avg",
        "rationale": "Sudden drops in row count usually mean a broken source.",
    })

    for c in df.select_dtypes(include=["number"]).columns:
        if _looks_like_id(c):
            continue
        alerts.append({
            "name": f"Outliers: {c}", "type": "Validity", "severity": "Low",
            "condition": f"[{c}] outside mean +/- 3*stddev",
            "rationale": "Flag extreme values that may be data errors.",
        })
        break  # una de ejemplo basta para el demo
    return alerts


# =====================================================================
# 7. CHAT - responde preguntas sobre el dataset
# =====================================================================
def _llm_answer(df: pd.DataFrame, pregunta: str) -> str | None:
    """Responde con Gemini si hay API key + libreria. Si no, None (fallback)."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2,
                                     google_api_key=api_key)
        contexto = (
            f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns.\n"
            f"Columns and dtypes:\n{df.dtypes.to_string()}\n\n"
            f"First rows:\n{df.head(5).to_string()}"
        )
        prompt = (
            "You are a data engineer assistant. Answer the user's question about "
            "this dataset briefly and factually.\n\n"
            f"{contexto}\n\nQuestion: {pregunta}"
        )
        return llm.invoke(prompt).content.strip()
    except Exception:
        return None


def answer_question(df: pd.DataFrame, pregunta: str) -> str:
    """Chat del agente: primero reglas rapidas, luego Gemini, luego ayuda."""
    q = pregunta.lower().strip()

    if any(w in q for w in ("how many row", "row count", "records", "cuantas fila", "registros")):
        return f"The dataset has **{len(df):,} rows**."

    if any(w in q for w in ("how many column", "cuantas columna")):
        return f"The dataset has **{df.shape[1]} columns**: {', '.join(df.columns)}."

    if "column" in q or "columna" in q:
        return "Columns: " + ", ".join(f"`{c}` ({df[c].dtype})" for c in df.columns)

    if any(w in q for w in ("missing", "null", "faltan", "nulos")):
        nulos = df.isna().sum()
        con_nulos = nulos[nulos > 0]
        if con_nulos.empty:
            return "No missing values detected. 🎉"
        detalle = ", ".join(f"`{c}`: {n}" for c, n in con_nulos.items())
        return f"Missing values by column: {detalle}."

    if any(w in q for w in ("duplicate", "duplicad")):
        d = int(df.duplicated().sum())
        return f"There {'is' if d == 1 else 'are'} **{d} duplicate row(s)**."

    if any(w in q for w in ("type", "dtype", "tipo")):
        return "Column types:\n" + "\n".join(f"- `{c}`: {t}" for c, t in df.dtypes.items())

    # Pregunta por una columna especifica: "unique values in region"
    for col in df.columns:
        if col.lower() in q:
            serie = df[col]
            return (f"Column `{col}`: type {serie.dtype}, "
                    f"{serie.nunique()} unique values, {serie.isna().sum()} nulls. "
                    f"Sample: {list(serie.dropna().unique()[:5])}")

    # Fallback a LLM.
    respuesta = _llm_answer(df, pregunta)
    if respuesta:
        return respuesta

    return (
        "I can answer questions like: *how many rows?*, *which columns?*, "
        "*where are the missing values?*, *are there duplicates?*, *what type is <column>?* "
        "Add a Gemini API key (GOOGLE_API_KEY) to unlock free-form questions."
    )


# =====================================================================
# ORQUESTADOR
# =====================================================================
def build_ingestion_report(df: pd.DataFrame, table_name: str = "ingested_data") -> dict:
    """Construye el reporte completo de ingesta/diagnostico (dict declarativo)."""
    # Nombre de tabla SQL seguro (sin espacios ni caracteres raros).
    table_name = re.sub(r"\W+", "_", table_name).strip("_") or "ingested_data"

    perfil = profile_columns(df)
    health = compute_health(df, perfil)
    pk = suggest_pk(perfil)
    indexes = suggest_indexes(df, perfil, pk)

    return {
        "meta": {"rows": int(len(df)), "columns": int(df.shape[1]), "table_name": table_name},
        "health": health,
        "cards": build_cards(df, health),
        "columns": perfil,
        "issues": detect_issues(df, perfil),
        "primary_key": pk,
        "indexes": indexes,
        "sql": build_sql_ddl(df, perfil, table_name, pk, indexes),
        "alerts": suggest_alerts(df, perfil, pk),
    }
