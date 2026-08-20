"""
Chat Intent - DATAChef Dashboard Agent
======================================

Traduce una peticion en lenguaje natural ("top 5 region by amount as bar
chart", "which stores are selling the most") a una "receta" de grafico, con
la MISMA filosofia SPEC-FIRST del dashboard_agent: aqui no se dibuja nada,
solo se produce un ChartRequest declarativo que la UI interpreta.

Reglas de diseño:
1. SCHEMA-GROUNDED: nunca inventa una columna. Toda dimension/medida se
   resuelve contra columnas reales del DataFrame, usando la MISMA deteccion
   de roles que el dashboard automatico (detect_column_roles), asi que
   "medida" y "dimension" significan lo mismo aqui que en el resto del agente.
2. DETERMINISTA: puras reglas, sin LLM. Funciona siempre, sin API key.
3. SI NO SE PUEDE RESOLVER, NO SE ADIVINA: se responde diciendo que columnas
   si existen, en vez de aproximar un grafico incorrecto.

Orden de resolucion (gana la primera que encuentre una dimension real):
    a) sintaxis estricta  "[top N] [MEDIDA by] DIMENSION as TIPO chart"
    b) nombre de columna real mencionado en el texto
    c) sinonimo de negocio ("stores" -> columna "store", "sellers" -> "seller")
    d) atajo: peticion de ranking ("top ...") y la tabla tiene UNA sola
       columna agrupable -> esa es la unica lectura honesta

La medida se resuelve igual, mas un extra: "selling"/"revenue"/"sold" indican
INTENCION de ventas sin nombrar la columna, asi que se usa la medida principal
(_pick_main_measure) en vez de degradar a un simple conteo de filas.

Funcion publica principal:  interpret_message(text, df) -> ChatChartResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from crew.dashboard_agent.dashboard_agent import (
    _pick_main_measure,
    detect_column_roles,
)


@dataclass(frozen=True)
class ChartRequest:
    """Receta declarativa de UN grafico pedido por chat.

    Comparte vocabulario con build_chart_specs del dashboard_agent (type / x /
    y / agg / top_n), asi que un grafico pedido por chat no es un caso especial
    para la UI: se dibuja con la misma logica que los automaticos.
    """

    chart_type: str          # "bar" | "line" | "pie"
    title: str
    dimension: str           # eje X / categoria  (columna real)
    measure: str | None = None   # eje Y (columna real) o None -> conteo
    agg: str = "sum"         # "sum" | "count"
    top_n: int | None = None

    def to_spec(self) -> dict[str, Any]:
        """Convierte a la misma forma de dict que usa build_chart_specs."""
        spec: dict[str, Any] = {
            "type": self.chart_type,
            "x": self.dimension,
            "y": self.measure,
            "agg": self.agg,
            "title": self.title,
        }
        if self.top_n is not None:
            spec["top_n"] = self.top_n
        return spec


@dataclass(frozen=True)
class ChatChartResult:
    """Resultado de un turno del chat: el grafico (si hubo) y que responder."""

    chart_request: ChartRequest | None
    reply: str


# =====================================================================
# 1. SINTAXIS ESTRICTA:  "[top N] [MEDIDA by] DIMENSION as TIPO chart"
# =====================================================================
_TYPE_PATTERN = re.compile(
    r"\bas\s+(?:an?\s+)?(pie|bar|line)(?:\s+chart)?\s*$", re.IGNORECASE
)
_TOP_N_PATTERN = re.compile(r"\btop\s+(\d{1,2})\b", re.IGNORECASE)
_BY_SPLIT = re.compile(r"\bby\b", re.IGNORECASE)


def _match_column(phrase: str, columns: tuple[str, ...]) -> str | None:
    """Resuelve texto libre a EXACTAMENTE una columna real. Nunca inventa.

    Primero coincidencia exacta (case-insensitive), luego subcadena en
    cualquier direccion (asi "sales" encuentra una columna "sales_amount").
    """
    needle = phrase.strip().lower()
    if not needle:
        return None
    for column in columns:
        if column.lower() == needle:
            return column
    for column in columns:
        lowered = column.lower()
        if needle in lowered or lowered in needle:
            return column
    return None


def _parse_strict(text: str, columns: tuple[str, ...]) -> ChartRequest | None:
    """Intenta la forma estricta. Devuelve None si el texto no encaja."""
    type_match = _TYPE_PATTERN.search(text)
    if not type_match:
        return None
    chart_type = type_match.group(1).lower()

    before_as = text[: type_match.start()].strip()
    top_match = _TOP_N_PATTERN.search(before_as)
    top_n = int(top_match.group(1)) if top_match else None
    before_as = _TOP_N_PATTERN.sub("", before_as).strip()
    if not before_as:
        return None

    parts = _BY_SPLIT.split(before_as, maxsplit=1)
    if len(parts) == 2:
        measure_phrase, dimension_phrase = parts
        dimension = _match_column(dimension_phrase, columns)
        if dimension is None:
            return None
        measure_phrase = measure_phrase.strip()
        measure = _match_column(measure_phrase, columns) if measure_phrase else None
        if measure_phrase and measure is None:
            # Nombro una medida que no es columna real -> descartar, no adivinar.
            return None
    else:
        dimension = _match_column(before_as, columns)
        if dimension is None:
            return None
        measure = None

    return ChartRequest(
        chart_type=chart_type,
        title=text[:1].upper() + text[1:],
        dimension=dimension,
        measure=measure,
        agg="sum" if measure else "count",
        top_n=top_n,
    )


# =====================================================================
# 2. LENGUAJE NATURAL SUELTO (fallbacks)
# =====================================================================
_TOP_KEYWORDS = re.compile(
    r"\b(top|most|highest|largest|best|leading|biggest)\b", re.IGNORECASE
)
_TYPE_KEYWORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pie", re.compile(r"\bpie\b", re.IGNORECASE)),
    ("line", re.compile(r"\b(line|trend|over\s+time)\b", re.IGNORECASE)),
    ("bar", re.compile(r"\bbar\b", re.IGNORECASE)),
)
_WORD_PATTERN = re.compile(r"[a-z0-9_]+")
_DEFAULT_TOP_N = 10

# "selling"/"revenue"/... expresan INTENCION de ventas sin nombrar la columna.
_SELLING_INTENT = re.compile(
    r"\b(selling|sold|sells?|sales|revenue|earn(?:ing)?s?|profit(?:s|able)?)\b",
    re.IGNORECASE,
)

# Palabras de negocio que suelen sustituir al nombre real de la columna.
# Cada entrada SOLO resuelve a una columna que existe y que ya fue clasificada
# como dimension: amplia el vocabulario, nunca inventa una columna.
_DIMENSION_SYNONYMS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\bsellers?\b|\bselling\b|\bsalespeople\b|\bsales\s*reps?\b",
                   re.IGNORECASE),
        ("seller", "salesperson", "sales_rep", "rep", "agent", "vendedor"),
    ),
    (
        re.compile(r"\bstores?\b|\bshops?\b|\bbranch(?:es)?\b|\blocations?\b|\boutlets?\b",
                   re.IGNORECASE),
        ("store", "shop", "branch", "location", "outlet", "tienda", "sucursal"),
    ),
    (
        re.compile(r"\bproducts?\b|\bitems?\b|\bskus?\b", re.IGNORECASE),
        ("product", "item", "sku", "producto"),
    ),
    (
        re.compile(r"\bcustomers?\b|\bclients?\b|\bbuyers?\b", re.IGNORECASE),
        ("customer", "client", "buyer", "cliente"),
    ),
    (
        re.compile(r"\bregions?\b|\bareas?\b|\bzones?\b|\bterritor\w*\b", re.IGNORECASE),
        ("region", "area", "zone", "territory", "zona"),
    ),
    (
        re.compile(r"\bcategor\w*\b", re.IGNORECASE),
        ("category", "type", "categoria"),
    ),
    (
        re.compile(r"\bcountr\w*\b|\bpais\b", re.IGNORECASE),
        ("country", "pais"),
    ),
    (
        re.compile(r"\bstatus\b|\bstate\b|\bestado\b", re.IGNORECASE),
        ("status", "state", "estado"),
    ),
)


def _mentioned_columns(text: str, columns: tuple[str, ...]) -> list[str]:
    """Columnas reales nombradas en el texto, en orden de aparicion."""
    lowered = text.lower()
    words = _WORD_PATTERN.findall(lowered)
    hits: list[tuple[int, str]] = []
    for column in columns:
        needle = column.lower()
        index = lowered.find(needle)
        if index == -1:
            for word in words:
                if len(word) >= 3 and (word in needle or needle in word):
                    index = lowered.find(word)
                    break
        if index != -1:
            hits.append((index, column))
    hits.sort(key=lambda item: item[0])
    ordered: list[str] = []
    for _, column in hits:
        if column not in ordered:
            ordered.append(column)
    return ordered


def _synonym_dimension(text: str, dimensions: tuple[str, ...]) -> str | None:
    """Dimension real encontrada via una palabra de negocio, si la hay."""
    for trigger, needles in _DIMENSION_SYNONYMS:
        if not trigger.search(text):
            continue
        for column in dimensions:
            lowered = column.lower()
            if any(needle in lowered for needle in needles):
                return column
    return None


def _chart_type(text: str) -> str:
    for chart_type, pattern in _TYPE_KEYWORDS:
        if pattern.search(text):
            return chart_type
    return "bar"


def _top_n(text: str) -> int | None:
    explicit = _TOP_N_PATTERN.search(text)
    if explicit:
        return int(explicit.group(1))
    return _DEFAULT_TOP_N if _TOP_KEYWORDS.search(text) else None


def _confirmation(request: ChartRequest) -> str:
    scope = f"top {request.top_n} " if request.top_n else ""
    by = f" by **{request.measure}**" if request.measure else ""
    return (
        f"Added: {scope}**{request.dimension}**{by} "
        f"as a **{request.chart_type}** chart."
    )


def _cant_resolve_reply(roles: dict) -> str:
    dims = ", ".join(roles["dimensions"][:8]) or "none detected"
    measures = ", ".join(roles["measures"][:8]) or "none detected"
    return (
        "I couldn't match that to a column in this table. "
        f"Group-by columns available: {dims}. Measures available: {measures}. "
        'Try something like "top 5 <column> by <measure> as bar chart".'
    )


# =====================================================================
# 3. ORQUESTADOR (la unica funcion que la UI llama)
# =====================================================================
def interpret_message(text: str, df: pd.DataFrame) -> ChatChartResult:
    """Convierte UN mensaje de chat en un ChartRequest, o explica por que no."""

    message = (text or "").strip()
    if not message:
        return ChatChartResult(
            None, 'Ask me for a chart, e.g. "top 5 region by amount as bar chart".'
        )

    columns = tuple(str(c) for c in df.columns)

    strict = _parse_strict(message, columns)
    if strict is not None:
        return ChatChartResult(strict, _confirmation(strict))

    roles = detect_column_roles(df)
    measure_priority = roles["measures"]
    measures = frozenset(measure_priority)
    dimensions = tuple(roles["dimensions"])
    ranking_intent = bool(_TOP_KEYWORDS.search(message))

    mentioned = _mentioned_columns(message, columns)
    dimension = next((c for c in mentioned if c in dimensions), None)
    if dimension is None:
        dimension = _synonym_dimension(message, dimensions)
    if dimension is None and ranking_intent and len(dimensions) == 1:
        dimension = dimensions[0]
    if dimension is None:
        return ChatChartResult(None, _cant_resolve_reply(roles))

    measure = next((c for c in mentioned if c in measures and c != dimension), None)
    if measure is None and measure_priority and _SELLING_INTENT.search(message):
        measure = _pick_main_measure(measure_priority)

    request = ChartRequest(
        chart_type=_chart_type(message),
        title=message[:1].upper() + message[1:],
        dimension=dimension,
        measure=measure,
        agg="sum" if measure else "count",
        top_n=_top_n(message),
    )
    return ChatChartResult(request, _confirmation(request))
