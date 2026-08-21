"""
Chat Intent - DATAChef Dashboard Agent
======================================

Translates a natural-language chart request ("top 5 stores selling the most",
"average temperature by store") into a chart "recipe", following the same
SPEC-FIRST philosophy as dashboard_agent: nothing is drawn here, this only
produces a declarative ChartRequest that the UI interprets.

Design rules:
1. SCHEMA-GROUNDED: never invents a column. Every dimension/measure is resolved
   against real DataFrame columns using the SAME role detection as the
   automatic dashboard (detect_column_roles), so "measure" and "dimension" mean
   the same thing here as everywhere else in the agent.
2. RULES FIRST: plain deterministic rules, no LLM. Always works, no API key,
   no quota, instant.
3. LLM ONLY AS A FALLBACK: when the rules cannot resolve the request, Gemini is
   asked to map it to real columns. The LLM only PROPOSES -- its answer is
   validated against the real schema, so an invented column is still rejected.
4. IF NOTHING RESOLVES, DO NOT GUESS: reply saying which columns do exist,
   instead of charting something arbitrary.

Requests may be in English or Spanish, and phrased loosely: "quiero ver las
ventas totales semanales" resolves the same as "show me weekly sales", without
the user having to name a column or say the word "chart".

Resolution order (first one that finds a real dimension wins):
    a) strict syntax   "[top N] [MEASURE by] DIMENSION as TYPE chart"
    b) a real column name mentioned in the text (exact, substring, or a close
       spelling such as "temperatura" -> "Temperature")
    c) a business-word synonym ("stores"/"tiendas" -> "store")
    d) the DATE column, when the user names it or the wording is temporal
       ("weekly", "semanales", "over time") -- the date column is not in
       roles["dimensions"], so it has to be offered explicitly
    e) a ranking request ("top ...") on a table with exactly ONE groupable
       column -> that is the only honest reading
    f) the LLM fallback

The measure is resolved the same way, plus: "selling"/"ventas"/"money" signal a
sales intent without naming the column, and a ranking or a time axis implies
plotting the main measure rather than counting rows.

On a time axis the chart defaults to a line, "top N" is dropped (meaningless
against a continuous timeline), and `grain` records the bucket the wording
implies -- "semanales" means weekly totals, not one point per calendar day.

Main public function:  interpret_message(text, df) -> ChatChartResult
"""

from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from crew.dashboard_agent.dashboard_agent import (
    _money_like,
    _pick_main_measure,
    detect_column_roles,
)
from utils.config import model_for


@dataclass(frozen=True)
class ChartRequest:
    """Declarative recipe for ONE chart requested through the chat.

    Shares vocabulary with build_chart_specs in dashboard_agent (type / x / y /
    agg / top_n), so a chat-requested chart is not a special case for the UI:
    it is drawn with the same logic as the automatic ones.
    """

    chart_type: str          # "bar" | "line" | "pie"
    title: str
    dimension: str           # X axis / category  (real column)
    measure: str | None = None   # Y axis (real column), or None -> row count
    agg: str = "sum"         # "sum" | "count" | "mean"
    top_n: int | None = None
    # Only meaningful when `dimension` is the date column: bucket size for the
    # time axis ("day" | "week" | "month" | "quarter" | "year"). None = leave
    # the raw dates alone.
    grain: str | None = None

    def to_spec(self) -> dict[str, Any]:
        """Convert to the same dict shape build_chart_specs produces."""
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
    """One chat turn: the chart (if any) and what to reply.

    `chart_request` is None both when the request could not be resolved AND
    when the user only wanted a number ("what is the sum of weekly sales") --
    in that second case `reply` carries the answer, so the UI simply shows the
    text and adds no chart.
    """

    chart_request: ChartRequest | None
    reply: str


# =====================================================================
# 1. STRICT SYNTAX:  "[top N] [MEASURE by] DIMENSION as TYPE chart"
# =====================================================================
_TYPE_PATTERN = re.compile(
    r"\bas\s+(?:an?\s+)?(pie|bar|line)(?:\s+chart)?\s*$", re.IGNORECASE
)
_TOP_N_PATTERN = re.compile(r"\btop\s+(\d{1,2})\b", re.IGNORECASE)
_BY_SPLIT = re.compile(r"\bby\b", re.IGNORECASE)


def _match_column(phrase: str, columns: tuple[str, ...]) -> str | None:
    """Resolve free text to EXACTLY one real column. Never invents one.

    Exact case-insensitive match first, then a substring match in either
    direction (so "sales" can find a "Weekly_Sales" column).
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
    """Try the strict form. Returns None if the text does not fit it."""
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
            # Named a measure that is not a real column -> drop, do not guess.
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
# 2. LOOSE NATURAL LANGUAGE (rule-based fallbacks)
# =====================================================================
_TOP_KEYWORDS = re.compile(
    r"\b(top|most|highest|largest|best|leading|biggest"
    r"|mejores?|mayores?|m[ai]s|principales|primeros?)\b",
    re.IGNORECASE,
)
_TYPE_KEYWORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pie", re.compile(r"\b(pie|pastel|torta|circular)\b", re.IGNORECASE)),
    (
        "line",
        re.compile(
            r"\b(line|trend|over\s+time|l[ai]nea|l[ai]neas|tendencia)\b",
            re.IGNORECASE,
        ),
    ),
    ("bar", re.compile(r"\b(bar|barras?)\b", re.IGNORECASE)),
)
_WORD_PATTERN = re.compile(r"[a-z0-9_]+")
_DEFAULT_TOP_N = 10

# "selling"/"revenue"/... express a sales intent without naming the column.
_SELLING_INTENT = re.compile(
    r"\b(selling|sold|sells?|sales|revenue|earn(?:ing)?s?|profit(?:s|able)?"
    r"|money|dinero|ventas?|ingresos?)\b",
    re.IGNORECASE,
)

# "average temperature by store" asks for a MEAN, not a sum. Summing
# temperatures (or prices, or percentages) is meaningless.
_AVG_INTENT = re.compile(r"\b(average|avg|mean|promedio|medi[ao])\b", re.IGNORECASE)

# Time wording. The date column lives in roles["date"], NOT in
# roles["dimensions"], so without this nothing time-based could ever resolve:
# "weekly sales", "ventas totales semanales" and "sales over time" all failed.
_TIME_INTENT = re.compile(
    r"\b(over\s+time|trend|time\s*series|daily|weekly|monthly|quarterly|yearly"
    r"|annual(?:ly)?|per\s+(?:day|week|month|quarter|year)"
    r"|by\s+(?:day|week|month|quarter|year|date|time)"
    r"|diari[ao]s?|semanal(?:es)?|mensual(?:es)?|trimestral(?:es)?|anual(?:es)?"
    r"|por\s+(?:d[ai]a|semana|mes|trimestre|a[nx]o|fecha)"
    r"|en\s+el\s+tiempo|tendencia|evoluci[ox]n|fecha|date)\b",
    re.IGNORECASE,
)

# How finely to bucket a date axis. "ventas semanales" means weekly totals, not
# one point per calendar day, so raw dates get resampled before aggregating.
_GRAIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("year", re.compile(r"\b(yearly|annual(?:ly)?|per\s+year|by\s+year"
                        r"|anual(?:es)?|por\s+a[nx]o)\b", re.IGNORECASE)),
    ("quarter", re.compile(r"\b(quarterly|per\s+quarter|by\s+quarter"
                           r"|trimestral(?:es)?|por\s+trimestre)\b", re.IGNORECASE)),
    ("month", re.compile(r"\b(monthly|per\s+month|by\s+month"
                         r"|mensual(?:es)?|por\s+mes)\b", re.IGNORECASE)),
    ("week", re.compile(r"\b(weekly|per\s+week|by\s+week"
                        r"|semanal(?:es)?|por\s+semana)\b", re.IGNORECASE)),
    ("day", re.compile(r"\b(daily|per\s+day|by\s+day|diari[ao]s?"
                       r"|por\s+d[ai]a)\b", re.IGNORECASE)),
)


def _time_grain(text: str) -> str | None:
    """Coarsest first, so "monthly" wins over a stray "day" elsewhere."""
    for grain, pattern in _GRAIN_PATTERNS:
        if pattern.search(text):
            return grain
    return None

# Business words that commonly stand in for the real column name. Each entry
# only ever resolves to a column that exists AND is already classified as a
# dimension: it widens the vocabulary, it never invents a column.
_DIMENSION_SYNONYMS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\bsellers?\b|\bselling\b|\bsalespeople\b|\bsales\s*reps?\b"
                   r"|\bvendedor(?:es)?\b|\bvendedora?s?\b",
                   re.IGNORECASE),
        ("seller", "salesperson", "sales_rep", "rep", "agent", "vendedor"),
    ),
    (
        re.compile(r"\bstores?\b|\bshops?\b|\bbranch(?:es)?\b|\blocations?\b|\boutlets?\b"
                   r"|\btiendas?\b|\bsucursal(?:es)?\b|\blocal(?:es)?\b",
                   re.IGNORECASE),
        ("store", "shop", "branch", "location", "outlet", "tienda", "sucursal"),
    ),
    (
        re.compile(r"\bproducts?\b|\bitems?\b|\bskus?\b|\bproductos?\b|\bart[ix]culos?\b",
         re.IGNORECASE),
        ("product", "item", "sku", "producto"),
    ),
    (
        re.compile(r"\bcustomers?\b|\bclients?\b|\bbuyers?\b|\bclientes?\b|\bcompradores?\b",
         re.IGNORECASE),
        ("customer", "client", "buyer", "cliente"),
    ),
    (
        re.compile(r"\bregions?\b|\bareas?\b|\bzones?\b|\bterritor\w*\b|\bregi[ox]n(?:es)?\b|\bzonas?\b",
         re.IGNORECASE),
        ("region", "area", "zone", "territory", "zona"),
    ),
    (
        re.compile(r"\bcategor\w*\b|\btipos?\b", re.IGNORECASE),
        ("category", "type", "categoria"),
    ),
    (
        re.compile(r"\bcountr\w*\b|\bpais\b", re.IGNORECASE),
        ("country", "pais"),
    ),
    (
        re.compile(r"\bholidays?\b|\bfestivos?\b|\bferiados?\b|\bvacaciones\b", re.IGNORECASE),
        ("holiday", "festivo"),
    ),
    (
        re.compile(r"\bstatus\b|\bstate\b|\bestado\b", re.IGNORECASE),
        ("status", "state", "estado"),
    ),
)


def _mentioned_columns(text: str, columns: tuple[str, ...]) -> list[str]:
    """Real columns named in the text, in order of appearance."""
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
        if index == -1:
            # Last resort: near-spellings. Catches the same word across
            # languages ("temperatura" -> "Temperature") and small typos, which
            # plain substring matching misses. The cutoff is deliberately high
            # so unrelated words never match -- this must not start guessing.
            for word in words:
                if len(word) < 4:
                    continue
                if difflib.SequenceMatcher(None, word, needle).ratio() >= 0.82:
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
    """A real dimension column found through a business word, if any."""
    for trigger, needles in _DIMENSION_SYNONYMS:
        if not trigger.search(text):
            continue
        for column in dimensions:
            lowered = column.lower()
            if any(needle in lowered for needle in needles):
                return column
    return None


def _explicit_chart_type(text: str) -> bool:
    """Whether the user actually named a chart type, rather than defaulting."""
    return any(pattern.search(text) for _, pattern in _TYPE_KEYWORDS)


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


# =====================================================================
# CHART OR JUST AN ANSWER?
# =====================================================================
# Some requests want a picture, others only want the number. "what is the sum
# of weekly sales" should answer in words; "quiero un grafico de ventas" should
# draw. Both still report the figures.
_CHART_INTENT = re.compile(
    r"\b(chart|graph|plot|visuali[sz]e|visuali[sz]aci[ox]n|draw|show\s+me\s+a"
    r"|gr[ai]fic[ao]s?|graficar|diagrama|barras?|pastel|torta|l[ai]neas?)\b",
    re.IGNORECASE,
)

# Explicitly does NOT want a picture.
_NO_CHART_INTENT = re.compile(
    r"\b(no\s+(?:graph|chart|grafico|gr[ai]fica)|sin\s+gr[ai]fic[ao]"
    r"|just\s+the\s+(?:number|total|answer)|only\s+the\s+(?:number|total)"
    r"|solo\s+el\s+(?:n[ux]mero|total)|s[ox]lo\s+el\s+(?:n[ux]mero|total))\b",
    re.IGNORECASE,
)

# Phrased as a question about a value rather than a request for a view.
_QUESTION_INTENT = re.compile(
    r"(^|\b)(what\s+is|what'?s|what\s+are|how\s+much|how\s+many|tell\s+me"
    r"|cu[aá]nto?s?|cu[aá]l\s+es|cuales\s+son|qu[eé]\s+es|dime)\b",
    re.IGNORECASE,
)


def _wants_chart(text: str) -> bool:
    """Whether to draw. Explicit wording wins; otherwise a question means text."""
    if _NO_CHART_INTENT.search(text):
        return False
    if _CHART_INTENT.search(text):
        return True
    return not _QUESTION_INTENT.search(text)


def _format_number(value: float, column: str = "") -> str:
    """Readable figure, with a currency mark when the column smells like money.

    The dollar sign is escaped because Streamlit renders markdown, where a pair
    of unescaped "$" opens a LaTeX math span -- two amounts in one sentence
    swallowed the text between them and dropped the bold formatting.
    """
    prefix = r"\$" if column and _money_like(column) else ""
    if value != value:  # NaN
        return "n/a"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{prefix}{int(value):,}"
    return f"{prefix}{value:,.2f}"


def _answer_sentence(df: pd.DataFrame, request: ChartRequest) -> str:
    """Plain-English answer carrying the actual numbers behind the chart.

    Kept defensive: an empty group, an all-null measure or a single category
    must degrade to a shorter sentence rather than raising and losing the reply.
    """
    column, measure = request.dimension, request.measure
    try:
        if measure is None:
            rows = len(df)
            groups = int(df[column].nunique(dropna=True))
            counts = df[column].value_counts(dropna=True)
            if counts.empty:
                return f"{_format_number(rows)} rows."
            return (
                f"{_format_number(rows)} rows across {groups} {column} values; "
                f"most common is **{counts.index[0]}** with "
                f"{_format_number(int(counts.iloc[0]))}."
            )

        values = pd.to_numeric(df[measure], errors="coerce")
        if values.notna().sum() == 0:
            return f"No numeric values found in {measure}."

        # A time axis reads as a span, not as a winner.
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            dates = df[column].dropna()
            span = ""
            if not dates.empty:
                span = f" between {dates.min().date()} and {dates.max().date()}"
            if request.agg == "mean":
                return f"Average {measure} is {_format_number(values.mean(), measure)}{span}."
            return f"Total {measure} is {_format_number(values.sum(), measure)}{span}."

        grouped = values.groupby(df[column])
        agg = grouped.mean() if request.agg == "mean" else grouped.sum()
        agg = agg.dropna()
        if agg.empty:
            return f"Total {measure} is {_format_number(values.sum(), measure)}."
        leader, best = agg.idxmax(), agg.max()

        if request.agg == "mean":
            return (
                f"Average {measure} is {_format_number(values.mean(), measure)} overall; "
                f"highest for {column} **{leader}** at {_format_number(best, measure)}."
            )
        total = values.sum()
        share = f" ({best / total * 100:.0f}% of the total)" if total else ""
        return (
            f"Total {measure} is {_format_number(total, measure)}; "
            f"{column} **{leader}** leads with {_format_number(best, measure)}{share}."
        )
    except Exception:  # never lose the reply over a formatting edge case
        return ""


def _scalar_answer(df: pd.DataFrame, measure: str, message: str) -> str:
    """Answer an aggregate question that needs no grouping at all.

    "what is the total sales" wants one number. Requiring a dimension for that
    forced the question into a refusal, even though the measure was obvious.
    """
    try:
        values = pd.to_numeric(df[measure], errors="coerce")
        if values.notna().sum() == 0:
            return f"No numeric values found in {measure}."
        if _AVG_INTENT.search(message):
            return f"Average {measure} is {_format_number(values.mean(), measure)}."
        return (
            f"Total {measure} is {_format_number(values.sum(), measure)} "
            f"across {len(df):,} rows."
        )
    except Exception:
        return ""


def _confirmation(request: ChartRequest) -> str:
    """State explicitly WHAT was charted, so a misreading is visible."""
    scope = f"top {request.top_n} " if request.top_n else ""
    if request.measure is None:
        what = "row count"
    else:
        label = {"sum": "total", "mean": "average", "count": "count of"}.get(
            request.agg, request.agg
        )
        what = f"{label} **{request.measure}**"
    per = f" per {request.grain}" if request.grain else ""
    return (
        f"Added: {what} by {scope}**{request.dimension}**{per}, "
        f"as a **{request.chart_type}** chart."
    )


def _cant_resolve_reply(roles: dict) -> str:
    """Guide the user with examples built from THEIR columns, not placeholders.

    The previous version listed the columns and then showed a generic
    "top 5 <column> by <measure>" template, leaving the user to assemble it.
    Real column names in real sentences are far easier to act on.
    """
    dimensions = roles["dimensions"]
    measures = roles["measures"]
    date_col = roles["date"]

    lines = ["I'm not sure which columns you meant. Here's what this table has:", ""]
    if dimensions:
        lines.append(f"- **Group by:** {', '.join(dimensions[:8])}")
    if date_col:
        lines.append(f"- **Time:** {date_col}")
    if measures:
        lines.append(f"- **Measure:** {', '.join(measures[:8])}")

    # Concrete, copy-pasteable examples using this table's own names.
    dim = dimensions[0] if dimensions else None
    measure = _pick_main_measure(measures) if measures else None
    examples: list[str] = []
    if dim and measure:
        examples.append(f"total {measure} by {dim}")
        examples.append(f"top 5 {dim} by {measure}")
    elif dim:
        examples.append(f"{dim}")
    if date_col and measure:
        examples.append(f"{measure} over time")
    if measure:
        examples.append(f"what is the total {measure}")

    if examples:
        lines += ["", "Try one of these:"]
        lines += [f"- `{e}`" for e in examples]
    lines.append("")
    lines.append(
        "You can ask in English or Spanish, and say *no graph* if you only "
        "want the number."
    )
    return "\n".join(lines)


# =====================================================================
# 3. LLM FALLBACK (only when the rules did not understand)
# =====================================================================
# Model comes from utils.config (DATACHEF_MODEL_CHAT, then DATACHEF_MODEL, then
# a default). The default is deliberately a different model from the
# transformation agent's: on the free tier quota is per model, and this call is
# optional, so it must not eat the quota the transformation needs.


def _llm_client():
    """Gemini client, or None if there is no key/SDK. Never raises."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai

        return genai.Client(api_key=api_key)
    except Exception:
        return None


def _extract_json(text: str) -> dict | None:
    """Pull the JSON object out of the reply, even inside a ```json block."""
    if not text:
        return None
    clean = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(clean[start : end + 1])
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _interpret_with_llm(
    message: str, df: pd.DataFrame, roles: dict
) -> ChartRequest | None:
    """Ask the LLM to translate the request into REAL columns.

    The LLM only PROPOSES; everything is validated against the real schema
    here, so ending up with an invented column stays impossible. If there is no
    key, the call fails, or the answer does not validate -> None, and the
    rule-based refusal is used instead.
    """
    client = _llm_client()
    if client is None:
        return None

    def role_of(column: str) -> str:
        if column in roles["dimensions"]:
            return "dimension"
        if column in roles["measures"]:
            return "measure"
        return "date" if column == roles["date"] else "other"

    described = "\n".join(
        f"- {c} (dtype={df[c].dtype}, role={role_of(c)})" for c in df.columns
    )
    prompt = f"""You translate chart requests into a specification.

REAL columns of the table (no others exist):
{described}

User request:
"{message}"

Return ONLY a JSON object with exactly this shape:
{{"dimension": "<column to group by>",
  "measure": "<numeric column to aggregate, or null to count rows>",
  "chart_type": "bar" | "line" | "pie",
  "agg": "sum" | "mean" | "count",
  "top_n": <integer 1-50, or null>,
  "grain": "day" | "week" | "month" | "quarter" | "year" | null}}

Rules:
- "dimension" and "measure" MUST be exact names from the list above.
- The request may be in ANY language, and phrased loosely. Treat any question
  about the data as a chart request: "quiero ver las ventas totales semanales"
  and "show me weekly sales" are the same request. Do not refuse just because
  the user did not use the word "chart".
- For anything over time, use the column whose role is "date" as the dimension,
  chart_type="line", and set "grain" to the bucket the user implies
  (semanal/weekly -> "week", mensual/monthly -> "month", ...).
- "grain" only applies when the dimension is the date column; otherwise null.
- If the user asks for "top N" or "the ones that sell the most", set top_n and
  order by the measure.
- Use agg="mean" for an average, "sum" for totals.
- If there is no clear measure, use measure=null and agg="count".
- Only return {{"dimension": null}} when the request genuinely cannot be
  answered with these columns.
- No explanations, only the JSON."""

    try:
        response = client.models.generate_content(
            model=model_for("chat"), contents=prompt
        )
        data = _extract_json(response.text)
    except Exception:
        return None
    if not data:
        return None

    columns = {str(c) for c in df.columns}
    dimension = data.get("dimension")
    if not isinstance(dimension, str) or dimension not in columns:
        return None

    measure = data.get("measure")
    if measure is not None and (not isinstance(measure, str) or measure not in columns):
        measure = None

    chart_type = data.get("chart_type")
    if chart_type not in ("bar", "line", "pie"):
        chart_type = "bar"

    agg = data.get("agg")
    if agg not in ("sum", "count", "mean"):
        agg = "sum" if measure else "count"
    if measure is None:
        agg = "count"

    top_n = data.get("top_n")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or not 1 <= top_n <= 50:
        top_n = None

    # A bucket size only means anything on the date column.
    grain = data.get("grain")
    if grain not in ("day", "week", "month", "quarter", "year"):
        grain = None
    if dimension != roles["date"]:
        grain = None

    return ChartRequest(
        chart_type=chart_type,
        title=message[:1].upper() + message[1:],
        dimension=dimension,
        measure=measure,
        agg=agg,
        top_n=top_n,
        grain=grain,
    )


def _deliver(
    df: pd.DataFrame, request: ChartRequest, message: str
) -> ChatChartResult:
    """Answer in words, draw a chart, or both, depending on what was asked.

    The figures are reported either way -- a chart alone makes the reader read
    values off an axis, when the headline number is what they asked for.
    """
    answer = _answer_sentence(df, request)
    if not _wants_chart(message):
        # Question, or an explicit "no graph": text only, no chart added.
        return ChatChartResult(None, answer or _confirmation(request))
    confirmation = _confirmation(request)
    return ChatChartResult(request, f"{confirmation}\n\n{answer}" if answer else confirmation)


# =====================================================================
# 4. ORCHESTRATOR (the only function the UI calls)
# =====================================================================
def interpret_message(text: str, df: pd.DataFrame) -> ChatChartResult:
    """Turn ONE chat message into a ChartRequest, or explain why it cannot."""

    message = (text or "").strip()
    if not message:
        return ChatChartResult(
            None, 'Ask me for a chart, e.g. "top 5 region by amount as bar chart".'
        )

    columns = tuple(str(c) for c in df.columns)

    strict = _parse_strict(message, columns)
    if strict is not None:
        return _deliver(df, strict, message)

    roles = detect_column_roles(df)
    measure_priority = roles["measures"]
    measures = frozenset(measure_priority)
    dimensions = tuple(roles["dimensions"])
    ranking_intent = bool(_TOP_KEYWORDS.search(message))

    date_col = roles["date"]
    time_intent = bool(_TIME_INTENT.search(message))

    mentioned = _mentioned_columns(message, columns)
    dimension = next((c for c in mentioned if c in dimensions), None)
    if dimension is None:
        dimension = _synonym_dimension(message, dimensions)
    if dimension is None and date_col:
        # Time axis. The date column is not in roles["dimensions"], so it has to
        # be offered explicitly -- either because the user named it, or because
        # the wording is temporal ("weekly sales", "ventas totales semanales").
        if date_col in mentioned or time_intent:
            dimension = date_col
    if dimension is None and ranking_intent and len(dimensions) == 1:
        dimension = dimensions[0]
    if dimension is None and not _wants_chart(message):
        # A plain aggregate question ("what is the total sales", "cuanto es el
        # total de ventas") needs no grouping -- just the number.
        named = next((c for c in mentioned if c in measures), None)
        if named is None and measure_priority and _SELLING_INTENT.search(message):
            named = _pick_main_measure(measure_priority)
        if named:
            answer = _scalar_answer(df, named, message)
            if answer:
                return ChatChartResult(None, answer)

    if dimension is None:
        # The rules did not understand. Ask the LLM, which does know that "the
        # stores that sell the most" means group by Store and sum Weekly_Sales
        # even when the user never writes those names.
        proposal = _interpret_with_llm(message, df, roles)
        if proposal is not None:
            return _deliver(df, proposal, message)
        return ChatChartResult(None, _cant_resolve_reply(roles))

    on_time_axis = dimension == date_col

    measure = next((c for c in mentioned if c in measures and c != dimension), None)
    if measure is None and measure_priority:
        # No measure named. Use the main measure when the message talks about
        # sales/money, asks for a ranking ("top 5 stores"), or plots over time:
        # in those, counting rows is almost never what was meant.
        if _SELLING_INTENT.search(message) or ranking_intent or on_time_axis:
            measure = _pick_main_measure(measure_priority)

    if measure is None:
        agg = "count"
    elif _AVG_INTENT.search(message):
        agg = "mean"
    else:
        agg = "sum"

    # A time axis reads as a line unless the user explicitly asked otherwise,
    # and a "top N" makes no sense against a continuous timeline.
    chart_type = _chart_type(message)
    if on_time_axis and not _explicit_chart_type(message):
        chart_type = "line"
    top_n = None if on_time_axis else _top_n(message)

    request = ChartRequest(
        chart_type=chart_type,
        title=message[:1].upper() + message[1:],
        dimension=dimension,
        measure=measure,
        agg=agg,
        top_n=top_n,
        grain=_time_grain(message) if on_time_axis else None,
    )
    return _deliver(df, request, message)
