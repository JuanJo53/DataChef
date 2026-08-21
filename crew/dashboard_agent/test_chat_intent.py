"""Tests del chat del dashboard (crew.dashboard_agent.chat_intent).

IMPORTANTE: el fixture `sin_llm` desactiva el fallback con LLM en TODAS las
pruebas por defecto. Asi las reglas se prueban de forma determinista, sin red,
sin API key y sin gastar cupo gratuito. Las pruebas que si quieren ejercitar el
camino del LLM inyectan un cliente falso a proposito.
"""

import pandas as pd
import pytest

from crew.dashboard_agent import chat_intent
from crew.dashboard_agent.chat_intent import interpret_message

FRAME = pd.DataFrame(
    {
        "order_id": list(range(1, 10)),
        "region": ["west", "east", "north"] * 3,
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    }
)

# Una sola dimension agrupable ("store") y una medida de ventas ("revenue").
SALES_FRAME = pd.DataFrame(
    {
        "order_id": list(range(1, 10)),
        "store": ["north", "south", "east"] * 3,
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0],
    }
)

MULTI_DIM_FRAME = pd.DataFrame(
    {
        "order_id": list(range(1, 10)),
        "region": ["west", "east", "north"] * 3,
        "category": ["a", "b", "c"] * 3,
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    }
)


@pytest.fixture(autouse=True)
def sin_llm(monkeypatch):
    """Por defecto: nada de red. Solo reglas."""
    monkeypatch.setattr(chat_intent, "_llm_client", lambda: None)


# ---------------------------------------------------------------------
# Reglas
# ---------------------------------------------------------------------
def test_strict_syntax_resolves():
    result = interpret_message("top 5 amount by region as pie chart", FRAME)

    assert result.chart_request is not None
    assert result.chart_request.chart_type == "pie"
    assert result.chart_request.dimension == "region"
    assert result.chart_request.measure == "amount"
    assert result.chart_request.agg == "sum"
    assert result.chart_request.top_n == 5


def test_loose_phrasing_without_explicit_type_defaults_to_bar():
    result = interpret_message("amount by region", FRAME)

    assert result.chart_request is not None
    assert result.chart_request.chart_type == "bar"
    assert result.chart_request.dimension == "region"


def test_dimension_only_message_counts_rows():
    # Sin ranking y sin medida nombrada -> contar filas.
    result = interpret_message("region", FRAME)

    assert result.chart_request is not None
    assert result.chart_request.measure is None
    assert result.chart_request.agg == "count"


def test_ranking_without_named_measure_uses_the_main_measure():
    # "top" implica ordenar por algo; contar filas casi nunca es la intencion.
    result = interpret_message("top 3 regions", FRAME)

    assert result.chart_request is not None
    assert result.chart_request.measure == "amount"
    assert result.chart_request.agg == "sum"
    assert result.chart_request.top_n == 3


def test_average_intent_uses_mean_not_sum():
    # Sumar temperaturas/precios no significa nada.
    result = interpret_message("average amount by region", FRAME)

    assert result.chart_request is not None
    assert result.chart_request.agg == "mean"


def test_top_selling_resolves_single_dimension_and_main_measure():
    result = interpret_message("order by top sellers selling the most", SALES_FRAME)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "store"
    assert result.chart_request.measure == "revenue"
    assert result.chart_request.agg == "sum"
    assert result.chart_request.top_n == 10


def test_business_synonym_resolves_a_real_column_by_another_name():
    # "shops" no es subcadena de "store": solo resuelve via sinonimos.
    result = interpret_message("which shops are selling the most", SALES_FRAME)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "store"
    assert result.chart_request.measure == "revenue"


def test_numeric_categorical_is_groupable():
    # 'Store' numerico (ids que se repiten) debe poder usarse para agrupar,
    # y 'Weekly_Sales' seguir siendo la medida.
    df = pd.DataFrame(
        {
            "Store": [1, 2, 3] * 20,
            "Weekly_Sales": [100.0 + i for i in range(60)],
        }
    )
    result = interpret_message("top 5 stores selling the most", df)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "Store"
    assert result.chart_request.measure == "Weekly_Sales"


def test_unknown_column_is_refused_when_rules_are_alone():
    result = interpret_message("what about widgets", MULTI_DIM_FRAME)

    assert result.chart_request is None
    assert "region" in result.reply or "amount" in result.reply


def test_blank_message_asks_for_a_request():
    result = interpret_message("   ", FRAME)

    assert result.chart_request is None
    assert result.reply


def test_to_spec_matches_build_chart_specs_shape():
    result = interpret_message("top 2 amount by region as bar chart", FRAME)
    spec = result.chart_request.to_spec()

    assert spec["type"] == "bar"
    assert spec["x"] == "region"
    assert spec["y"] == "amount"
    assert spec["agg"] == "sum"
    assert spec["top_n"] == 2


# ---------------------------------------------------------------------
# Fallback con LLM (cliente falso: sigue sin tocar la red)
# ---------------------------------------------------------------------
class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, text, boom=False):
        self._text, self._boom, self.calls = text, boom, 0

    def generate_content(self, model, contents):
        self.calls += 1
        if self._boom:
            raise RuntimeError("api caida")
        return _FakeResp(self._text)


class _FakeClient:
    def __init__(self, text, boom=False):
        self.models = _FakeModels(text, boom)


def _usar_llm(monkeypatch, text, boom=False):
    fake = _FakeClient(text, boom)
    monkeypatch.setattr(chat_intent, "_llm_client", lambda: fake)
    return fake


def test_llm_rescues_a_request_the_rules_could_not_parse(monkeypatch):
    _usar_llm(
        monkeypatch,
        '```json\n{"dimension":"category","measure":"amount",'
        '"chart_type":"pie","agg":"sum","top_n":3}\n```',
    )
    result = interpret_message("break it down by product family", MULTI_DIM_FRAME)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "category"
    assert result.chart_request.measure == "amount"
    assert result.chart_request.chart_type == "pie"
    assert result.chart_request.top_n == 3


def test_llm_inventing_a_column_is_rejected(monkeypatch):
    # La proteccion clave: el LLM propone, pero se valida contra el esquema real.
    _usar_llm(monkeypatch, '{"dimension":"supplier","measure":"amount"}')
    result = interpret_message("break it down by supplier", MULTI_DIM_FRAME)

    assert result.chart_request is None


def test_llm_inventing_a_measure_falls_back_to_counting(monkeypatch):
    _usar_llm(monkeypatch, '{"dimension":"category","measure":"profit_margin"}')
    result = interpret_message("break it down by family", MULTI_DIM_FRAME)

    assert result.chart_request is not None
    assert result.chart_request.measure is None
    assert result.chart_request.agg == "count"


def test_llm_declining_is_respected(monkeypatch):
    _usar_llm(monkeypatch, '{"dimension": null}')
    result = interpret_message("what is the meaning of life", MULTI_DIM_FRAME)

    assert result.chart_request is None


def test_llm_failure_falls_back_to_the_rule_based_refusal(monkeypatch):
    _usar_llm(monkeypatch, "", boom=True)
    result = interpret_message("something unparseable", MULTI_DIM_FRAME)

    assert result.chart_request is None
    # Falls back to the guided reply, which lists this table's real columns.
    assert "region" in result.reply and "amount" in result.reply


def test_llm_is_not_called_when_the_rules_already_understood(monkeypatch):
    # No gastar cupo cuando las reglas bastan.
    fake = _usar_llm(monkeypatch, '{"dimension":"category"}')
    result = interpret_message("top 5 amount by region as bar chart", FRAME)

    assert result.chart_request is not None
    assert fake.models.calls == 0


# ---------------------------------------------------------------------
# Ambiguous / Spanish phrasing, and the time axis
# ---------------------------------------------------------------------
TIME_FRAME = pd.DataFrame(
    {
        "Store": [1, 2, 3] * 40,
        "Date": pd.date_range("2010-01-01", periods=120, freq="D"),
        "Weekly_Sales": [1000.0 + i for i in range(120)],
        "Temperature": [20.0 + (i % 30) for i in range(120)],
    }
)


@pytest.mark.parametrize(
    "message",
    [
        "quiero ver las ventas totales semanales",
        "quiero un grafico con las ventas totales semanales",
        "total sales by week",
        "weekly sales over time",
    ],
)
def test_weekly_sales_phrasings_all_resolve_without_the_llm(message):
    # The reported bug: only the phrasing containing "grafico" worked, and only
    # because it fell through to the LLM. All of these must resolve on rules.
    result = interpret_message(message, TIME_FRAME)

    assert result.chart_request is not None, message
    assert result.chart_request.dimension == "Date"
    assert result.chart_request.measure == "Weekly_Sales"
    assert result.chart_request.agg == "sum"
    assert result.chart_request.grain == "week"
    assert result.chart_request.chart_type == "line"


def test_time_axis_drops_top_n_and_defaults_to_a_line():
    # "top N" is meaningless against a continuous timeline.
    result = interpret_message("las mayores ventas mensuales", TIME_FRAME)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "Date"
    assert result.chart_request.grain == "month"
    assert result.chart_request.top_n is None
    assert result.chart_request.chart_type == "line"


def test_explicit_chart_type_still_wins_on_a_time_axis():
    result = interpret_message("ventas mensuales en barras", TIME_FRAME)

    assert result.chart_request is not None
    assert result.chart_request.chart_type == "bar"
    assert result.chart_request.grain == "month"


def test_spanish_dimension_synonym_resolves():
    result = interpret_message("ventas por tienda", TIME_FRAME)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "Store"
    assert result.chart_request.measure == "Weekly_Sales"
    assert result.chart_request.grain is None


def test_close_spelling_across_languages_resolves():
    # "temperatura" is not a substring of "Temperature"; only fuzzy matching
    # bridges the two.
    result = interpret_message("promedio de temperatura por tienda", TIME_FRAME)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "Store"
    assert result.chart_request.measure == "Temperature"
    assert result.chart_request.agg == "mean"


def test_fuzzy_matching_does_not_invent_a_column():
    # The cutoff must be tight enough that an unrelated word matches nothing.
    result = interpret_message("quiero ver los helicopteros", TIME_FRAME)

    assert result.chart_request is None


def test_grain_is_ignored_when_not_on_the_date_column():
    result = interpret_message("ventas semanales por tienda", TIME_FRAME)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "Store"
    assert result.chart_request.grain is None


# ---------------------------------------------------------------------
# Chart vs. plain answer
# ---------------------------------------------------------------------
def test_a_question_is_answered_in_words_without_adding_a_chart():
    result = interpret_message("what is the total amount", FRAME)

    assert result.chart_request is None      # nothing drawn
    assert "Total amount" in result.reply
    assert "450" in result.reply             # 10+20+...+90


def test_explicit_no_graph_is_honoured():
    result = interpret_message("amount by region no graph", FRAME)

    assert result.chart_request is None
    assert "amount" in result.reply


def test_spanish_question_is_answered_without_a_chart():
    result = interpret_message("cuanto es el total de amount", FRAME)

    assert result.chart_request is None
    assert "450" in result.reply


def test_asking_for_a_chart_still_draws_one():
    result = interpret_message("quiero un grafico de amount por region", FRAME)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "region"


def test_a_chart_reply_also_reports_the_numbers():
    # The point of the request: do not make the reader read values off an axis.
    result = interpret_message("amount by region", FRAME)

    assert result.chart_request is not None
    assert "Total amount" in result.reply
    assert "450" in result.reply


def test_average_question_answers_with_the_mean():
    result = interpret_message("what is the average amount", FRAME)

    assert result.chart_request is None
    assert "Average amount" in result.reply
    assert "50" in result.reply


def test_money_columns_are_formatted_with_a_currency_mark():
    df = pd.DataFrame({"region": ["a", "b"] * 5, "sales": [100.0] * 10})
    result = interpret_message("what is the total sales", df)

    assert "$1,000" in result.reply


def test_the_guided_reply_uses_real_column_names_not_placeholders():
    result = interpret_message("what about widgets", MULTI_DIM_FRAME)

    assert result.chart_request is None
    assert "<column>" not in result.reply
    assert "total amount by region" in result.reply
    assert "no graph" in result.reply


def test_strict_syntax_swaps_an_inverted_dimension_and_measure():
    # The grammar reads "MEASURE by DIMENSION", so "stores by temperature"
    # literally parses backwards. Roles make the intent unambiguous.
    df = pd.DataFrame(
        {
            "Store": [1, 2, 3] * 20,
            "Temperature": [20.0 + i for i in range(60)],
        }
    )
    result = interpret_message("top 3 stores by temperature as bar chart", df)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "Store"
    assert result.chart_request.measure == "Temperature"
    assert result.chart_request.top_n == 3


def test_strict_syntax_does_not_swap_when_already_correct():
    result = interpret_message("top 5 amount by region as pie chart", FRAME)

    assert result.chart_request is not None
    assert result.chart_request.dimension == "region"
    assert result.chart_request.measure == "amount"
