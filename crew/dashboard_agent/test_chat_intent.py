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
    assert "couldn't match" in result.reply


def test_llm_is_not_called_when_the_rules_already_understood(monkeypatch):
    # No gastar cupo cuando las reglas bastan.
    fake = _usar_llm(monkeypatch, '{"dimension":"category"}')
    result = interpret_message("top 5 amount by region as bar chart", FRAME)

    assert result.chart_request is not None
    assert fake.models.calls == 0
