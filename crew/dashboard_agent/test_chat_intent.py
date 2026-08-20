"""Tests del chat del dashboard (crew.dashboard_agent.chat_intent).

Los fixtures repiten categorias a proposito: detect_column_roles solo trata
una columna de texto como dimension si tiene <= 30 valores distintos, y
_looks_like_id descarta columnas *_id como medida. Estos datos cumplen ambas
reglas, igual que los datos reales que llegan a la capa gold.
"""

import pandas as pd

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
    result = interpret_message("region", FRAME)

    assert result.chart_request is not None
    assert result.chart_request.measure is None
    assert result.chart_request.agg == "count"


def test_top_selling_resolves_single_dimension_and_main_measure():
    # Ni "store" ni "revenue" se nombran: "store" es la unica columna
    # agrupable y "selling" indica intencion de ventas.
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


def test_unknown_column_is_refused_not_guessed():
    # Dos dimensiones reales y el mensaje no nombra ninguna ni un sinonimo:
    # elegir una seria adivinar.
    result = interpret_message("what about widgets", MULTI_DIM_FRAME)

    assert result.chart_request is None
    assert "region" in result.reply or "amount" in result.reply


def test_blank_message_asks_for_a_request():
    result = interpret_message("   ", FRAME)

    assert result.chart_request is None
    assert result.reply


def test_to_spec_matches_build_chart_specs_shape():
    # El dict debe ser consumible por ui.charts._aggregate igual que los
    # graficos automaticos.
    result = interpret_message("top 2 amount by region as bar chart", FRAME)
    spec = result.chart_request.to_spec()

    assert spec["type"] == "bar"
    assert spec["x"] == "region"
    assert spec["y"] == "amount"
    assert spec["agg"] == "sum"
    assert spec["top_n"] == 2
