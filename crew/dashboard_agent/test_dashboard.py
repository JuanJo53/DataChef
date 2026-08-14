"""
Tests del Dashboard Agent - DATAChef
====================================

Verifica la funcionalidad del agente de dashboards sin depender de los otros
agentes ni de una API key (todo corre con reglas, offline).

Se puede correr de DOS formas:

  1) Como script (no necesita pytest):
        python crew/dashboard_agent/test_dashboard.py

  2) Con pytest (si esta instalado):
        pytest crew/dashboard_agent/test_dashboard.py -v
"""

import os
import sys

import pandas as pd

# Permite importar el agente sin importar desde que carpeta se ejecute el test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard_agent import (  # noqa: E402
    _pick_main_measure,
    build_chart_specs,
    build_dashboard_spec,
    build_kpis,
    build_rule_based_insights,
    detect_column_roles,
)
from exporters import to_powerbi, to_tableau  # noqa: E402


# =====================================================================
# DATOS DE PRUEBA (simulan la capa GOLD que recibe el agente)
# =====================================================================
def sample_sales_df() -> pd.DataFrame:
    """Dataset de ventas: fecha + 2 medidas + 2 dimensiones."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "region": ["North", "South", "East", "West"],
            "product": ["Widget", "Gadget", "Widget", "Gizmo"],
            "units": [12, 7, 20, 5],
            "revenue": [240.0, 175.5, 400.0, 149.75],
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def sample_orders_df() -> pd.DataFrame:
    """Dataset con OTRAS columnas (order_id, amount, status...) para probar
    que el agente se adapta a un esquema distinto -> schema-flexible.
    Imita el mockup nuevo del equipo en ui/app.py."""
    df = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105],
            "customer_id": ["C-001", "C-002", "C-003", "C-003", "C-005"],
            "region": ["North", "North", "South", "South", "West"],
            "order_date": ["2024-01-03", "2024-01-05", "bad-date", "2024-01-08", "2024-01-10"],
            "amount": [120.5, 80.0, 330.0, 330.0, 95.0],
            "status": ["completed", "completed", "pending", "pending", "completed"],
        }
    )
    # to_datetime con coerce: 'bad-date' se vuelve NaT (como haria transformation.py).
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df


# =====================================================================
# TESTS
# =====================================================================
def test_detecta_roles_de_columnas():
    """Clasifica bien fecha / medidas / dimensiones."""
    roles = detect_column_roles(sample_sales_df())
    assert roles["date"] == "date"
    assert set(roles["measures"]) == {"units", "revenue"}
    assert set(roles["dimensions"]) == {"region", "product"}


def test_elige_medida_principal_de_negocio():
    """Prefiere un nombre de negocio (revenue) sobre otro (units)."""
    assert _pick_main_measure(["units", "revenue"]) == "revenue"
    assert _pick_main_measure(["amount", "count"]) == "amount"
    assert _pick_main_measure([]) is None  # sin medidas -> None


def test_kpis_con_formato_correcto():
    """El primer KPI cuenta registros; las columnas de dinero salen como currency."""
    df = sample_sales_df()
    kpis = build_kpis(df, detect_column_roles(df))

    assert kpis[0]["label"] == "Total records"
    assert kpis[0]["value"] == len(df)
    assert kpis[0]["format"] == "int"

    revenue_kpi = next(k for k in kpis if k["label"] == "Total revenue")
    assert revenue_kpi["format"] == "currency"
    assert revenue_kpi["value"] == df["revenue"].sum()


def test_genera_graficos_linea_barra_pastel():
    """Con fecha + dimensiones debe proponer linea, barras y pastel."""
    df = sample_sales_df()
    charts = build_chart_specs(df, detect_column_roles(df))
    tipos = [c["type"] for c in charts]

    assert "line" in tipos   # tendencia en el tiempo
    assert "bar" in tipos    # comparativo por dimension
    assert "pie" in tipos    # composicion

    linea = next(c for c in charts if c["type"] == "line")
    assert linea["x"] == "date"
    assert linea["y"] == "revenue"


def test_insights_por_reglas():
    """Los insights no van vacios y mencionan el total y el lider."""
    df = sample_sales_df()
    insights = build_rule_based_insights(df, detect_column_roles(df))

    assert len(insights) >= 2
    assert any("Total revenue" in frase for frase in insights)
    assert any("leads" in frase for frase in insights)


def test_spec_completo_offline():
    """build_dashboard_spec devuelve la estructura esperada, motor = reglas."""
    df = sample_sales_df()
    spec = build_dashboard_spec(df, use_llm=False)

    for llave in ("title", "engine", "meta", "roles", "kpis", "charts", "insights"):
        assert llave in spec, f"Falta la llave '{llave}' en el spec"

    assert spec["engine"] == "rule-based"
    assert spec["meta"]["rows"] == len(df)
    assert spec["meta"]["columns"] == len(df.columns)
    assert len(spec["kpis"]) >= 1
    assert len(spec["charts"]) >= 1


def test_flexibilidad_de_esquema():
    """Con columnas totalmente distintas (orders) tambien funciona."""
    df = sample_orders_df()
    roles = detect_column_roles(df)

    assert roles["date"] == "order_date"
    assert "amount" in roles["measures"]
    assert "region" in roles["dimensions"]
    assert _pick_main_measure(roles["measures"]) == "amount"

    spec = build_dashboard_spec(df, use_llm=False)
    assert len(spec["charts"]) >= 1  # produjo dashboard sobre otro esquema


def test_sin_columnas_numericas_usa_fallback():
    """Si no hay medidas, cae a un grafico de CONTEO por dimension (no vacio)."""
    df = pd.DataFrame({"region": ["N", "S", "N"], "status": ["a", "b", "a"]})
    roles = detect_column_roles(df)

    assert roles["measures"] == []
    charts = build_chart_specs(df, roles)
    assert len(charts) >= 1                    # fallback: conteo por dimension
    assert charts[0]["agg"] == "count"
    assert len(build_rule_based_insights(df, roles)) >= 1

    spec = build_dashboard_spec(df, use_llm=False)     # no debe lanzar excepcion
    assert spec["engine"] == "rule-based"


def test_export_powerbi_genera_dax():
    """El exporter de Power BI produce medidas DAX a partir del spec."""
    spec = build_dashboard_spec(sample_sales_df(), use_llm=False)
    pkg = to_powerbi(spec, table_name="Sales")
    assert any("SUM('Sales'" in m for m in pkg["dax_measures"])
    assert any("Row Count" in m for m in pkg["dax_measures"])
    assert pkg["target"] == "powerbi"


def test_export_tableau_sin_key_no_conecta():
    """Sin api_key, Tableau devuelve 'not_connected' (campo dejado abierto)."""
    spec = build_dashboard_spec(sample_sales_df(), use_llm=False)
    res = to_tableau(spec)
    assert res["status"] == "not_connected"
    assert res["package"]["target"] == "tableau"


# =====================================================================
# MODO SCRIPT: corre todos los tests e imprime PASS/FAIL (sin pytest)
# =====================================================================
if __name__ == "__main__":
    tests = {n: f for n, f in sorted(globals().items()) if n.startswith("test_")}
    print(f"Corriendo {len(tests)} tests del Dashboard Agent...\n")

    fallos = 0
    for nombre, funcion in tests.items():
        try:
            funcion()
            print(f"  PASS  {nombre}")
        except AssertionError as e:
            fallos += 1
            print(f"  FAIL  {nombre}  ->  {e}")
        except Exception as e:  # error inesperado
            fallos += 1
            print(f"  ERROR {nombre}  ->  {type(e).__name__}: {e}")

    print("\n" + "=" * 45)
    if fallos == 0:
        print(f"RESULTADO: OK - los {len(tests)} tests pasaron")
    else:
        print(f"RESULTADO: {fallos} test(s) fallaron de {len(tests)}")
    sys.exit(1 if fallos else 0)
