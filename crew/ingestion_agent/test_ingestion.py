"""
Tests del Ingestion Agent - DATAChef
====================================

Corre offline (sin API key). Dos formas:
  python crew/ingestion_agent/test_ingestion.py
  pytest crew/ingestion_agent/test_ingestion.py -v
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion_agent import (  # noqa: E402
    _sql_type,
    answer_question,
    build_ingestion_report,
    detect_issues,
    profile_columns,
    suggest_pk,
)


def sample_df() -> pd.DataFrame:
    """Dataset con: PK unica, columna PII, un nulo, fechas y numeros."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com", None],
            "region": ["N", "S", "N", "S", "W"],
            "amount": [10.5, 20.0, 30.0, 40.0, 50.0],
            "signup_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )


# ---------------------------------------------------------------------
def test_infiere_tipos_sql():
    assert _sql_type(pd.Series([1, 2, 3])) in ("INT", "BIGINT")
    assert _sql_type(pd.Series([1.5, 2.5])) == "DECIMAL(18,2)"
    assert _sql_type(pd.to_datetime(pd.Series(["2024-01-01"]))) == "DATETIME2"
    assert _sql_type(pd.Series(["hola", "mundo"])).startswith("NVARCHAR")


def test_perfil_por_columna():
    perfil = profile_columns(sample_df())
    por_nombre = {c["name"]: c for c in perfil}
    assert por_nombre["email"]["nulls"] == 1
    assert por_nombre["customer_id"]["is_pk_candidate"] is True
    assert por_nombre["email"]["is_pii"] is True


def test_llave_primaria_y_ddl():
    df = sample_df()
    perfil = profile_columns(df)
    assert suggest_pk(perfil) == "customer_id"

    report = build_ingestion_report(df, "customers")
    assert "PRIMARY KEY CLUSTERED ([customer_id])" in report["sql"]["create_table"]
    assert "CREATE TABLE [customers]" in report["sql"]["create_table"]


def test_detecta_nulos_y_pii():
    issues = detect_issues(sample_df(), profile_columns(sample_df()))
    ids = [i["id"] for i in issues]
    assert "nulls_email" in ids          # nulo detectado
    assert "pii_email" in ids            # PII detectada


def test_detecta_duplicados():
    df = pd.DataFrame({"a": [1, 1], "b": ["x", "x"]})  # fila repetida
    issues = detect_issues(df, profile_columns(df))
    assert any(i["id"] == "dup_rows" for i in issues)


def test_salud_del_dato():
    health = build_ingestion_report(sample_df())["health"]
    assert 0 <= health["score"] <= 100
    assert health["grade"] in ("A", "B", "C", "D")
    assert "email" in health["pii_columns"]


def test_alertas_sugeridas():
    alerts = build_ingestion_report(sample_df(), "customers")["alerts"]
    tipos = {a["type"] for a in alerts}
    assert "Completeness" in tipos       # por el nulo en email
    assert "Uniqueness" in tipos         # por la PK
    assert "Timeliness" in tipos         # por signup_date


def test_chat_responde_por_reglas():
    df = sample_df()
    assert "5" in answer_question(df, "how many rows are there?")
    assert "region" in answer_question(df, "which columns do I have?")
    assert "email" in answer_question(df, "where are the missing values?")
    assert "duplicate" in answer_question(df, "are there duplicates?").lower()


def test_reporte_completo():
    report = build_ingestion_report(sample_df(), "customers")
    for llave in ("meta", "health", "cards", "columns", "issues",
                  "primary_key", "indexes", "sql", "alerts"):
        assert llave in report, f"Falta la llave '{llave}'"
    assert report["meta"]["table_name"] == "customers"
    assert len(report["cards"]) >= 4


# ---------------------------------------------------------------------
if __name__ == "__main__":
    tests = {n: f for n, f in sorted(globals().items()) if n.startswith("test_")}
    print(f"Corriendo {len(tests)} tests del Ingestion Agent...\n")
    fallos = 0
    for nombre, funcion in tests.items():
        try:
            funcion()
            print(f"  PASS  {nombre}")
        except AssertionError as e:
            fallos += 1
            print(f"  FAIL  {nombre}  ->  {e}")
        except Exception as e:
            fallos += 1
            print(f"  ERROR {nombre}  ->  {type(e).__name__}: {e}")
    print("\n" + "=" * 45)
    print(f"RESULTADO: OK - los {len(tests)} tests pasaron" if not fallos
          else f"RESULTADO: {fallos} fallaron de {len(tests)}")
    sys.exit(1 if fallos else 0)
