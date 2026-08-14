"""
Exporters del Dashboard Agent - DATAChef
========================================

El agente produce un DashboardSpec NEUTRAL (kpis, charts, insights, roles).
Este modulo lo traduce a distintos destinos SIN tocar el agente:

    DashboardSpec ─┬─> Streamlit   (ui/charts.py, ya implementado)
                   ├─> Power BI    (medidas DAX + relaciones)  -> to_powerbi()
                   └─> Tableau     (definicion de hojas)        -> to_tableau()

Asi, "abrirlo a Power BI / Tableau mas tarde" no requiere reescribir nada:
solo se agrega/completa un exporter. El campo `api_key` de Tableau queda
declarado para cuando quieras conectar.
"""

from __future__ import annotations

from typing import Any


# =====================================================================
# POWER BI
# =====================================================================
def _dax_measure(table: str, measure: str) -> str:
    """Convierte una medida del spec en una medida DAX de Power BI."""
    label = measure.replace("_", " ").title()
    return f"Total {label} = SUM('{table}'[{measure}])"


def to_powerbi(spec: dict, table_name: str = "DataChef",
               connect: bool = False, mcp_client: Any = None) -> dict:
    """Traduce el spec a un paquete Power BI-friendly.

    Devuelve:
      - dax_measures: lista de medidas DAX listas para pegar en Power BI.
      - relationships: relaciones sugeridas (dimension -> tabla dimension).

    Si `connect=True` y se pasa un `mcp_client` (el servidor
    powerbi-modeling-mcp), aqui se crearian en vivo. Se deja el gancho marcado.
    """
    roles = spec.get("roles", {})
    measures = roles.get("measures", [])
    dimensions = roles.get("dimensions", [])

    dax_measures = [f"Row Count = COUNTROWS('{table_name}')"]
    dax_measures += [_dax_measure(table_name, m) for m in measures]

    relationships = [
        {
            "from": f"'{table_name}'[{d}]",
            "to": f"'Dim_{d.title()}'[{d}]",
            "cardinality": "many-to-one",
            "cross_filter": "single",
        }
        for d in dimensions
    ]

    package = {
        "target": "powerbi",
        "table": table_name,
        "dax_measures": dax_measures,
        "relationships": relationships,
    }

    if connect and mcp_client is not None:
        # ── GANCHO PARA powerbi-modeling-mcp ──────────────────────────
        # Aqui se empujaria el modelo en vivo, por ejemplo:
        #   for dax in dax_measures:
        #       mcp_client.measure_operations(action="create", expression=dax, ...)
        #   for rel in relationships:
        #       mcp_client.relationship_operations(action="create", **rel)
        # Se deja sin ejecutar para no tocar un modelo real por accidente.
        raise NotImplementedError(
            "Live Power BI push not wired yet — connect powerbi-modeling-mcp here."
        )

    return package


# =====================================================================
# TABLEAU  (campo api_key dejado abierto para el futuro)
# =====================================================================
def to_tableau(spec: dict, api_key: str | None = None,
               server_url: str | None = None) -> dict:
    """Traduce el spec a una definicion de hojas de Tableau.

    El parametro `api_key` (y `server_url`) queda declarado para cuando
    quieras conectar con la Tableau REST / Hyper API. Mientras no se pase,
    solo devuelve la definicion exportable (status 'not_connected').
    """
    sheets = [
        {
            "title": c["title"],
            "type": c["type"],
            "x": c.get("x"),
            "y": c.get("y"),
            "aggregation": c.get("agg", "sum"),
        }
        for c in spec.get("charts", [])
    ]
    package = {"target": "tableau", "sheets": sheets}

    if not api_key:
        return {"status": "not_connected",
                "reason": "No Tableau API key provided.",
                "package": package}

    # ── GANCHO PARA Tableau REST/Hyper API ────────────────────────────
    # Con api_key + server_url, aqui iria la publicacion real del workbook.
    raise NotImplementedError(
        "Tableau connection pending — api_key received, wire the REST API here."
    )
