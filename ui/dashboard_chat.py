"""
UI del chat del Dashboard.
Deja al usuario pedir graficos extra en lenguaje natural sobre la capa gold.
Consume crew.dashboard_agent.chat_intent (reglas, sin LLM) y dibuja con la
MISMA funcion de agregacion que los graficos automaticos (ui.charts._aggregate),
asi un grafico pedido por chat no es un caso especial.
"""

import plotly.express as px
import streamlit as st

from crew.dashboard_agent.chat_intent import ChartRequest, interpret_message
from ui.charts import _aggregate, _shades

_HISTORY_KEY = "dashboard_chat_history"
_REQUESTS_KEY = "dashboard_chat_requests"


def _render_chart(df, request: ChartRequest, color: str, key: str) -> None:
    spec = request.to_spec()
    data = _aggregate(df, spec)
    # La columna Y es "count" en graficos de conteo, o la medida real.
    ycol = "count" if (spec.get("agg") == "count" or spec.get("y") is None) else spec["y"]

    st.subheader(request.title)
    if request.chart_type == "line":
        fig = px.line(data, x=request.dimension, y=ycol, markers=True,
                      color_discrete_sequence=[color])
    elif request.chart_type == "pie":
        fig = px.pie(data, names=request.dimension, values=ycol,
                     color_discrete_sequence=_shades(color, len(data)))
    else:
        fig = px.bar(data, x=request.dimension, y=ycol,
                     color_discrete_sequence=[color])
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_dashboard_chat(df, color: str = "#22D3EE") -> None:
    """Dibuja el chat y los graficos que el usuario haya pedido en el."""

    history = st.session_state.setdefault(_HISTORY_KEY, [])
    requests = st.session_state.setdefault(_REQUESTS_KEY, [])

    st.subheader("💬 Ask for a chart")
    st.caption(
        'Ask in plain language — e.g. "top 5 region by amount as bar chart", '
        '"which stores are selling the most", or just "status". '
        "DataChef only charts columns that really exist in your data."
    )

    for role, message in history:
        with st.chat_message(role):
            st.markdown(message)

    prompt = st.chat_input("e.g. top 10 products by revenue as bar chart")
    if prompt:
        history.append(("user", prompt))
        result = interpret_message(prompt, df)
        if result.chart_request is not None:
            requests.append(result.chart_request)
        history.append(("assistant", result.reply))
        st.rerun()

    if requests:
        head, ctrl = st.columns([4, 1])
        head.markdown("##### Charts from this chat")
        if ctrl.button("Clear", key="dashboard_chat_clear"):
            requests.clear()
            history.clear()
            st.rerun()
        for index, request in enumerate(requests):
            _render_chart(df, request, color, key=f"dashboard_chat_chart_{index}")
