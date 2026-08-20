"""
Dashboard chat UI.
Lets the user ask for extra charts in natural language over the gold layer.
Consumes crew.dashboard_agent.chat_intent (rules first, LLM only as a
fallback) and draws with the SAME aggregation used by the automatic charts
(ui.charts._aggregate), so a chat chart is never a special case.

The colour picker and per-chart type selector from the "Customize dashboard"
panel apply here too: the colour is read from the same session_state key the
panel writes, and each chat chart gets its own type dropdown, exactly like the
automatic ones.
"""

import plotly.express as px
import streamlit as st

from crew.dashboard_agent.chat_intent import ChartRequest, interpret_message
from ui.charts import _aggregate, _shades

_HISTORY_KEY = "dashboard_chat_history"
_REQUESTS_KEY = "dashboard_chat_requests"

# Same key the "Customize dashboard" colour picker writes to in ui/charts.py,
# so picking a colour up there also recolours the charts down here.
_SHARED_COLOR_KEY = "dash_color"
_DEFAULT_COLOR = "#22D3EE"

# Same options the automatic charts offer, so both behave identically.
_TYPE_OPTIONS = ["bar", "line", "area", "pie"]


def _shared_color() -> str:
    """Colour chosen in the 'Customize dashboard' panel, or the default."""
    return st.session_state.get(_SHARED_COLOR_KEY, _DEFAULT_COLOR)


def _render_chart(df, request: ChartRequest, chart_type: str, color: str, key: str) -> None:
    spec = request.to_spec()
    data = _aggregate(df, spec)
    # Y is "count" on counting charts, or the real measure.
    ycol = "count" if (spec.get("agg") == "count" or spec.get("y") is None) else spec["y"]

    # A numeric dimension (e.g. Store = 1..45) is a CATEGORY, not a scale.
    # Without this Plotly draws a continuous axis and the 5 bars of a "top 5"
    # end up thin and spread across the whole numeric range.
    categorical = chart_type in ("bar", "pie")
    if categorical:
        data = data.copy()
        data[request.dimension] = data[request.dimension].astype(str)

    if chart_type == "line":
        fig = px.line(data, x=request.dimension, y=ycol, markers=True,
                      color_discrete_sequence=[color])
    elif chart_type == "area":
        fig = px.area(data, x=request.dimension, y=ycol,
                      color_discrete_sequence=[color])
    elif chart_type == "pie":
        fig = px.pie(data, names=request.dimension, values=ycol,
                     color_discrete_sequence=_shades(color, len(data)))
    else:
        fig = px.bar(data, x=request.dimension, y=ycol,
                     color_discrete_sequence=[color])
        # Plotly still reads "16" as a number and would revert to a continuous
        # axis even though the data is text, so declare the axis categorical.
        fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_dashboard_chat(df) -> None:
    """Render the chat plus any charts the user has requested in it."""

    history = st.session_state.setdefault(_HISTORY_KEY, [])
    requests = st.session_state.setdefault(_REQUESTS_KEY, [])

    st.subheader("Ask for a chart")
    st.caption(
        'Ask in plain language — e.g. "top 5 stores selling the most", '
        '"average temperature by store", or just "status". '
        "DataChef only charts columns that really exist in your data."
    )

    for role, message in history:
        with st.chat_message(role):
            st.markdown(message)

    prompt = st.chat_input("e.g. top 10 products by revenue as bar chart")
    if prompt:
        history.append(("user", prompt))
        # Paint the user's message BEFORE interpreting, and cover the slow path
        # (when rules fail and the LLM is consulted) with a spinner. Without
        # this the page just looked frozen.
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Interpreting your request..."):
            result = interpret_message(prompt, df)
        if result.chart_request is not None:
            requests.append(result.chart_request)
        history.append(("assistant", result.reply))
        st.rerun()

    if not requests:
        return

    head, ctrl = st.columns([4, 1])
    head.markdown("##### Charts from this chat")
    if ctrl.button("Clear", key="dashboard_chat_clear"):
        requests.clear()
        history.clear()
        st.rerun()

    color = _shared_color()
    for index, request in enumerate(requests):
        # Title plus a per-chart type selector, mirroring the automatic charts.
        title_col, type_col = st.columns([4, 1])
        title_col.subheader(request.title)
        default_type = (
            request.chart_type if request.chart_type in _TYPE_OPTIONS else "bar"
        )
        chart_type = type_col.selectbox(
            "Type",
            _TYPE_OPTIONS,
            index=_TYPE_OPTIONS.index(default_type),
            key=f"dashboard_chat_type_{index}",
            label_visibility="collapsed",
        )
        _render_chart(
            df, request, chart_type, color, key=f"dashboard_chat_chart_{index}"
        )
