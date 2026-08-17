"""DataChef product entry point: a thin Streamlit shell over the controller.

Streamlit renders evidence and collects input. It decides nothing. Every
question that matters — is the plan valid, does approval match, did quality
assurance pass, may this user download gold — is answered by
``DataChefController`` and simply displayed here.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from datachef.application import ScreenId
from ui import state as ui_state
from ui.screens import render_screen


LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "img",
    "Gemini_Generated_Image_adhg9madhg9madhg.png",
)

_PROGRESS = (
    (ScreenId.UPLOAD, "1 · Upload"),
    (ScreenId.INTENT, "2 · Intent"),
    (ScreenId.PLAN, "3 · Plan"),
    (ScreenId.APPROVAL, "4 · Approval"),
    (ScreenId.QA, "5 · Quality"),
    (ScreenId.RESULTS, "6 · Results"),
)

_ORDER = {screen: index for index, (screen, _) in enumerate(_PROGRESS)}
_ORDER[ScreenId.DIAGNOSE] = 0


def _render_sidebar(controller, state) -> None:
    session = controller.session
    st.sidebar.title("DataChef")
    st.sidebar.caption("Offline, deterministic, human-approved data preparation.")

    current = _ORDER.get(session.screen, 0)
    st.sidebar.markdown("### Progress")
    for screen, label in _PROGRESS:
        position = _ORDER[screen]
        if position < current:
            st.sidebar.markdown(f"✅ {label}")
        elif position == current:
            st.sidebar.markdown(f"**➡️ {label}**")
        else:
            st.sidebar.markdown(f"◻️ {label}")

    st.sidebar.markdown("---")
    preview = st.sidebar.toggle(
        "Show local data preview",
        value=session.preview_enabled,
        key=ui_state.PREVIEW_WIDGET,
        help=(
            "Preview rows are presentation only. They never enter evidence, "
            "artifacts, the manifest, or any provider context."
        ),
    )
    if preview != session.preview_enabled:
        controller.set_preview_enabled(preview)

    st.sidebar.markdown("---")
    if st.sidebar.button(
        "Reset session",
        key=ui_state.RESET_WIDGET,
        use_container_width=True,
    ):
        ui_state.request_reset(state)
        st.rerun()
    st.sidebar.caption(
        f"Revision {session.revision} · {len(session.command_history)} command(s) recorded"
    )


def run_app() -> None:
    st.set_page_config(page_title="DataChef", layout="wide")

    state = st.session_state
    # Reset runs before any widget is instantiated so widget keys can be cleared.
    ui_state.apply_pending_reset(state)

    controller = ui_state.get_controller(state)

    if os.path.exists(LOGO_PATH):
        st.logo(LOGO_PATH, size="large")

    _render_sidebar(controller, state)

    session = controller.session
    st.title("DataChef")
    st.caption(
        "Upload → Intent → Plan → Approval → Quality → Results. "
        "Nothing leaves this machine and no plan runs without your approval."
    )
    render_screen(session.screen, controller, state)


if __name__ == "__main__":
    run_app()
