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
from datachef.application.session import furthest_screen_for_workflow_stage
from ui import state as ui_state
from ui.screens import render_screen


LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "img",
    "Gemini_Generated_Image_adhg9madhg9madhg.png",
)

# Display labels only. This tuple is the presentation order of the seven
# screens, and every one of them is a place the user can actually be. Quality
# assurance is absent because it is not a screen: it remains the mandatory
# internal gate that decides whether Results has any gold to show.
_PROGRESS = (
    (ScreenId.UPLOAD, "1 · Upload"),
    (ScreenId.DIAGNOSE, "2 · Diagnostics"),
    (ScreenId.INTENT, "3 · Objective"),
    (ScreenId.PLAN, "4 · Plan"),
    (ScreenId.APPROVAL, "5 · Approve"),
    (ScreenId.RESULTS, "6 · Results"),
    (ScreenId.DASHBOARD, "7 · Dashboard"),
)

_ORDER = {screen: index for index, (screen, _) in enumerate(_PROGRESS)}


def _reached_position(session) -> int:
    """Furthest stage the workflow itself has reached.

    Derived from the workflow stage through the public
    ``furthest_screen_for_workflow_stage`` mapping, so the UI never invents its
    own notion of progress -- including whether the dashboard is earned, which
    only a passing quality gate decides. Combined with the controller's current
    screen so that navigating back does not hide the stages already earned.
    """

    positions = [_ORDER.get(session.screen, 0)]
    runtime = session.workflow_runtime
    if runtime is not None:
        positions.append(_ORDER[furthest_screen_for_workflow_stage(runtime.state.stage)])
    return max(positions)


def _render_stage_indicator(controller, session) -> None:
    current = _ORDER.get(session.screen, 0)
    reached = _reached_position(session)
    st.sidebar.markdown("### Progress")
    for screen, label in _PROGRESS:
        position = _ORDER[screen]
        if position == current:
            st.sidebar.markdown(f"**➡️ {label}**")
        elif position <= reached:
            # Revisiting an already-reached stage. Navigation authorizes nothing:
            # each screen re-derives what it may show from controller evidence.
            if st.sidebar.button(
                f"✅ {label}",
                key=f"{ui_state.STAGE_NAV_WIDGET}_{screen.value}",
                use_container_width=True,
            ):
                controller.navigate(screen)
                st.rerun()
        else:
            st.sidebar.markdown(f"◻️ {label}")


def _render_sidebar(controller, state) -> None:
    session = controller.session
    st.sidebar.title("DataChef")
    st.sidebar.caption("Offline, deterministic, human-approved data preparation.")

    _render_stage_indicator(controller, session)

    st.sidebar.markdown("---")
    registry = ui_state.agent_registry(state)
    if registry is not None and registry.live:
        st.sidebar.markdown("**Planner:** 🤖 AI planner")
        st.sidebar.caption("CrewAI crew, working inside the deterministic allow-list.")
    else:
        st.sidebar.markdown("**Planner:** ⚙️ deterministic")
        st.sidebar.caption("Rule-based planner. No provider is contacted.")

    st.sidebar.markdown("---")
    # The controller flag is the single source of truth. When it is changed from
    # another control (the upload screen), seed this toggle's widget state once
    # before it is instantiated so the two never fight over ownership.
    if state.get(ui_state.PREVIEW_SYNC) != session.preview_enabled:
        state[ui_state.PREVIEW_WIDGET] = session.preview_enabled
        state[ui_state.PREVIEW_SYNC] = session.preview_enabled
    preview = st.sidebar.toggle(
        "Show local data preview",
        key=ui_state.PREVIEW_WIDGET,
        help=(
            "Preview rows are presentation only. They never enter evidence, "
            "artifacts, the manifest, or any provider context."
        ),
    )
    if preview != session.preview_enabled:
        controller.set_preview_enabled(preview)
        state[ui_state.PREVIEW_SYNC] = preview

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


def _load_local_configuration() -> None:
    """Make .env visible to this process without overriding the real environment.

    Nothing else on the DataChef path loads it, so without this the credential
    never reaches os.environ and live mode could never activate. ``override=False``
    keeps any real environment variable authoritative, and .env ships
    DATACHEF_OFFLINE=true, so the offline default is preserved. No value from the
    file is ever printed, logged, or rendered.
    """

    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
    except Exception:
        # A missing or unreadable .env simply leaves the environment as it was.
        pass


def run_app() -> None:
    st.set_page_config(page_title="DataChef", layout="wide")

    _load_local_configuration()

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
        "Upload → Diagnostics → Objective → Plan → Approve → Results "
        "→ Dashboard. Nothing leaves this machine and no plan runs without "
        "your approval."
    )
    render_screen(session.screen, controller, state)


if __name__ == "__main__":
    run_app()
