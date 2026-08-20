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
from ui.styles import apply_global_styles
from ui.styles import upload_styles

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
from ui.charts import render_charts
from ui.ingestion_view import render_ingestion
from ui.styles import _apply_custom_styles




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

"""
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
"""

def _init_state():
    defaults = {
        "stage": 0,
        "uploaded_file": None,
        "raw_df": None,
        "gold_df": None,
        "generated_code": None,
        "dashboard_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _demo_dataframe():
    return pd.DataFrame(
        {
            "order_id": [101, 102, 103, 103, 104],
            "customer_id": ["C-001", None, "C-003", "C-003", "C-005"],
            "region": ["North", "North", "South", "South", "West"],
            "order_date": [
                "2026-01-03",
                "2026-01-05",
                "bad-date",
                "2026-01-08",
                "2026-01-10",
            ],
            "amount": ["$120.5", "$80.0", "$330.0", "$330.0", "$95.0"],
            "status": [
                "completed",
                "completed",
                "pending",
                "pending",
                "completed",
            ],
        }
    )



def _render_stage_nav():

    stages = [
        "1. Upload",
        "2. Diagnose",
        "3. Transform",
        "4. Dashboard",
    ]

    st.sidebar.title("Workflow")

    current_stage = st.session_state["stage"]

    for index, label in enumerate(stages):

        button_type = (
            "primary"
            if index == current_stage
            else "secondary"
        )

        if st.sidebar.button(
            label,
            key=f"nav_{index}",
            use_container_width=True,
            type=button_type,
        ):

            st.session_state["stage"] = index

            st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "DataChef Agentic Platform"
    )


def _render_upload_stage():

    st.header("Stage 1: Ingest and diagnose")
    st.subheader("Upload a raw dataset")

    # =====================================================
    # UPLOAD AREA
    # =====================================================

    uploaded_file = st.file_uploader(
        "Choose a CSV or JSON file",
        type=["csv", "json"],
        label_visibility="collapsed",
    )

    st.markdown(
        "<div style='height: 100px;'></div>",
        unsafe_allow_html=True,
    )

    # =====================================================
    # DEMO DATASET BUTTON
    # =====================================================

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button(
            "Load demo dataset",
            use_container_width=True,
        ):
            st.session_state["raw_df"] = _demo_dataframe()
            st.session_state["uploaded_file"] = "demo_orders.csv"
            st.success("Demo dataset loaded.")

    # =====================================================
    # READ FILE
    # =====================================================

    if uploaded_file is not None:

        st.session_state["uploaded_file"] = uploaded_file.name

        if uploaded_file.name.endswith(".csv"):
            st.session_state["raw_df"] = pd.read_csv(uploaded_file)

        else:
            st.sidebar.markdown(f"◻️ {label}")

def _render_stage_indicator(controller, session) -> None:
    current = _ORDER.get(session.screen, 0)
    reached = _reached_position(session)

    st.sidebar.markdown(
        '<div class="progress-title">Progress</div>',
        unsafe_allow_html=True,
    )
    # =====================================================
    # DATASET LOADED
    # =====================================================

    if st.session_state["raw_df"] is not None:

        df = st.session_state["raw_df"]

        st.markdown(
            "<div style='height: 20px;'></div>",
            unsafe_allow_html=True,
        )

        st.success(
            f"Loaded: {st.session_state['uploaded_file']}"
        )

        # =================================================
        # DATA PREVIEW
        # =================================================

        st.markdown("### Data preview")

        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=300,
        )

        st.markdown(
            "<div style='height: 15px;'></div>",
            unsafe_allow_html=True,
        )

        # =================================================
        # OVERVIEW
        # =================================================

        st.markdown("### Dataset overview")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric(
                "Rows",
                len(df),
            )

        with col_b:
            st.metric(
                "Columns",
                df.shape[1],
            )

        with col_c:
            st.metric(
                "Data types",
                df.dtypes.nunique(),
            )

        st.markdown(
            "<div style='height: 18px;'></div>",
            unsafe_allow_html=True,
        )

        # =================================================
        # ANALYZE BUTTON
        # =================================================

        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            if st.button(
                "Analyze data →",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["stage"] = 1
                st.rerun()


def _render_diagnose_stage():
    st.header("Stage 2: Ingestion & diagnosis")
    st.markdown(
        "The ingestion agent profiles the raw data, scores its health, and "
        "suggests SQL, indexes and quality alerts. Ask it anything below."
    )
    if st.session_state["raw_df"] is None:
        st.warning("Load a dataset first in the upload stage.")
        return

    df = _to_gold(st.session_state["raw_df"])
    table_name = st.session_state.get("uploaded_file") or "ingested_data"
    render_ingestion(df, table_name=table_name)

    st.markdown("---")
    if st.button("Accept diagnosis and continue", type="primary"):
        st.session_state["stage"] = 2
        st.rerun()


def _render_transform_stage():
    st.header("Stage 3: Transformation and processing")
    st.markdown(
        "The **Action Agent** executes transformations directly in a **Local Sandbox** "
        "with automatic **Self-Healing** capabilities upon encountering syntax errors."
    )

    for screen, label in _PROGRESS:
        position = _ORDER[screen]

        # Current stage
        if position == current:
            st.sidebar.markdown(
                f"""
                <div class="stage-current">
                    <span>●</span>
                    <span>{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Completed / already reached stage
        elif position <= reached:
            if st.sidebar.button(
                f"✓  {label}",
                key=f"{ui_state.STAGE_NAV_WIDGET}_{screen.value}",
                use_container_width=True,
            ):
                controller.navigate(screen)
                st.rerun()

        # Future stage
        else:
            st.sidebar.markdown(
                f"""
                <div class="stage-pending">
                    <span>○</span>
                    <span>{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )            



def _render_sidebar(controller, state) -> None:
    session = controller.session

    if os.path.exists(LOGO_PATH):
        st.sidebar.image(
            LOGO_PATH,
            width="stretch",
        )
    with st.sidebar:
        st.image(LOGO_PATH, width=200)

    _init_state()
    _render_stage_nav()
    _apply_custom_styles()

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image(LOGO_PATH, width="stretch")
    st.markdown(
    """
        <p style="
            text-align: center;
            font-family: 'Oxanium', sans-serif;
            font-size: 25px;
            font-weight: 500;
            color: #B9B7D0;
            letter-spacing: 1px;
            margin-top: -12px;
            margin-bottom: 8px;
        ">
            Agentic workflow for ingestion, transformation, and dashboarding
        </p>
        """,
        unsafe_allow_html=True,
    )


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

    st.set_page_config(
        page_title="DataChef",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_global_styles()
    upload_styles()
    _load_local_configuration()

    if os.path.exists(LOGO_PATH):
        left, center, right = st.columns([1, 2, 1])

        with center:
            st.image(
                LOGO_PATH,
                width="stretch",
            )
        st.markdown(
            """
            <div class="main-subtitle">
                Agentic workflow for ingestion, transformation, and dashboarding
            </div>
            """,
            unsafe_allow_html=True,
        )

    state = st.session_state
    # Reset runs before any widget is instantiated so widget keys can be cleared.
    ui_state.apply_pending_reset(state)

    controller = ui_state.get_controller(state)

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
