from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

import datachef.application.dashboard as dashboard_module

from datachef.application import (
    CsvParserOptions,
    DashboardContext,
    DashboardFailure,
    DashboardHandoff,
    DataChefController,
    UploadFormat,
    UploadRequest,
    build_dashboard_handoff,
)
from datachef.contracts import (
    DownstreamUse,
    HumanDecision,
    PIIHandling,
    QAStatus,
    UserIntent,
    WorkflowStage,
)
from datachef.workflow import WorkflowRuntime


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
REPO_ROOT = Path(__file__).parents[2]
AUTHORED_QUESTION = "Which region leads on amount?"
CSV = (
    b"order_id,region,amount,ordered_on\n"
    b"1,North,10,2026-01-01\n"
    b"2,South,20,2026-01-02\n"
    b"2,South,20,2026-01-02\n"
    b"3,North,30,2026-01-03\n"
)


def _intent(**overrides) -> UserIntent:
    payload = {
        "intent_id": "intent-dashboard",
        "user_goal": "Prepare the table for analysis.",
        "downstream_use": DownstreamUse.ANALYSIS,
        "selected_key_columns": ("order_id",),
        "required_columns": ("order_id",),
        "acceptable_row_loss_pct": 50,
        "pii_handling": PIIHandling.NONE,
        "questions": (AUTHORED_QUESTION,),
    }
    payload.update(overrides)
    return UserIntent(**payload)


def _controller(content: bytes = CSV) -> DataChefController:
    controller = DataChefController(clock=lambda: NOW)
    assert controller.load_upload(
        UploadRequest(
            content=content,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    ).changed
    assert controller.diagnose().changed
    return controller


def _gold_controller() -> DataChefController:
    controller = _controller()
    controller.submit_intent(_intent(), ())
    assert controller.prepare_plan(command_id="plan").code == "PLAN_AWAITING_APPROVAL"
    assert controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="approve",
    ).changed
    assert controller.execute_current_plan(command_id="execute").changed
    runtime = controller.session.workflow_runtime
    assert runtime is not None and runtime.state.stage is WorkflowStage.QA_PASSED
    return controller


def _gold_runtime() -> WorkflowRuntime:
    runtime = _gold_controller().session.workflow_runtime
    assert runtime is not None
    return runtime


def _legacy_spec(gold: pd.DataFrame) -> dict:
    from crew.dashboard_agent.dashboard_agent import build_dashboard_spec

    return build_dashboard_spec(gold.copy(deep=True), use_llm=False)


def _patched_spec(monkeypatch, mutate) -> DashboardHandoff | DashboardFailure:
    import crew.dashboard_agent.dashboard_agent as legacy

    real = legacy.build_dashboard_spec

    def fake(frame, use_llm=True):
        return mutate(copy.deepcopy(real(frame, use_llm=use_llm)))

    monkeypatch.setattr(legacy, "build_dashboard_spec", fake)
    return _gold_controller().build_dashboard_handoff()


def test_pass_run_produces_a_handoff_from_the_legacy_rule_based_builder() -> None:
    controller = _gold_controller()
    state = controller.session.workflow_runtime.state

    handoff = controller.build_dashboard_handoff()

    assert isinstance(handoff, DashboardHandoff)
    context = handoff.context
    assert isinstance(context, DashboardContext)
    assert context.dataset_id == state.dataset_identity.dataset_id
    assert context.result_fingerprint == state.execution_result.result_fingerprint
    assert context.plan_id == state.transformation_plan.plan_id
    assert context.qa_report_id == state.qa_report.qa_report_id
    assert context.downstream_use is DownstreamUse.ANALYSIS
    assert context.authored_questions == (AUTHORED_QUESTION,)
    spec = handoff.dashboard_spec()
    assert spec["engine"] == "rule-based"
    assert spec["meta"] == {"rows": 3, "columns": 4}


def test_handoff_reuses_the_untouched_legacy_specification() -> None:
    controller = _gold_controller()
    gold = controller.session.workflow_runtime.gold_dataframe

    handoff = controller.build_dashboard_handoff()

    assert isinstance(handoff, DashboardHandoff)
    expected = _legacy_spec(gold)
    actual = handoff.dashboard_spec()
    assert actual["roles"] == expected["roles"]
    assert actual["kpis"] == expected["kpis"]
    assert actual["charts"] == expected["charts"]
    assert actual["insights"] == expected["insights"]
    assert actual["meta"] == expected["meta"]


def test_generic_legacy_title_is_replaced_only_in_the_copy() -> None:
    controller = _gold_controller()
    gold = controller.session.workflow_runtime.gold_dataframe

    handoff = controller.build_dashboard_handoff()

    assert isinstance(handoff, DashboardHandoff)
    assert _legacy_spec(gold)["title"] == dashboard_module.LEGACY_TITLE
    assert handoff.dashboard_spec()["title"] == "DataChef Dashboard"


def test_legacy_builder_receives_use_llm_false_and_a_private_copy(monkeypatch) -> None:
    import crew.dashboard_agent.dashboard_agent as legacy

    real = legacy.build_dashboard_spec
    calls: list[dict] = []

    def recording(frame, use_llm=True, *args, **kwargs):
        calls.append({"use_llm": use_llm, "frame": frame})
        return real(frame, use_llm=use_llm, *args, **kwargs)

    monkeypatch.setattr(legacy, "build_dashboard_spec", recording)
    controller = _gold_controller()
    gold = controller.session.workflow_runtime.gold_dataframe

    handoff = controller.build_dashboard_handoff()

    assert isinstance(handoff, DashboardHandoff)
    assert len(calls) == 1
    assert calls[0]["use_llm"] is False
    passed = calls[0]["frame"]
    assert passed is not gold
    passed.loc[passed.index[0], "amount"] = 999_999
    assert 999_999 not in set(
        controller.session.workflow_runtime.gold_dataframe["amount"]
    )


def test_handoff_owns_defensive_copies_of_gold_and_spec() -> None:
    controller = _gold_controller()
    handoff = controller.build_dashboard_handoff()
    assert isinstance(handoff, DashboardHandoff)
    before = controller.session.workflow_runtime.gold_dataframe["amount"].tolist()

    frame = handoff.gold_frame()
    frame.loc[frame.index[0], "amount"] = 999_999
    spec = handoff.dashboard_spec()
    spec["charts"].clear()
    spec["title"] = "tampered"

    assert handoff.gold_frame()["amount"].tolist() == before
    assert controller.session.workflow_runtime.gold_dataframe["amount"].tolist() == before
    assert handoff.dashboard_spec()["charts"]
    assert handoff.dashboard_spec()["title"] == "DataChef Dashboard"
    assert handoff.gold_frame() is not handoff.gold_frame()


def test_source_and_authoritative_gold_survive_the_handoff() -> None:
    controller = _gold_controller()
    source_before = controller.session.source.raw_copy()
    gold_before = controller.session.workflow_runtime.gold_dataframe.copy(deep=True)

    controller.build_dashboard_handoff()

    pd.testing.assert_frame_equal(controller.session.source.raw_copy(), source_before)
    pd.testing.assert_frame_equal(
        controller.session.workflow_runtime.gold_dataframe,
        gold_before,
    )


def test_handoff_is_runtime_only_and_hides_backing_storage() -> None:
    handoff = _gold_controller().build_dashboard_handoff()
    assert isinstance(handoff, DashboardHandoff)

    with pytest.raises(AttributeError):
        handoff.context = None
    with pytest.raises(AttributeError):
        getattr(handoff, "_DashboardHandoff__gold")
    with pytest.raises(TypeError):
        pickle.dumps(handoff)
    assert "gold=<private>" in repr(handoff)


def test_handoff_identity_tracks_the_verified_result() -> None:
    first = _gold_controller().build_dashboard_handoff()
    repeated = _gold_controller().build_dashboard_handoff()
    other = _controller(
        b"order_id,region,amount,ordered_on\n"
        b"7,West,70,2026-02-01\n"
        b"8,East,80,2026-02-02\n"
        b"8,East,80,2026-02-02\n"
    )
    other.submit_intent(_intent(), ())
    other.prepare_plan(command_id="plan")
    other.record_human_decision(HumanDecision.APPROVE, command_id="approve")
    other.execute_current_plan(command_id="execute")
    second = other.build_dashboard_handoff()

    assert isinstance(first, DashboardHandoff) and isinstance(second, DashboardHandoff)
    assert isinstance(repeated, DashboardHandoff)
    assert first.context.handoff_id == repeated.context.handoff_id
    assert first.context.handoff_id != second.context.handoff_id
    assert first.context.result_fingerprint != second.context.result_fingerprint


def test_controller_forwards_only_the_selected_suggested_questions() -> None:
    controller = _controller()
    suggested = controller.session.suggested_questions
    assert suggested
    controller.submit_intent(
        _intent(),
        (),
        selected_question_ids=(suggested[0].question_id,),
    )
    controller.prepare_plan(command_id="plan")
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approve")
    controller.execute_current_plan(command_id="execute")

    handoff = controller.build_dashboard_handoff()

    assert isinstance(handoff, DashboardHandoff)
    assert tuple(
        item.question_id for item in handoff.context.selected_questions
    ) == (suggested[0].question_id,)


def test_building_the_handoff_does_not_mutate_the_session() -> None:
    controller = _gold_controller()
    before = controller.session

    controller.build_dashboard_handoff()
    after = controller.session

    assert after.revision == before.revision
    assert after.command_history == before.command_history
    assert after.screen is before.screen
    assert after.workflow_runtime.state == before.workflow_runtime.state


def test_raw_only_session_produces_no_handoff() -> None:
    controller = _controller()

    failure = controller.build_dashboard_handoff()

    assert isinstance(failure, DashboardFailure)
    assert failure.code.value == "GOLD_UNAVAILABLE"


def test_rejected_plan_produces_no_handoff() -> None:
    controller = _controller()
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan")
    controller.record_human_decision(HumanDecision.REJECT, command_id="reject")
    controller.execute_current_plan(command_id="execute")

    failure = controller.build_dashboard_handoff()

    assert isinstance(failure, DashboardFailure)
    assert failure.code.value == "GOLD_UNAVAILABLE"


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    (
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={"stage": WorkflowStage.QA_WARNING}
                ),
            ),
            "GOLD_UNAVAILABLE",
            id="warn-stage",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={"stage": WorkflowStage.QA_FAILED}
                ),
            ),
            "GOLD_UNAVAILABLE",
            id="fail-stage",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={"stage": WorkflowStage.EXECUTION_FAILED}
                ),
            ),
            "GOLD_UNAVAILABLE",
            id="execution-failed-stage",
        ),
        pytest.param(
            lambda runtime: replace(runtime, gold_dataframe=None),
            "GOLD_UNAVAILABLE",
            id="missing-gold",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={
                        "qa_report": runtime.state.qa_report.model_copy(
                            update={"status": QAStatus.FAIL}
                        )
                    }
                ),
            ),
            "GOLD_EVIDENCE_MISMATCH",
            id="forged-qa-status",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={
                        "execution_result": runtime.state.execution_result.model_copy(
                            update={"plan_id": "foreign-plan"}
                        )
                    }
                ),
            ),
            "GOLD_EVIDENCE_MISMATCH",
            id="foreign-plan",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={
                        "qa_report": runtime.state.qa_report.model_copy(
                            update={"dataset_id": "foreign-dataset"}
                        )
                    }
                ),
            ),
            "GOLD_EVIDENCE_MISMATCH",
            id="foreign-qa",
        ),
    ),
)
def test_untrustworthy_runtimes_produce_no_handoff(mutate, expected_code: str) -> None:
    tampered = mutate(_gold_runtime())

    failure = build_dashboard_handoff(tampered, _intent())

    assert isinstance(failure, DashboardFailure)
    assert failure.code.value == expected_code


def test_stale_fingerprint_after_modified_gold_produces_no_handoff() -> None:
    runtime = _gold_runtime()
    forged = runtime.gold_dataframe.copy(deep=True)
    forged.loc[forged.index[0], "amount"] = 999_999

    failure = build_dashboard_handoff(replace(runtime, gold_dataframe=forged), _intent())

    assert isinstance(failure, DashboardFailure)
    assert failure.code.value == "GOLD_EVIDENCE_MISMATCH"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        pytest.param(
            lambda spec: {**spec, "engine": "gemini"}, "invalid-engine", id="invalid-engine"
        ),
        pytest.param(
            lambda spec: {**spec, "error": "provider unavailable"},
            "error-field",
            id="error-field",
        ),
        pytest.param(
            lambda spec: {**spec, "meta": {"rows": 999, "columns": 4}},
            "wrong-shape",
            id="wrong-row-count",
        ),
        pytest.param(
            lambda spec: {**spec, "meta": {"rows": 3, "columns": 99}},
            "wrong-shape",
            id="wrong-column-count",
        ),
        pytest.param(
            lambda spec: {
                **spec,
                "charts": [{**spec["charts"][0], "x": "not_a_column"}],
            },
            "missing-chart-column",
            id="missing-chart-column",
        ),
        pytest.param(
            lambda spec: {
                **spec,
                "charts": [{**spec["charts"][0], "type": "sankey"}],
            },
            "unsupported-chart-type",
            id="unsupported-chart-type",
        ),
        pytest.param(
            lambda spec: {
                **spec,
                "charts": [{**spec["charts"][0], "agg": "median"}],
            },
            "unsupported-aggregation",
            id="unsupported-aggregation",
        ),
        pytest.param(
            lambda spec: {**spec, "roles": {**spec["roles"], "measures": ["ghost"]}},
            "missing-role-column",
            id="missing-role-column",
        ),
        pytest.param(
            lambda spec: {**spec, "kpis": "not-a-list"},
            "wrong-structure",
            id="malformed-kpis",
        ),
        pytest.param(lambda spec: {"engine": "rule-based"}, "missing-keys", id="missing-keys"),
        pytest.param(lambda spec: ["not", "a", "dict"], "not-a-dict", id="not-a-dict"),
    ),
)
def test_unsupported_or_malformed_specifications_are_refused(
    monkeypatch,
    mutate,
    reason: str,
) -> None:
    outcome = _patched_spec(monkeypatch, mutate)

    assert isinstance(outcome, DashboardFailure), reason
    assert outcome.code.value == "DASHBOARD_SPEC_INVALID"
    assert outcome.safe_message and outcome.suggested_action


def test_legacy_builder_exception_is_sanitized(monkeypatch) -> None:
    import crew.dashboard_agent.dashboard_agent as legacy

    def failing(frame, use_llm=True):
        raise RuntimeError("private legacy detail")

    monkeypatch.setattr(legacy, "build_dashboard_spec", failing)

    outcome = _gold_controller().build_dashboard_handoff()

    assert isinstance(outcome, DashboardFailure)
    assert outcome.code.value == "DASHBOARD_BUILDER_FAILURE"
    assert "private legacy detail" not in repr(outcome)


def test_application_import_graph_never_loads_the_legacy_dashboard_or_providers() -> None:
    script = textwrap.dedent(
        """
        import sys
        import datachef.application
        import datachef.application.dashboard

        loaded = set(sys.modules)
        forbidden = {
            name
            for name in loaded
            if name.split(".")[0] in {"crewai", "langchain_google_genai", "streamlit"}
            or name.startswith("google.genai")
            or name.startswith("crew.dashboard_agent")
            or name.startswith("crew.transformation_agent")
        }
        assert not forbidden, sorted(forbidden)
        print("OK")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_credentialed_offline_handoff_touches_no_provider_or_network() -> None:
    """Adversarial: a fake key is present, every egress path is blocked."""

    script = textwrap.dedent(
        """
        import os, socket, sys

        os.environ["GOOGLE_API_KEY"] = "fake-key-must-never-be-used"
        os.environ["GEMINI_API_KEY"] = "fake-key-must-never-be-used"

        attempts = []

        def blocked(*args, **kwargs):
            attempts.append("connect")
            raise AssertionError("network access is blocked in this test")

        socket.create_connection = blocked
        socket.socket.connect = blocked

        import httpx
        httpx.Client.send = blocked
        httpx.AsyncClient.send = blocked
        import requests
        requests.Session.send = blocked
        requests.adapters.HTTPAdapter.send = blocked

        class Guard:
            blocked_prefixes = (
                "crewai",
                "google.genai",
                "langchain_google_genai",
                "crew.transformation_agent",
            )

            def find_spec(self, fullname, path=None, target=None):
                if any(
                    fullname == name or fullname.startswith(name + ".")
                    for name in self.blocked_prefixes
                ):
                    raise AssertionError("forbidden module import: " + fullname)
                return None

        sys.meta_path.insert(0, Guard())

        import datachef.application.dashboard as dashboard
        assert "crew.dashboard_agent.dashboard_agent" not in sys.modules

        import crew.dashboard_agent.dashboard_agent as legacy
        real = legacy.build_dashboard_spec
        observed = []

        def recording(frame, use_llm=True):
            observed.append(use_llm)
            return real(frame, use_llm=use_llm)

        legacy.build_dashboard_spec = recording

        from datetime import datetime, timezone
        from datachef.application import (
            CsvParserOptions, DataChefController, UploadFormat, UploadRequest,
        )
        from datachef.contracts import (
            DownstreamUse, HumanDecision, PIIHandling, UserIntent,
        )
        import pandas as pd

        NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
        CSV = (
            b"order_id,region,amount,ordered_on\\n"
            b"1,North,10,2026-01-01\\n"
            b"2,South,20,2026-01-02\\n"
            b"2,South,20,2026-01-02\\n"
            b"3,North,30,2026-01-03\\n"
        )
        controller = DataChefController(clock=lambda: NOW)
        controller.load_upload(
            UploadRequest(
                content=CSV,
                declared_suffix=".csv",
                format=UploadFormat.CSV,
                parser_options=CsvParserOptions(encoding="utf-8-sig"),
            )
        )
        controller.diagnose()
        controller.submit_intent(
            UserIntent(
                intent_id="adversarial",
                user_goal="Prepare the table for analysis.",
                downstream_use=DownstreamUse.ANALYSIS,
                selected_key_columns=("order_id",),
                required_columns=("order_id",),
                acceptable_row_loss_pct=50,
                pii_handling=PIIHandling.NONE,
            ),
            (),
        )
        controller.prepare_plan(command_id="plan")
        controller.record_human_decision(HumanDecision.APPROVE, command_id="approve")
        controller.execute_current_plan(command_id="execute")

        source_before = controller.session.source.raw_copy()
        gold_before = controller.session.workflow_runtime.gold_dataframe.copy(deep=True)

        handoff = controller.build_dashboard_handoff()

        assert isinstance(handoff, dashboard.DashboardHandoff), handoff
        assert observed == [False], observed
        assert attempts == [], attempts
        assert "crewai" not in sys.modules
        assert not any(name.startswith("google.genai") for name in sys.modules)
        assert "langchain_google_genai" not in sys.modules
        assert handoff.dashboard_spec()["engine"] == "rule-based"
        assert "fake-key-must-never-be-used" not in repr(handoff.context)

        pd.testing.assert_frame_equal(controller.session.source.raw_copy(), source_before)
        pd.testing.assert_frame_equal(
            controller.session.workflow_runtime.gold_dataframe, gold_before
        )
        frame = handoff.gold_frame()
        frame.loc[frame.index[0], "amount"] = 999_999
        assert handoff.gold_frame()["amount"].tolist() == gold_before["amount"].tolist()
        print("OK")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
