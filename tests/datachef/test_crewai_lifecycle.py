from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from datachef.contracts import WorkflowState
from datachef.planning import RuleBasedPlanner, RuleBasedReviewer
from datachef.workflow import WorkflowRuntime


def test_deterministic_service_import_does_not_import_crewai() -> None:
    code = (
        "import sys; import datachef.workflow.service; "
        "print(any(name == 'crewai' or name.startswith('crewai.') for name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "DATACHEF_OFFLINE": "true",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.stdout.strip() == "False"


def test_safe_flow_boundary_isolates_storage_and_cleans_up_on_success_and_failure(
    raw_dataframe,
    user_intent,
    monkeypatch,
    tmp_path,
) -> None:
    import datachef.workflow.crewai_flow as flow_module

    original_local_app_data = str(tmp_path / "profile")
    monkeypatch.setenv("LOCALAPPDATA", original_local_app_data)
    initial = WorkflowRuntime(
        state=WorkflowState(),
        raw_dataframe=raw_dataframe.copy(deep=True),
        user_intent=user_intent,
    )

    with flow_module.isolated_crewai_runtime():
        from crewai.events.event_bus import crewai_event_bus

        flow = flow_module._create_phase1a_flow(
            initial,
            user_intent,
            RuleBasedPlanner(),
            RuleBasedReviewer(),
        )
        serialized_flow = flow.model_dump_json()
        assert "fictional.one@example.test" not in serialized_flow

    original_shutdown = crewai_event_bus.shutdown
    shutdown_calls = []

    def shutdown_spy(*, wait=True):
        shutdown_calls.append(wait)
        return original_shutdown(wait=wait)

    monkeypatch.setattr(crewai_event_bus, "shutdown", shutdown_spy)
    completed = flow_module.run_phase1a_flow(
        initial,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )
    assert completed.state.stage.value == "AWAITING_APPROVAL"

    class FailingFlow:
        def kickoff(self):
            raise RuntimeError("synthetic kickoff failure")

    monkeypatch.setattr(
        flow_module,
        "_create_phase1a_flow",
        lambda *args, **kwargs: FailingFlow(),
    )
    with pytest.raises(RuntimeError, match="synthetic kickoff failure"):
        flow_module.run_phase1a_flow(
            initial,
            user_intent,
            RuleBasedPlanner(),
            RuleBasedReviewer(),
        )

    assert shutdown_calls == [True, True]
    assert os.environ["LOCALAPPDATA"] == original_local_app_data
    assert not (Path(original_local_app_data) / "CrewAI").exists()


def test_cleanup_failure_is_primary_only_when_kickoff_succeeds(
    raw_dataframe,
    user_intent,
    monkeypatch,
) -> None:
    import datachef.workflow.crewai_flow as flow_module

    initial = WorkflowRuntime(
        state=WorkflowState(),
        raw_dataframe=raw_dataframe.copy(deep=True),
    )

    class SuccessfulFlow:
        workflow_runtime = initial

        def kickoff(self):
            return None

    with flow_module.isolated_crewai_runtime():
        from crewai.events.event_bus import crewai_event_bus

    monkeypatch.setattr(flow_module, "_create_phase1a_flow", lambda *a, **k: SuccessfulFlow())

    def failing_shutdown(*, wait=True):
        del wait
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(crewai_event_bus, "shutdown", failing_shutdown)
    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        flow_module.run_phase1a_flow(
            initial,
            user_intent,
            RuleBasedPlanner(),
            RuleBasedReviewer(),
        )


def test_cleanup_failure_does_not_mask_kickoff_failure(
    raw_dataframe,
    user_intent,
    monkeypatch,
) -> None:
    import datachef.workflow.crewai_flow as flow_module

    initial = WorkflowRuntime(
        state=WorkflowState(),
        raw_dataframe=raw_dataframe.copy(deep=True),
    )

    class FailingFlow:
        def kickoff(self):
            raise ValueError("primary kickoff failure")

    with flow_module.isolated_crewai_runtime():
        from crewai.events.event_bus import crewai_event_bus

    monkeypatch.setattr(flow_module, "_create_phase1a_flow", lambda *a, **k: FailingFlow())

    def failing_shutdown(*, wait=True):
        del wait
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(crewai_event_bus, "shutdown", failing_shutdown)
    with pytest.raises(ValueError, match="primary kickoff failure") as captured:
        flow_module.run_phase1a_flow(
            initial,
            user_intent,
            RuleBasedPlanner(),
            RuleBasedReviewer(),
        )

    assert any("RuntimeError" in note for note in captured.value.__notes__)
    assert all("secondary cleanup failure" not in note for note in captured.value.__notes__)
