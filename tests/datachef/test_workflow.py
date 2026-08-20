from __future__ import annotations

from datetime import datetime, timezone

from datachef.contracts import (
    HumanApproval,
    HumanDecision,
    InvariantKind,
    QualityInvariant,
    ReviewerDecision,
    WorkflowStage,
    WorkflowState,
)
from datachef.planning import RuleBasedPlanner, RuleBasedReviewer, SequenceReviewer
from datachef.workflow import WorkflowRuntime, execute_workflow, prepare_workflow


def _approval(runtime: WorkflowRuntime) -> HumanApproval:
    plan = runtime.state.transformation_plan
    assert plan is not None
    return HumanApproval(
        dataset_id=plan.dataset_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        decision=HumanDecision.APPROVE,
        approved_operation_ids=tuple(
            operation.operation_id for operation in plan.operations
        ),
        decided_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )


def test_workflow_pauses_for_matching_human_approval(raw_dataframe, user_intent) -> None:
    runtime = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )

    paused = execute_workflow(runtime, None)

    assert runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert paused.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert paused is runtime
    assert paused.transformed_dataframe is None


def test_reviewer_rejection_ends_workflow(raw_dataframe, user_intent) -> None:
    runtime = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        SequenceReviewer((ReviewerDecision.REJECT,)),
    )

    assert runtime.state.stage is WorkflowStage.PLAN_REJECTED
    assert runtime.state.last_error_code == "PLAN_REJECTED_BY_REVIEWER"


def test_qa_failure_blocks_gold_promotion(raw_dataframe, user_intent) -> None:
    runtime = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )
    impossible = QualityInvariant(
        invariant_id="must-have-nonexistent-column",
        kind=InvariantKind.REQUIRED_COLUMN,
        column="not_present",
        mandatory=True,
    )

    completed = execute_workflow(
        runtime,
        _approval(runtime),
        user_invariants=(impossible,),
    )

    assert completed.state.stage is WorkflowStage.QA_FAILED
    assert completed.transformed_dataframe is not None
    assert completed.gold_dataframe is None


def test_complete_offline_happy_path_reaches_qa_passed(raw_dataframe, user_intent) -> None:
    runtime = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )

    completed = execute_workflow(runtime, _approval(runtime))

    assert completed.state.stage is WorkflowStage.QA_PASSED
    assert completed.state.qa_report is not None
    assert completed.gold_dataframe is not None


def test_workflow_state_serializes_and_reconstructs(raw_dataframe, user_intent) -> None:
    runtime = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )

    restored = WorkflowState.model_validate_json(runtime.state.model_dump_json())

    assert restored == runtime.state
    assert restored.stage is WorkflowStage.AWAITING_APPROVAL


def test_workflow_state_serialization_uses_sanitized_intent(
    raw_dataframe,
    user_intent,
) -> None:
    sensitive = "fictional.one@example.test"
    intent = user_intent.model_copy(
        update={"user_goal": f"Analyze account {sensitive}"}
    )
    runtime = prepare_workflow(
        raw_dataframe,
        intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )

    serialized = runtime.state.model_dump_json()

    assert sensitive not in serialized
    assert "Analyze account" not in serialized


def test_real_crewai_flow_routes_pause_and_approved_execution(
    raw_dataframe,
    user_intent,
) -> None:
    from datachef.workflow.crewai_flow import (
        run_phase1a_flow,
    )

    initial = WorkflowRuntime(
        state=WorkflowState(),
        raw_dataframe=raw_dataframe.copy(deep=True),
    )
    planner = RuleBasedPlanner()
    reviewer = RuleBasedReviewer()
    planned_runtime = run_phase1a_flow(initial, user_intent, planner, reviewer)
    assert planned_runtime.state.stage is WorkflowStage.AWAITING_APPROVAL

    waiting_runtime = run_phase1a_flow(
        planned_runtime,
        user_intent,
        planner,
        reviewer,
    )
    assert waiting_runtime is planned_runtime

    completed_runtime = run_phase1a_flow(
        planned_runtime,
        user_intent,
        planner,
        reviewer,
        approval=_approval(planned_runtime),
    )
    assert completed_runtime.state.stage is WorkflowStage.QA_PASSED
    assert completed_runtime.gold_dataframe is not None


def test_real_crewai_flow_routes_revision_and_rejection(
    raw_dataframe,
    user_intent,
) -> None:
    from datachef.workflow.crewai_flow import (
        run_phase1a_flow,
    )

    initial = WorkflowRuntime(
        state=WorkflowState(),
        raw_dataframe=raw_dataframe.copy(deep=True),
    )
    revised_planner = RuleBasedPlanner()
    revised_reviewer = SequenceReviewer(
        (ReviewerDecision.REVISE, ReviewerDecision.ACCEPT)
    )
    rejected_reviewer = SequenceReviewer((ReviewerDecision.REJECT,))
    revised_runtime = run_phase1a_flow(
        initial,
        user_intent,
        revised_planner,
        revised_reviewer,
    )
    assert revised_runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert revised_planner.calls == 2
    assert revised_reviewer.calls == 2

    rejected_runtime = run_phase1a_flow(
        initial,
        user_intent,
        RuleBasedPlanner(),
        rejected_reviewer,
    )
    assert rejected_runtime.state.stage is WorkflowStage.PLAN_REJECTED
    assert rejected_reviewer.calls == 1


def test_real_crewai_flow_routes_qa_failure(raw_dataframe, user_intent) -> None:
    from datachef.workflow.crewai_flow import (
        run_phase1a_flow,
    )

    runtime = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )
    impossible = QualityInvariant(
        invariant_id="flow-must-have-missing-column",
        kind=InvariantKind.REQUIRED_COLUMN,
        column="not_present",
        mandatory=True,
    )
    completed = run_phase1a_flow(
        runtime,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
        approval=_approval(runtime),
        user_invariants=(impossible,),
    )

    assert completed.state.stage is WorkflowStage.QA_FAILED
    assert completed.gold_dataframe is None


def test_new_core_contains_no_generated_source_execution() -> None:
    from pathlib import Path

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("datachef").rglob("*.py")
    )

    assert "eval(" not in source
    assert "exec(" not in source
    assert "subprocess" not in source
