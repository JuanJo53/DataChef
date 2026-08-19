"""Compiled user requests as authoritative constraints on the canonical plan.

The planner may assist; it may not overrule the user. A request compiled locally
from the objective is a deterministic instruction, so after any planner has
proposed, the requests are enforced against that proposal:

* a request the plan omits is added;
* a request the plan contradicts -- same operation on the same columns but with
  different parameters -- replaces the contradicting operation, because the
  parameters are the instruction ("impute price with the *median*"), not a
  detail the planner may reinterpret;
* an operation targeting a column the user asked to drop is removed, since the
  user asked for the column to go, not to be prepared first;
* requested operations are emitted in the order the objective stated them, so
  "finally drop the duplicates" really is last and the imputations before it are
  measured on the pre-deduplication data;
* an operation already satisfying a request is reused but repositioned into that
  order, so the deterministic and live planners cannot disagree about sequence.

Two policies are preserved exactly as they were, because committed behaviour
depends on them:

* ``CAST_COLUMN`` requests are still answered only by the diagnosis-driven
  ``CANDIDATE_TYPE_CONVERSION`` route. A cast the diagnosis does not support
  stays unplanned and surfaces as a blocking ``REQUEST_NOT_PLANNED``.
* A ``DEDUPLICATE_BY_KEYS`` request is enforced only when the row-loss estimator
  can price it. Planning an unpriced deduplication would show the human "0 rows
  removed" for an operation that could empty the table, so it stays unplanned
  and is reported instead.

Enforcement is idempotent: applying it to a plan that already satisfies the
requests returns an equivalent plan, so a request-aware planner and this layer
may both run without duplicating work.
"""

from __future__ import annotations

from datachef.contracts import (
    OperationType,
    PlanningContext,
    RequestedOperation,
    RiskLevel,
    TransformationOperation,
    TransformationPlan,
)
from datachef.planning.plan import create_transformation_plan

# Only the request types with no diagnosis-driven path of their own.
ENFORCEABLE_REQUEST_TYPES = frozenset(
    {
        OperationType.DROP_COLUMN,
        OperationType.IMPUTE_MISSING,
        OperationType.DEDUPLICATE_BY_KEYS,
    }
)


def _signature(
    operation_type: OperationType,
    target_columns: tuple[str, ...],
) -> tuple[OperationType, tuple[str, ...]]:
    return (operation_type, tuple(target_columns))


def enforceable_requests(
    requested_operations: tuple[RequestedOperation, ...],
    context: PlanningContext,
) -> tuple[RequestedOperation, ...]:
    """The subset this layer is allowed to force into a plan."""

    priced_key_sets = {
        tuple(metric.key_columns)
        for metric in context.diagnostic_report.key_duplicate_metrics
    }
    allowed: list[RequestedOperation] = []
    seen: set[tuple[OperationType, tuple[str, ...]]] = set()
    for request in requested_operations:
        if request.operation_type not in ENFORCEABLE_REQUEST_TYPES:
            continue
        if (
            request.operation_type is OperationType.DEDUPLICATE_BY_KEYS
            and tuple(request.target_columns) not in priced_key_sets
        ):
            continue
        signature = _signature(request.operation_type, request.target_columns)
        if signature in seen:
            continue
        seen.add(signature)
        allowed.append(request)
    return tuple(allowed)


def _as_operation(
    request: RequestedOperation,
    taken_ids: set[str],
) -> TransformationOperation:
    suffix = "-".join(request.target_columns)
    base = f"op-{request.operation_type.value.lower()}-{suffix}"
    operation_id = base
    counter = 2
    while operation_id in taken_ids:
        operation_id = f"{base}-{counter}"
        counter += 1
    taken_ids.add(operation_id)
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=request.operation_type,
        target_columns=request.target_columns,
        parameters=request.parameters,
        # Grounded in the user's own request rather than a fabricated issue link.
        user_requirement_ids=(request.request_id,),
        rationale="The objective explicitly requested this operation.",
        expected_effect="Apply the operation the user asked for.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )


def enforce_requested_operations(
    plan: TransformationPlan,
    requested_operations: tuple[RequestedOperation, ...],
    context: PlanningContext,
) -> TransformationPlan:
    """Return a plan that honours every enforceable request, order-safe."""

    allowed = enforceable_requests(requested_operations, context)
    if not allowed:
        return plan

    by_signature = {
        _signature(request.operation_type, request.target_columns): request
        for request in allowed
    }
    dropped_columns = {
        column
        for request in allowed
        if request.operation_type is OperationType.DROP_COLUMN
        for column in request.target_columns
    }

    # Planner operations that already satisfy a request are reused but not left
    # where the planner put them: the compiled request sequence is the user's
    # stated order, so the satisfying operation is repositioned into it.
    satisfying: dict[tuple[OperationType, tuple[str, ...]], TransformationOperation] = {}
    planner_only: list[TransformationOperation] = []
    for operation in plan.operations:
        signature = _signature(operation.operation_type, operation.target_columns)
        request = by_signature.get(signature)
        if request is not None:
            if operation.parameters == request.parameters:
                satisfying[signature] = operation
            # Parameters differ: the request wins, so the planner's version is
            # dropped here and re-created below from the request itself.
            continue
        if operation.operation_type is not OperationType.DROP_COLUMN and (
            operation.target_columns
            and all(column in dropped_columns for column in operation.target_columns)
        ):
            # The user asked for this column to go, so preparing it first is not
            # work they asked for.
            continue
        planner_only.append(operation)

    taken_ids = {operation.operation_id for operation in planner_only}
    taken_ids.update(operation.operation_id for operation in satisfying.values())

    # The requests, in the order the objective stated them. "finally drop the
    # duplicates" therefore lands last, and the imputations it follows are
    # measured on the pre-deduplication data exactly as written.
    requested: list[TransformationOperation] = []
    for request in allowed:
        signature = _signature(request.operation_type, request.target_columns)
        existing = satisfying.get(signature)
        requested.append(
            existing if existing is not None else _as_operation(request, taken_ids)
        )

    # Planner-only drops still go last; nothing the user asked for may target a
    # column a planner drop removed.
    planner_non_drop = [
        operation
        for operation in planner_only
        if operation.operation_type is not OperationType.DROP_COLUMN
    ]
    planner_drops = [
        operation
        for operation in planner_only
        if operation.operation_type is OperationType.DROP_COLUMN
    ]
    operations = tuple(planner_non_drop + requested + planner_drops)
    if operations == tuple(plan.operations):
        return plan
    return create_transformation_plan(
        dataset_id=plan.dataset_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        version=plan.version,
        operations=operations,
        summary=plan.summary,
    )


class RequestAwarePlanner:
    """Wrap any planner so compiled requests survive its proposal.

    The live crew plans freely and then this layer reconciles the result with
    what the user actually asked for. It satisfies the ``Planner`` protocol and
    forwards every other attribute, so the agent trace the UI reads is still the
    inner planner's.
    """

    def __init__(
        self,
        inner: object,
        requested_operations: tuple[RequestedOperation, ...],
    ) -> None:
        self._inner = inner
        self._requested_operations = tuple(requested_operations)

    @property
    def inner(self) -> object:
        return self._inner

    def accept_requests(
        self,
        requested_operations: tuple[RequestedOperation, ...],
    ) -> None:
        self._requested_operations = tuple(requested_operations)
        accept = getattr(self._inner, "accept_requests", None)
        if callable(accept):
            accept(self._requested_operations)

    def propose(
        self,
        context: PlanningContext,
        *,
        attempt: int,
    ) -> TransformationPlan:
        plan = self._inner.propose(context, attempt=attempt)
        return enforce_requested_operations(
            plan,
            self._requested_operations,
            context,
        )

    def __getattr__(self, name: str) -> object:
        # Trace and any other planner attribute the shell reads.
        return getattr(self._inner, name)


__all__ = [
    "ENFORCEABLE_REQUEST_TYPES",
    "RequestAwarePlanner",
    "enforce_requested_operations",
    "enforceable_requests",
]
