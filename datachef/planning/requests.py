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

Two policies are deliberate:

* ``CAST_COLUMN`` requests are answered by a diagnosis-driven conversion plan,
  or classified as already satisfied when the row-free schema already has a
  numeric dtype. Ambiguous and destructive requests remain blocked.
* A ``DEDUPLICATE_BY_KEYS`` request is enforced only when the row-loss estimator
  can price it. Planning an unpriced deduplication would show the human "0 rows
  removed" for an operation that could empty the table, so it stays unplanned
  and is reported instead.

Enforcement is idempotent: applying it to a plan that already satisfies the
requests returns an equivalent plan, so a request-aware planner and this layer
may both run without duplicating work.
"""

from __future__ import annotations

from pandas.api.types import is_bool_dtype, is_numeric_dtype

from datachef.contracts import (
    CastColumnParameters,
    CastTarget,
    OperationType,
    PlanningContext,
    RequestAssessment,
    RequestAssessmentStatus,
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
        OperationType.COMPUTE_COLUMN,
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
    conditional_drop_exclusions: tuple[str, ...] = (),
) -> TransformationPlan:
    """Return a plan that honours every enforceable request, order-safe."""

    allowed = enforceable_requests(requested_operations, context)
    explicit_drop_columns = {
        column
        for request in allowed
        if request.operation_type is OperationType.DROP_COLUMN
        for column in request.target_columns
    }
    excluded_drop_columns = set(conditional_drop_exclusions).difference(
        explicit_drop_columns
    )
    if not allowed and not excluded_drop_columns:
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
        if (
            operation.operation_type is OperationType.DROP_COLUMN
            and excluded_drop_columns.intersection(operation.target_columns)
        ):
            # A locally measured false conditional is authoritative. A planner
            # cannot reinterpret it as permission to drop the same column.
            continue
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


def assess_requested_operation(
    request: RequestedOperation,
    context: PlanningContext,
    plan: TransformationPlan,
) -> RequestAssessment:
    """Classify one request using only the canonical plan and row-free schema."""

    def parameters_match(operation: TransformationOperation) -> bool:
        if (
            request.operation_type is OperationType.CAST_COLUMN
            and isinstance(request.parameters, CastColumnParameters)
            and isinstance(operation.parameters, CastColumnParameters)
        ):
            # The request says what dtype is wanted. The validated plan owns
            # the execution error policy and QA still rejects destructive loss.
            return operation.parameters.target_type is request.parameters.target_type
        return operation.parameters == request.parameters

    matching = tuple(
        operation.operation_id
        for operation in plan.operations
        if operation.operation_type is request.operation_type
        and operation.target_columns == request.target_columns
        and parameters_match(operation)
    )
    if matching:
        return RequestAssessment(
            request_id=request.request_id,
            status=RequestAssessmentStatus.PLANNED,
            matched_operation_ids=matching,
        )

    if (
        request.operation_type is OperationType.CAST_COLUMN
        and len(request.target_columns) == 1
        and isinstance(request.parameters, CastColumnParameters)
        and request.parameters.target_type is CastTarget.NUMERIC
    ):
        target = request.target_columns[0]
        schema = next((item for item in context.column_schema if item.name == target), None)
        if schema is not None and is_numeric_dtype(schema.dtype) and not is_bool_dtype(
            schema.dtype
        ):
            return RequestAssessment(
                request_id=request.request_id,
                status=RequestAssessmentStatus.ALREADY_SATISFIED,
            )

    return RequestAssessment(
        request_id=request.request_id,
        status=RequestAssessmentStatus.BLOCKED_UNPLANNED,
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
        conditional_drop_exclusions: tuple[str, ...] = (),
    ) -> None:
        self._inner = inner
        self._requested_operations = tuple(requested_operations)
        self._conditional_drop_exclusions = tuple(conditional_drop_exclusions)

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
            self._conditional_drop_exclusions,
        )

    def __getattr__(self, name: str) -> object:
        # Trace and any other planner attribute the shell reads.
        return getattr(self._inner, name)


__all__ = [
    "ENFORCEABLE_REQUEST_TYPES",
    "RequestAwarePlanner",
    "assess_requested_operation",
    "enforce_requested_operations",
    "enforceable_requests",
]
