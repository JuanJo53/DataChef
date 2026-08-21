"""Compile a free-form objective into typed requests, locally and deterministically.

This is the layer that was missing. The user types an objective in prose; the
planner may only act on typed, allow-listed requests. Nothing here reaches a
provider: the objective text, the raw frame and every value read from it stay in
this process. What leaves is a tuple of ``RequestedTransformation``, which
carries an operation type, column names and closed parameter enums, and no
free-form text at all.

Two deliberate properties:

* **Deterministic.** No model, no clock, no randomness. The same objective and
  the same frame always compile to the same requests in the same order.
* **Conservative.** A clause that cannot be matched to a supported operation
  compiles to nothing, and the agent remains free to report it as out of scope.
  Compiling nothing is always safe: the plan simply does not contain that
  operation, and the human sees what is and is not there.

The conditional forms are evaluated against locally measured facts rather than
guessed, because "over 40% null" is a measurement and "there is no mode" is a
question about the data.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datachef.application.models import ApplicationFinding, RequestedTransformation

from datachef.contracts import (
    ComputeColumnParameters,
    ComputeOperator,
    DiagnosticReport,
    DropColumnParameters,
    ImputeMissingParameters,
    ImputeStrategy,
    KeepPolicy,
    OperationType,
    DeduplicateByKeysParameters,
)

# Clause boundaries. "otherwise" is kept attached to the clause before it so a
# conditional and its alternative are read as one unit.
_CLAUSE_SPLIT = re.compile(r"[,.;]|\bfinally\b|\bthen\b|\band then\b", re.IGNORECASE)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

_STRATEGY_WORDS: tuple[tuple[str, ImputeStrategy], ...] = (
    ("median", ImputeStrategy.MEDIAN),
    ("mean", ImputeStrategy.MEAN),
    ("average", ImputeStrategy.MEAN),
    ("mode", ImputeStrategy.MODE),
)

_DROP_WORDS = ("drop", "remove", "delete", "discard")
_IMPUTE_WORDS = ("impute", "fill", "replace the missing", "replace missing")
_DUPLICATE_WORDS = ("duplicate", "deduplicate", "dedupe")
_NULL_WORDS = ("nulls", "null", "missing", "nan", "empty")
_ZERO_WORDS = ("0s", "zeroes", "zeros", "zero values", "0 values", "zero")
_NO_MODE_WORDS = ("no mode", "without a mode", "there is no mode", "there's no mode")
_KEEP_ONLY_START = re.compile(r"\b(?:keep|retain|preserve)\s+only\b", re.IGNORECASE)
_KEEP_ONLY_END = re.compile(r"[.;]|\b(?:and\s+then|then|finally)\b", re.IGNORECASE)
_COMPUTE_START = re.compile(r"^\s*(?:compute|calculate|derive|create)\s+", re.IGNORECASE)
_COMPUTE_OPERATORS: tuple[tuple[str, ComputeOperator], ...] = (
    ("multiplied by", ComputeOperator.MULTIPLY),
    ("divided by", ComputeOperator.DIVIDE),
    ("subtracted from", ComputeOperator.SUBTRACT),
    ("added to", ComputeOperator.ADD),
    ("times", ComputeOperator.MULTIPLY),
    ("plus", ComputeOperator.ADD),
    ("minus", ComputeOperator.SUBTRACT),
)
_EXACTLY_ONE_DISTINCT = re.compile(
    r"\b(?:constant|(?:has|contains|with)\s+(?:a\s+)?(?:single|only\s+one)"
    r"(?:\s+distinct)?\s+value|(?:has|contains|with)\s+(?:exactly\s+)?one"
    r"\s+distinct\s+value)\b",
    re.IGNORECASE,
)
_MORE_THAN_ONE_DISTINCT = re.compile(
    r"\b(?:has|contains|with)\s+more\s+than\s+one\s+distinct\s+value\b",
    re.IGNORECASE,
)
_GENERIC_ONE_DISTINCT = re.compile(
    r"\b(?:if\s+there\s+are\s+)?columns?\s+with\s+(?:exactly\s+)?one\s+"
    r"distinct\s+value\b.*\b(?:drop|remove|delete|discard)\s+"
    r"(?:them|those\s+columns?)\b",
    re.IGNORECASE,
)
_GENERIC_ONE_DISTINCT_CONDITION = re.compile(
    r"\b(?:if\s+there\s+are\s+)?columns?\s+with\s+(?:exactly\s+)?one\s+"
    r"distinct\s+value\b",
    re.IGNORECASE,
)
_PREDICTION_TARGET = re.compile(
    r"\b(?:predict|predicting|target(?:\s+variable)?(?:\s+is)?|forecast)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ColumnFacts:
    """Locally measured facts one clause may need. Counts, never values."""

    column: str
    row_count: int
    null_count: int
    zero_count: int
    unique_count: int
    has_repeated_value: bool

    @property
    def null_pct(self) -> float:
        if not self.row_count:
            return 0.0
        return self.null_count / self.row_count * 100.0

    @property
    def mode_exists(self) -> bool:
        """A mode exists when some non-null value occurs more than once.

        Deliberately stricter than ``Series.mode()``, which is empty only for an
        all-null column and otherwise returns *every* value when each occurs
        once. On a column of unique product titles that would report a mode and
        then impute an arbitrary single observation. "There is no mode" is read
        here as "no value repeats", which is the sense the request carries.
        """

        return self.has_repeated_value


@dataclass(frozen=True, slots=True)
class ObjectiveCompilation:
    """Typed local result: executable requests plus sanitized compilation evidence."""

    requests: tuple[RequestedTransformation, ...]
    keep_only_columns: tuple[str, ...]
    findings: tuple[ApplicationFinding, ...]
    conditional_drop_exclusions: tuple[str, ...]


def measure_columns(frame: pd.DataFrame) -> dict[str, ColumnFacts]:
    """Measure every column locally. The frame never leaves this process."""

    row_count = int(len(frame))
    facts: dict[str, ColumnFacts] = {}
    for column in frame.columns:
        series = frame[column]
        try:
            zero_count = int(series.eq(0).sum())
        except (TypeError, ValueError):
            zero_count = 0
        try:
            counts = series.value_counts(dropna=True)
            has_repeated = bool(len(counts) and int(counts.iloc[0]) >= 2)
        except (TypeError, ValueError):
            has_repeated = False
        facts[str(column)] = ColumnFacts(
            column=str(column),
            row_count=row_count,
            null_count=int(series.isna().sum()),
            zero_count=zero_count,
            unique_count=int(series.nunique(dropna=True)),
            has_repeated_value=has_repeated,
        )
    return facts


def _clauses(objective: str) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_SPLIT.split(objective) if part and part.strip()]
    merged: list[str] = []
    for part in parts:
        # An "otherwise" belongs to the condition it qualifies.
        if merged and part.lower().startswith("otherwise"):
            merged[-1] = merged[-1] + ", " + part
        elif (
            merged
            and _GENERIC_ONE_DISTINCT_CONDITION.search(merged[-1])
            and re.match(
                r"^\s*(?:drop|remove|delete|discard)\s+(?:them|those\s+columns?)\b",
                part,
                flags=re.IGNORECASE,
            )
        ):
            merged[-1] = merged[-1] + ", " + part
        else:
            merged.append(part)
    return merged


def _columns_in(clause: str, known: tuple[str, ...]) -> tuple[str, ...]:
    """Column names mentioned in a clause, longest first so prefixes lose.

    Order follows the clause, not the schema, so "based on the asin column"
    nominates asin rather than whatever came first in the table.
    """

    lowered = clause.lower()
    hits: list[tuple[int, str]] = []
    for column in sorted(known, key=len, reverse=True):
        pattern = re.compile(r"(?<![0-9A-Za-z_])" + re.escape(column.lower()) + r"(?![0-9A-Za-z_])")
        match = pattern.search(lowered)
        if match is None:
            # Treat punctuation/whitespace variants of a complete compound
            # label as that one label before considering shorter labels inside
            # it.  This makes ``Profit From Sales w/o discount`` resolve to the
            # real compound column even when the CSV header contains doubled
            # spaces, without turning the words Profit and Sales into two
            # destructive requests.
            tokens = _tokens(column)
            if len(tokens) >= 2:
                flexible = re.compile(
                    r"(?<![0-9A-Za-z])"
                    + r"[^0-9A-Za-z]+".join(re.escape(token) for token in tokens)
                    + r"(?![0-9A-Za-z])",
                    re.IGNORECASE,
                )
                match = flexible.search(clause)
        if match is None:
            continue
        if any(
            match.start() >= start and match.end() <= end
            for start, end in [(h[0], h[0] + len(h[1])) for h in hits]
        ):
            continue
        hits.append((match.start(), column))
    # A hand-written objective may name a column slightly differently from the
    # schema; resolve those only when the reference is unambiguous.
    matched = {column for _, column in hits}
    hits.extend(_relaxed_matches(clause, known, matched))
    return tuple(column for _, column in sorted(hits))



def _tokens(name: str) -> tuple[str, ...]:
    """Split an identifier into lowercase words: boughtInLastMonth -> bought in last month."""

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return tuple(part.lower() for part in re.split(r"[^0-9A-Za-z]+", spaced) if part)


def _is_ordered_subsequence(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    position = 0
    for token in needle:
        while position < len(haystack) and haystack[position] != token:
            position += 1
        if position == len(haystack):
            return False
        position += 1
    return True


def _relaxed_matches(
    clause: str,
    known: tuple[str, ...],
    already: set[str],
) -> list[tuple[int, str]]:
    """Resolve a near-miss column reference, deterministically and only when unique.

    An objective written by hand says "boughtLastMonth" for a column actually
    called "boughtInLastMonth". Rather than fuzzy distance, this asks a structural
    question: are the words of the written name an ordered subsequence of the
    words of a real column name? "bought last month" is a subsequence of "bought
    in last month", so it resolves; "price" is not a subsequence of anything else.

    Two guards keep it honest. The written reference must carry at least two
    words, so a single loose word can never rename a column. And if more than one
    column matches, nothing is resolved -- an ambiguous reference is left alone
    rather than guessed at.
    """

    hits: list[tuple[int, str]] = []
    column_tokens = {column: _tokens(column) for column in known}
    for match in re.finditer(r"[0-9A-Za-z_]{4,}", clause):
        word = match.group(0)
        needle = _tokens(word)
        if len(needle) < 2:
            continue
        candidates = [
            column
            for column, tokens in column_tokens.items()
            if column not in already
            and needle != tokens
            and _is_ordered_subsequence(needle, tokens)
        ]
        if len(candidates) != 1:
            continue
        column = candidates[0]
        if any(column == existing for _, existing in hits):
            continue
        hits.append((match.start(), column))

    # Also resolve a spaced phrase to one camel/snake schema name. For example,
    # "best seller" is the unique two-token suffix of ``isBestSeller``. This is
    # still structural matching, not fuzzy distance, and ambiguity returns no
    # match.
    words = tuple(
        (match.start(), match.group(0).lower())
        for match in re.finditer(r"[0-9A-Za-z]+", clause)
    )
    phrase_candidates: list[tuple[int, str]] = []
    for start in range(len(words)):
        for length in range(2, min(6, len(words) - start) + 1):
            needle = tuple(word for _, word in words[start : start + length])
            candidates = [
                column
                for column, tokens in column_tokens.items()
                if column not in already
                and column not in {item for _, item in hits}
                and _is_ordered_subsequence(needle, tokens)
            ]
            if len(candidates) == 1:
                phrase_candidates.append((words[start][0], candidates[0]))
    for position, column in phrase_candidates:
        if sum(item == column for _, item in phrase_candidates) != 1:
            continue
        if column not in {item for _, item in hits}:
            hits.append((position, column))
    return hits


def _prediction_target(objective: str, known: tuple[str, ...]) -> str | None:
    match = _PREDICTION_TARGET.search(objective)
    if match is None:
        return None
    window = objective[match.end() : match.end() + 120]
    columns = _columns_in(window, known)
    return columns[0] if columns else None


def _mentions(clause: str, words: tuple[str, ...]) -> bool:
    """Whole-word matching. "mode" must not be found inside "modelling"."""

    lowered = clause.lower()
    return any(
        re.search(r"(?<![0-9a-z])" + re.escape(word) + r"(?![0-9a-z])", lowered)
        for word in words
    )


def _threshold(clause: str) -> float | None:
    match = _PERCENT.search(clause)
    return float(match.group(1)) if match else None


def _strategy(clause: str) -> ImputeStrategy | None:
    lowered = clause.lower()
    best: tuple[int, ImputeStrategy] | None = None
    for word, strategy in _STRATEGY_WORDS:
        # Whole word only: "model" and "modelling" are not the mode strategy.
        match = re.search(
            r"(?<![0-9a-z])" + re.escape(word) + r"(?![0-9a-z])", lowered
        )
        if match is None:
            continue
        if best is None or match.start() < best[0]:
            best = (match.start(), strategy)
    return best[1] if best else None


def _parse_compute_clause(
    clause: str,
    known: tuple[str, ...],
) -> ComputeColumnParameters | None:
    """Parse a small closed family of binary-column arithmetic requests."""

    start = _COMPUTE_START.match(clause)
    if start is None:
        return None
    exact = {
        name.casefold(): name
        for name in known
        if sum(item.casefold() == name.casefold() for item in known) == 1
    }
    names = "|".join(re.escape(name) for name in sorted(known, key=len, reverse=True))
    if not names:
        return None
    output = r"(?P<output>[A-Za-z_][0-9A-Za-z_ ]*?)"
    column = rf"(?P<{{name}}>{names})"
    body = clause[start.end() :].strip().rstrip(".").strip()
    body = re.sub(r"^a\s+new\s+column\s+called\s+", "", body, flags=re.IGNORECASE)

    parsed: tuple[str, str, str, ComputeOperator] | None = None
    symbol = re.fullmatch(
        output
        + r"\s*=\s*"
        + column.format(name="left")
        + r"\s*(?P<operator>[+\-*/])\s*"
        + column.format(name="right"),
        body,
        flags=re.IGNORECASE,
    )
    if symbol is not None:
        operators = {
            "+": ComputeOperator.ADD,
            "-": ComputeOperator.SUBTRACT,
            "*": ComputeOperator.MULTIPLY,
            "/": ComputeOperator.DIVIDE,
        }
        parsed = (
            symbol.group("output"),
            symbol.group("left"),
            symbol.group("right"),
            operators[symbol.group("operator")],
        )

    if parsed is None:
        by_verb = re.fullmatch(
            output
            + r"\s+by\s+(?P<verb>multiplying|adding|dividing)\s+"
            + column.format(name="left")
            + r"\s+(?:by|and|to)\s+"
            + column.format(name="right"),
            body,
            flags=re.IGNORECASE,
        )
        if by_verb is not None:
            operators = {
                "multiplying": ComputeOperator.MULTIPLY,
                "adding": ComputeOperator.ADD,
                "dividing": ComputeOperator.DIVIDE,
            }
            parsed = (
                by_verb.group("output"),
                by_verb.group("left"),
                by_verb.group("right"),
                operators[by_verb.group("verb").casefold()],
            )

    if parsed is None:
        from_form = re.fullmatch(
            output
            + r"\s+from\s+"
            + column.format(name="left")
            + r"\s+(?P<verb>multiplied|added|divided)\s+(?:by|to)\s+"
            + column.format(name="right"),
            body,
            flags=re.IGNORECASE,
        )
        if from_form is not None:
            operators = {
                "multiplied": ComputeOperator.MULTIPLY,
                "added": ComputeOperator.ADD,
                "divided": ComputeOperator.DIVIDE,
            }
            parsed = (
                from_form.group("output"),
                from_form.group("left"),
                from_form.group("right"),
                operators[from_form.group("verb").casefold()],
            )

    if parsed is None:
        output_and_inputs = re.split(
            r"\s+as\s+", body, maxsplit=1, flags=re.IGNORECASE
        )
        if len(output_and_inputs) == 2:
            output_column, inputs = (part.strip() for part in output_and_inputs)
            operator_match: tuple[int, int, ComputeOperator] | None = None
            for phrase, operator in _COMPUTE_OPERATORS:
                match = re.search(
                    r"(?<![0-9A-Za-z_])"
                    + re.escape(phrase)
                    + r"(?![0-9A-Za-z_])",
                    inputs,
                    flags=re.IGNORECASE,
                )
                if match is None:
                    continue
                if operator_match is not None:
                    return None
                operator_match = (match.start(), match.end(), operator)
            if operator_match is not None:
                start_index, end_index, operator = operator_match
                parsed = (
                    output_column,
                    inputs[:start_index].strip(),
                    inputs[end_index:].strip(),
                    operator,
                )

    if parsed is None:
        return None
    output_column, left_name, right_name, operator = parsed
    left = exact.get(left_name.casefold())
    right = exact.get(right_name.casefold())
    if left is None or right is None:
        return None
    return ComputeColumnParameters(
        left_column=left,
        right_column=right,
        output_column=output_column,
        operator=operator,
    )


def _compile_standard_requests(
    objective: str,
    frame: pd.DataFrame,
    report: DiagnosticReport,
    findings: list[ApplicationFinding] | None = None,
    forbidden_conditional_drops: frozenset[str] = frozenset(),
    conditional_drop_exclusions: list[str] | None = None,
) -> tuple[RequestedTransformation, ...]:
    """Compile an objective into typed requests. Returns RequestedTransformation.

    Imported lazily to keep ``datachef.application.models`` free to import this
    module's contracts without a cycle.
    """

    from datachef.application.models import ApplicationFinding, RequestedTransformation

    if not objective or not objective.strip():
        return ()

    known = tuple(str(column) for column in frame.columns)
    facts = measure_columns(frame)
    key_metrics = {
        tuple(metric.key_columns) for metric in report.key_duplicate_metrics
    }

    requests: list[RequestedTransformation] = []
    claimed: set[str] = set()

    def add(operation_type, columns, parameters, *, before_dedup: bool = False) -> None:
        # One request per column per operation family; the first clause wins, so
        # a later loose mention cannot override an explicit earlier instruction.
        marker = f"{operation_type.value}:{','.join(columns)}"
        if marker in claimed:
            return
        for column in columns:
            if f"{OperationType.DROP_COLUMN.value}:{column}" in claimed:
                return  # already being dropped; nothing else is worth planning
        claimed.add(marker)
        suffix = "-".join(re.sub(r"[^0-9A-Za-z]+", "", column) or "col" for column in columns)
        request = RequestedTransformation(
                request_id=f"request-{operation_type.value.lower()}-{suffix}",
                operation_type=operation_type,
                target_columns=tuple(columns),
                parameters=parameters,
        )
        if before_dedup:
            index = next(
                (
                    index
                    for index, existing in enumerate(requests)
                    if existing.operation_type is OperationType.DEDUPLICATE_BY_KEYS
                ),
                len(requests),
            )
            requests.insert(index, request)
        else:
            requests.append(request)

    def exclude(column: str) -> None:
        if conditional_drop_exclusions is not None and column not in conditional_drop_exclusions:
            conditional_drop_exclusions.append(column)

    def condition_not_met() -> None:
        if findings is not None:
            findings.append(
                ApplicationFinding(
                    code="CONDITIONAL_DROP_NOT_MET",
                    blocking=False,
                    safe_message=(
                        "A requested conditional drop was measured locally and its "
                        "condition was not met."
                    ),
                )
            )

    def has_specific_non_drop_request(column: str) -> bool:
        return any(
            marker.endswith(f":{column}")
            and not marker.startswith(f"{OperationType.DROP_COLUMN.value}:")
            for marker in claimed
        )

    for clause in _clauses(objective):
        if _COMPUTE_START.match(clause):
            parameters = _parse_compute_clause(clause, known)
            if parameters is None:
                if findings is not None:
                    findings.append(
                        ApplicationFinding(
                            code="COMPUTE_COLUMN_UNSUPPORTED",
                            blocking=True,
                            safe_message=(
                                "The computed-column request must name two exact current "
                                "columns and one supported arithmetic operator."
                            ),
                        )
                    )
            else:
                add(
                    OperationType.COMPUTE_COLUMN,
                    (parameters.left_column, parameters.right_column),
                    parameters,
                )
            continue

        if _GENERIC_ONE_DISTINCT.search(clause):
            matching = tuple(
                column for column in known if facts[column].unique_count == 1
            )
            nonmatching = tuple(column for column in known if column not in matching)
            for column in nonmatching:
                exclude(column)
            if not matching:
                condition_not_met()
                continue
            if len(matching) == len(known):
                for column in matching:
                    exclude(column)
                if findings is not None:
                    findings.append(
                        ApplicationFinding(
                            code="CONDITIONAL_DROP_ALL_COLUMNS",
                            blocking=True,
                            safe_message=(
                                "The conditional request would remove every column, "
                                "so no conditional drops were planned."
                            ),
                        )
                    )
                continue
            for column in matching:
                if has_specific_non_drop_request(column):
                    exclude(column)
                    if findings is not None:
                        findings.append(
                            ApplicationFinding(
                                code="CONDITIONAL_DROP_SUPERSEDED",
                                blocking=False,
                                safe_message=(
                                    "A broader conditional drop was superseded by an "
                                    "earlier, more specific requested transformation."
                                ),
                            )
                        )
                    continue
                if column in forbidden_conditional_drops:
                    exclude(column)
                    if findings is not None:
                        findings.append(
                            ApplicationFinding(
                                code="CONDITIONAL_DROP_COLUMN_PROTECTED",
                                blocking=True,
                                safe_message=(
                                    "A constant-column drop cannot remove the prediction "
                                    "target, sensitive column, or a required/protected column."
                                ),
                            )
                        )
                    continue
                add(
                    OperationType.DROP_COLUMN,
                    (column,),
                    DropColumnParameters(),
                    before_dedup=True,
                )
            continue

        columns = _columns_in(clause, known)
        if not columns:
            continue

        # Deduplication is explicit and column-anchored: "duplicates based on X".
        if _mentions(clause, _DUPLICATE_WORDS):
            keys = tuple(columns)
            if keys in key_metrics or len(keys) >= 1:
                add(
                    OperationType.DEDUPLICATE_BY_KEYS,
                    keys,
                    DeduplicateByKeysParameters(keys=keys, keep=KeepPolicy.FIRST),
                )
            continue

        threshold = _threshold(clause)
        wants_drop = _mentions(clause, _DROP_WORDS)
        wants_impute = _mentions(clause, _IMPUTE_WORDS)
        strategy = _strategy(clause)

        # --- conditional constant-column drop -------------------------------
        predicate = None
        if wants_drop and _MORE_THAN_ONE_DISTINCT.search(clause):
            predicate = "MORE_THAN_ONE"
        elif wants_drop and _EXACTLY_ONE_DISTINCT.search(clause):
            predicate = "EXACTLY_ONE"
        if predicate is not None:
            for column in columns:
                fact = facts.get(column)
                matched = bool(
                    fact is not None
                    and (
                        (predicate == "EXACTLY_ONE" and fact.unique_count == 1)
                        or (predicate == "MORE_THAN_ONE" and fact.unique_count > 1)
                    )
                )
                if not matched:
                    exclude(column)
                    condition_not_met()
                    continue
                if column in forbidden_conditional_drops:
                    if findings is not None:
                        findings.append(
                            ApplicationFinding(
                                code="CONDITIONAL_DROP_COLUMN_PROTECTED",
                                blocking=True,
                                safe_message=(
                                    "A constant-column drop cannot remove the prediction "
                                    "target or a required/protected column."
                                ),
                            )
                        )
                    continue
                add(OperationType.DROP_COLUMN, (column,), DropColumnParameters())
            continue

        # --- conditional forms, decided against measured facts ---------------
        if threshold is not None and _mentions(clause, _NULL_WORDS):
            for column in columns:
                fact = facts.get(column)
                if fact is None:
                    continue
                over = fact.null_pct > threshold
                # "over N% null and there is no mode -> drop, otherwise impute"
                if _mentions(clause, _NO_MODE_WORDS) or _mentions(clause, ("mode",)):
                    if over and not fact.mode_exists:
                        add(OperationType.DROP_COLUMN, (column,), DropColumnParameters())
                    elif fact.null_count:
                        add(
                            OperationType.IMPUTE_MISSING,
                            (column,),
                            ImputeMissingParameters(strategy=ImputeStrategy.MODE),
                        )
                    continue
                # "over N% null and zeros as values -> drop"
                if _mentions(clause, _ZERO_WORDS):
                    null_or_zero = bool(
                        re.search(
                            r"\b(?:nulls?|missing(?:\s+values?)?)\s+or\s+"
                            r"(?:zeroes|zeros|0s|zero\s+values?)\b",
                            clause,
                            flags=re.IGNORECASE,
                        )
                    )
                    affected_pct = (
                        (fact.null_count + fact.zero_count) / fact.row_count * 100.0
                        if fact.row_count
                        else 0.0
                    )
                    matched = (
                        affected_pct > threshold
                        if null_or_zero
                        else over and fact.zero_count > 0
                    )
                    if matched:
                        add(OperationType.DROP_COLUMN, (column,), DropColumnParameters())
                    else:
                        exclude(column)
                        condition_not_met()
                    continue
                if wants_drop and over:
                    add(OperationType.DROP_COLUMN, (column,), DropColumnParameters())
                elif wants_drop:
                    exclude(column)
                    condition_not_met()
                elif wants_impute and strategy is not None and fact.null_count:
                    add(
                        OperationType.IMPUTE_MISSING,
                        (column,),
                        ImputeMissingParameters(strategy=strategy),
                    )
            continue

        # --- unconditional imputation ---------------------------------------
        if (wants_impute or strategy is not None) and strategy is not None:
            for column in columns:
                fact = facts.get(column)
                if fact is not None and fact.null_count:
                    add(
                        OperationType.IMPUTE_MISSING,
                        (column,),
                        ImputeMissingParameters(strategy=strategy),
                    )
            continue

        # --- unconditional drop ---------------------------------------------
        if wants_drop:
            for column in columns:
                add(OperationType.DROP_COLUMN, (column,), DropColumnParameters())

    return tuple(requests)


def _keep_only_span(objective: str) -> tuple[int, int, str] | None:
    matches = tuple(_KEEP_ONLY_START.finditer(objective))
    if not matches:
        return None
    if len(matches) != 1:
        first = matches[0]
        return (first.start(), len(objective), "")
    start = matches[0]
    tail = objective[start.end() :]
    end_match = _KEEP_ONLY_END.search(tail)
    segment_end = start.end() + (end_match.start() if end_match else len(tail))
    return (start.start(), segment_end, objective[start.end() : segment_end])


def _resolve_keep_only(
    segment: str,
    known: tuple[str, ...],
) -> tuple[tuple[str, ...], bool]:
    """Resolve exact case-insensitive schema names; never fuzzy-match."""

    hits: list[tuple[int, int, str]] = []
    ambiguous = False
    for column in sorted(known, key=len, reverse=True):
        pattern = re.compile(
            r"(?<![0-9A-Za-z_])" + re.escape(column) + r"(?![0-9A-Za-z_])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(segment):
            same_span = [item for item in hits if item[:2] == match.span()]
            if same_span and all(item[2] != column for item in same_span):
                ambiguous = True
            if any(
                match.start() < end and match.end() > start
                for start, end, _ in hits
                if (start, end) != match.span()
            ):
                continue
            hits.append((match.start(), match.end(), column))

    selected_by_position: list[str] = []
    for _, _, column in sorted(hits):
        if column not in selected_by_position:
            selected_by_position.append(column)

    residual = list(segment)
    for start, end, _ in hits:
        residual[start:end] = " " * (end - start)
    remainder = "".join(residual)
    remainder = re.sub(r"[,/&+]", " ", remainder)
    remainder = re.sub(r"\b(?:and|the|column|columns|fields?)\b", " ", remainder, flags=re.IGNORECASE)
    unresolved = bool(re.sub(r"\s+", "", remainder))
    return tuple(selected_by_position), ambiguous or unresolved or not selected_by_position


def build_keep_only_requests(
    selected_columns: tuple[str, ...],
    available_columns: tuple[str, ...],
) -> tuple[RequestedTransformation, ...]:
    """Compile a validated keep selection into ordinary drop requests."""

    from datachef.application.models import ApplicationFinding, RequestedTransformation

    if not selected_columns:
        return ()
    if any(not column.strip() for column in selected_columns):
        raise ValueError("keep-only columns must contain non-whitespace text")
    if len(set(selected_columns)) != len(selected_columns):
        raise ValueError("keep-only columns must be unique")
    if not set(selected_columns).issubset(available_columns):
        raise ValueError("keep-only columns must exist in the current schema")
    if not available_columns:
        raise ValueError("keep-only requires a nonempty schema")
    selected = set(selected_columns)
    dropped = tuple(column for column in available_columns if column not in selected)
    return tuple(
        RequestedTransformation(
            request_id=(
                "request-keep-only-drop-"
                + (re.sub(r"[^0-9A-Za-z]+", "", column) or "col")
            ),
            operation_type=OperationType.DROP_COLUMN,
            target_columns=(column,),
            parameters=DropColumnParameters(),
        )
        for column in dropped
    )


def compile_objective(
    objective: str,
    frame: pd.DataFrame,
    report: DiagnosticReport,
    *,
    required_columns: tuple[str, ...] = (),
    protected_columns: tuple[str, ...] = (),
) -> ObjectiveCompilation:
    """Compile supported prose and report unresolved keep-only clauses safely."""

    from datachef.application.models import ApplicationFinding

    known = tuple(str(column) for column in frame.columns)
    target = _prediction_target(objective, known)
    pii_columns = tuple(
        profile.name for profile in report.column_profiles if profile.possible_pii
    )
    forbidden_conditional_drops = frozenset(
        (
            *required_columns,
            *protected_columns,
            *pii_columns,
            *((target,) if target else ()),
        )
    )
    compilation_findings: list[ApplicationFinding] = []
    conditional_drop_exclusions: list[str] = []
    span = _keep_only_span(objective)
    if span is None:
        return ObjectiveCompilation(
            requests=_compile_standard_requests(
                objective,
                frame,
                report,
                compilation_findings,
                forbidden_conditional_drops,
                conditional_drop_exclusions,
            ),
            keep_only_columns=(),
            findings=tuple(compilation_findings),
            conditional_drop_exclusions=tuple(
                column for column in known if column in conditional_drop_exclusions
            ),
        )

    start, end, segment = span
    selected, unsupported = _resolve_keep_only(segment, known)
    before = _compile_standard_requests(
        objective[:start],
        frame,
        report,
        compilation_findings,
        forbidden_conditional_drops,
        conditional_drop_exclusions,
    )
    after = _compile_standard_requests(
        objective[end:],
        frame,
        report,
        compilation_findings,
        forbidden_conditional_drops,
        conditional_drop_exclusions,
    )
    if unsupported:
        return ObjectiveCompilation(
            requests=before + after,
            keep_only_columns=(),
            findings=tuple(compilation_findings) + (
                ApplicationFinding(
                    code="KEEP_ONLY_UNSUPPORTED",
                    blocking=True,
                    safe_message=(
                        "The keep-only clause does not resolve to a unique, nonempty "
                        "set of current columns."
                    ),
                ),
            ),
            conditional_drop_exclusions=tuple(
                column for column in known if column in conditional_drop_exclusions
            ),
        )

    canonical = tuple(column for column in known if column in set(selected))
    keep_requests = build_keep_only_requests(canonical, known)
    findings: tuple[ApplicationFinding, ...] = tuple(compilation_findings)
    if not keep_requests:
        findings += (
            ApplicationFinding(
                code="KEEP_ONLY_ALREADY_SATISFIED",
                blocking=False,
                safe_message="The keep-only selection already contains every column.",
            ),
        )
    return ObjectiveCompilation(
        requests=before + keep_requests + after,
        keep_only_columns=canonical,
        findings=findings,
        conditional_drop_exclusions=tuple(
            column for column in known if column in conditional_drop_exclusions
        ),
    )


def compile_requests(
    objective: str,
    frame: pd.DataFrame,
    report: DiagnosticReport,
) -> tuple[RequestedTransformation, ...]:
    """Backward-compatible request-only projection of ``compile_objective``."""

    return compile_objective(objective, frame, report).requests


__all__ = [
    "ColumnFacts",
    "ObjectiveCompilation",
    "build_keep_only_requests",
    "compile_objective",
    "compile_requests",
    "measure_columns",
]
