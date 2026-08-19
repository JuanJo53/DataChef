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

import pandas as pd

from datachef.contracts import (
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
_NULL_WORDS = ("null", "missing", "nan", "empty")
_ZERO_WORDS = ("0s", "zeros", "zero values", "0 values", "zero")
_NO_MODE_WORDS = ("no mode", "without a mode", "there is no mode", "there's no mode")


@dataclass(frozen=True, slots=True)
class ColumnFacts:
    """Locally measured facts one clause may need. Counts, never values."""

    column: str
    row_count: int
    null_count: int
    zero_count: int
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
    return hits


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


def compile_requests(
    objective: str,
    frame: pd.DataFrame,
    report: DiagnosticReport,
) -> tuple[object, ...]:
    """Compile an objective into typed requests. Returns RequestedTransformation.

    Imported lazily to keep ``datachef.application.models`` free to import this
    module's contracts without a cycle.
    """

    from datachef.application.models import RequestedTransformation

    if not objective or not objective.strip():
        return ()

    known = tuple(str(column) for column in frame.columns)
    facts = measure_columns(frame)
    key_metrics = {
        tuple(metric.key_columns) for metric in report.key_duplicate_metrics
    }

    requests: list[RequestedTransformation] = []
    claimed: set[str] = set()

    def add(operation_type, columns, parameters) -> None:
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
        requests.append(
            RequestedTransformation(
                request_id=f"request-{operation_type.value.lower()}-{suffix}",
                operation_type=operation_type,
                target_columns=tuple(columns),
                parameters=parameters,
            )
        )

    for clause in _clauses(objective):
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
                    if over and fact.zero_count > 0:
                        add(OperationType.DROP_COLUMN, (column,), DropColumnParameters())
                    continue
                if wants_drop and over:
                    add(OperationType.DROP_COLUMN, (column,), DropColumnParameters())
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


__all__ = ["ColumnFacts", "compile_requests", "measure_columns"]
