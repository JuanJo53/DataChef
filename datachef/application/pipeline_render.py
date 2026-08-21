"""Deterministic rendering of a standalone pandas script from an approved plan.

This is a pure template function over the human-approved ``TransformationPlan``:
no model is involved, nothing is measured at render time, and the same plan
always renders byte-identical text. The rendered script imports nothing from
DataChef, so an approved plan stays runnable long after this application is
gone.

Fidelity to ``datachef/transform`` is the whole point, and three details of that
package are easy to get wrong:

* The null-key pre-check for ``DEDUPLICATE_BY_KEYS`` lives in the *runner*
  (``run_allowlisted_plan``), not in the handler, and it raises *before* the
  deduplication runs. The rendered script reproduces the guard in that position.
* There is no ``reset_index`` anywhere in ``datachef/transform``, so
  ``drop_duplicates`` leaves a gapped index. The rendered script does not reset
  either: the only ``reset_index(drop=True)`` is applied at bundle time in
  ``artifacts.py``, so this script reproduces the executor's frame and therefore
  the gold table it is bundled beside.
* ``CAST_COLUMN`` lower-cases ``errors`` off the enum, and ``CastErrorPolicy``
  defaults to ``RAISE`` while the rule-based planner passes ``COERCE``. The
  policy is read from the plan and emitted verbatim, never defaulted here.

Nothing in the product may execute what this module renders. Drift between the
rendered text and the executor is closed by the test suite, which runs the
rendered script in a separate interpreter and compares fingerprints; at runtime
the script is bytes to be downloaded, never code to be run.

The wording above is deliberate: ``tests/datachef/test_workflow.py`` scans this
package for the names of execution primitives, and prose about a risk must not
read to that scan as the risk itself.
"""

from __future__ import annotations

import ast

from datachef.contracts import (
    CastColumnParameters,
    ComputeColumnParameters,
    ComputeOperator,
    DropColumnParameters,
    ImputeMissingParameters,
    ImputeStrategy,
    NormalizeNumericTextParameters,
    CastErrorPolicy,
    CastTarget,
    DeduplicateByKeysParameters,
    DropDuplicateRowsParameters,
    KeepPolicy,
    NormalizeMissingTokensParameters,
    OperationType,
    RenameColumnParameters,
    TransformationOperation,
    TransformationPlan,
    TrimWhitespaceParameters,
)

PIPELINE_MEDIA_TYPE = "text/x-python"

_BODY_INDENT = "    "

_TRIPLE = '"""'


def _text(value: str) -> str:
    """Emit a Python string literal. ``repr`` is deterministic for ``str``."""

    return repr(value)


def _text_tuple(values: tuple[str, ...]) -> str:
    """Emit a tuple literal, never a set literal.

    A set literal's *iteration order* depends on the hash seed, so emitting one
    would make the rendered text reorder between processes. Every set the
    handlers build is therefore rebuilt inside the script from an ordered tuple.
    """

    if not values:
        return "()"
    if len(values) == 1:
        return "(" + _text(values[0]) + ",)"
    return "(" + ", ".join(_text(value) for value in values) + ")"


def _text_list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_text(value) for value in values) + "]"


_COMMENT_PLACEHOLDER = "?"

# The comment allow-list. Printable ASCII, enumerated: nothing here can end a
# comment, because the tokenizer ends one only at a line break, and no line
# break is in this set. Printable non-ASCII is admitted separately below so an
# international column name stays readable.
_COMMENT_ALLOWED = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " _-.,:;()[]{}/\\|=+*<>?!@#$%^&'\"`~"
)

# A column name is not a paragraph. Bounding the fragment keeps one hostile
# label from turning the header into a wall of text; the data path is untouched.
_COMMENT_FRAGMENT_LIMIT = 120


def _comment_safe(value: str) -> str:
    """Reduce plan-derived text to characters that cannot leave a comment.

    An allow-list, deliberately, and never a list of forbidden characters: the
    next hostile character is the one nobody enumerated. A character survives
    only if it is in ``_COMMENT_ALLOWED``, or if it is non-ASCII *and*
    ``str.isprintable()``. That second test is a positive Unicode property, not
    an enumeration, and it excludes by construction every C0 and C1 control,
    ``U+2028`` and ``U+2029``, every other line and paragraph separator, the
    bidirectional format controls that could reorder a comment on screen, and
    every surrogate, private-use and unassigned code point. Anything rejected
    becomes a visible placeholder, so a mangled label reads as mangled rather
    than as silently different text.

    This guards the *comment* path only. Plan data that the script acts on goes
    through ``_text``/``_text_list``, which emit Python literals via ``repr``
    and are already exact; nothing here touches them.
    """

    cleaned = "".join(
        character
        if (
            character in _COMMENT_ALLOWED
            or (ord(character) > 0x7F and character.isprintable())
        )
        else _COMMENT_PLACEHOLDER
        for character in value
    )
    if len(cleaned) > _COMMENT_FRAGMENT_LIMIT:
        cleaned = cleaned[:_COMMENT_FRAGMENT_LIMIT] + "..."
    return cleaned


def _keep(policy: KeepPolicy) -> str:
    """Mirror ``datachef.transform.operations._keep_value`` exactly."""

    return "first" if policy is KeepPolicy.FIRST else "last"


def _trim_whitespace_lines(operation: TransformationOperation, slug: str) -> list[str]:
    del slug
    assert isinstance(operation.parameters, TrimWhitespaceParameters)
    return [
        "for _column in " + _text_list(operation.target_columns) + ":",
        "    frame[_column] = frame[_column].map(",
        "        lambda value: value.strip() if isinstance(value, str) else value",
        "    )",
    ]


def _normalize_missing_tokens_lines(
    operation: TransformationOperation,
    slug: str,
) -> list[str]:
    parameters = operation.parameters
    assert isinstance(parameters, NormalizeMissingTokensParameters)
    return [
        "_tokens_" + slug + " = set(" + _text_tuple(parameters.tokens) + ")",
        "_lowered_" + slug + " = {token.casefold() for token in _tokens_" + slug + "}",
        "_case_sensitive_" + slug + " = " + repr(parameters.case_sensitive),
        "",
        "def _normalize_" + slug + "(value):",
        "    if not isinstance(value, str):",
        "        return value",
        "    matches = (",
        "        value in _tokens_" + slug,
        "        if _case_sensitive_" + slug,
        "        else value.casefold() in _lowered_" + slug,
        "    )",
        "    return pd.NA if matches else value",
        "",
        "for _column in " + _text_list(operation.target_columns) + ":",
        "    frame[_column] = frame[_column].map(_normalize_" + slug + ")",
    ]


def _cast_column_lines(operation: TransformationOperation, slug: str) -> list[str]:
    parameters = operation.parameters
    assert isinstance(parameters, CastColumnParameters)
    column = operation.target_columns[0]
    # The handler lower-cases the enum value; it never defaults the policy.
    errors = parameters.errors.value.lower()
    before = "_before_" + slug
    after = "_after_" + slug
    lines = [before + " = frame[" + _text(column) + "].copy(deep=True)"]
    if parameters.target_type is CastTarget.STRING:
        lines.append(after + " = " + before + ".astype('string')")
    elif parameters.target_type is CastTarget.NUMERIC:
        lines.append(
            after + " = pd.to_numeric(" + before + ", errors=" + _text(errors) + ")"
        )
    elif parameters.target_type is CastTarget.BOOLEAN:
        if parameters.errors is CastErrorPolicy.COERCE:
            unmatched = "        return pd.NA"
        else:
            unmatched = (
                "        raise ValueError("
                + _text("value is not in the approved Boolean token catalogue")
                + ")"
            )
        lines.extend(
            [
                "_true_values_" + slug + " = {",
                "    value.casefold() for value in "
                + _text_tuple(parameters.true_values),
                "}",
                "_false_values_" + slug + " = {",
                "    value.casefold() for value in "
                + _text_tuple(parameters.false_values),
                "}",
                "",
                "def _convert_" + slug + "(value):",
                "    if pd.isna(value):",
                "        return pd.NA",
                "    if isinstance(value, bool):",
                "        return value",
                "    normalized = str(value).strip().casefold()",
                "    if normalized in _true_values_" + slug + ":",
                "        return True",
                "    if normalized in _false_values_" + slug + ":",
                "        return False",
                unmatched,
                "",
                after
                + " = "
                + before
                + ".map(_convert_"
                + slug
                + ").astype('boolean')",
            ]
        )
    elif parameters.target_type is CastTarget.DATETIME:
        if parameters.datetime_format is not None:
            datetime_format = _text(parameters.datetime_format)
        else:
            datetime_format = "None"
        lines.extend(
            [
                after + " = pd.to_datetime(",
                "    " + before + ",",
                "    format=" + datetime_format + ",",
                "    errors=" + _text(errors) + ",",
                "    utc=" + repr(parameters.utc) + ",",
                ")",
            ]
        )
    else:  # pragma: no cover - the enum makes this unreachable
        raise ValueError("unsupported cast target")
    lines.append("frame[" + _text(column) + "] = " + after)
    return lines


def _rename_column_lines(operation: TransformationOperation, slug: str) -> list[str]:
    del slug
    parameters = operation.parameters
    assert isinstance(parameters, RenameColumnParameters)
    source = operation.target_columns[0]
    return [
        "frame.rename(",
        "    columns={" + _text(source) + ": " + _text(parameters.new_name) + "},",
        "    inplace=True,",
        ")",
    ]


def _drop_duplicate_rows_lines(
    operation: TransformationOperation,
    slug: str,
) -> list[str]:
    del slug
    parameters = operation.parameters
    assert isinstance(parameters, DropDuplicateRowsParameters)
    # No subset, and no reset_index: the executor leaves the index gapped.
    return [
        "frame = frame.drop_duplicates(keep=" + _text(_keep(parameters.keep)) + ")"
    ]


def _deduplicate_by_keys_lines(
    operation: TransformationOperation,
    slug: str,
) -> list[str]:
    parameters = operation.parameters
    assert isinstance(parameters, DeduplicateByKeysParameters)
    keys = _text_list(parameters.keys)
    # The guard belongs here, before the drop: the runner raises on a null key
    # rather than silently collapsing rows whose identity is unknown.
    return [
        "_key_frame_" + slug + " = frame.loc[:, " + keys + "]",
        "if bool(_key_frame_" + slug + ".isna().any(axis=1).any()):",
        "    raise ValueError("
        + _text("null keys are unsafe for key deduplication")
        + ")",
        "frame = frame.drop_duplicates(",
        "    subset=" + keys + ",",
        "    keep=" + _text(_keep(parameters.keep)) + ",",
        ")",
    ]



# Mirrors datachef.transform.operations._CURRENCY_SYMBOLS and
# _THOUSANDS_SEPARATORS. Emitted as ordered tuples, never set literals, so the
# rendered text cannot reorder between processes.
_CURRENCY_SYMBOLS_ORDERED = ("$", "€", "£", "¥", "₹", "₩", "₽")
_THOUSANDS_SEPARATORS_ORDERED = (",", "_")


def _drop_column_lines(operation: TransformationOperation, slug: str) -> list[str]:
    del slug
    assert isinstance(operation.parameters, DropColumnParameters)
    # frame.drop raises on an absent column, exactly as the handler does.
    return [
        "frame = frame.drop(columns=" + _text_list(operation.target_columns) + ")"
    ]


def _normalize_numeric_text_lines(
    operation: TransformationOperation,
    slug: str,
) -> list[str]:
    parameters = operation.parameters
    assert isinstance(parameters, NormalizeNumericTextParameters)
    lines = [
        "_currency_" + slug + " = set(" + _text_tuple(_CURRENCY_SYMBOLS_ORDERED) + ")",
        "_thousands_" + slug + " = set("
        + _text_tuple(_THOUSANDS_SEPARATORS_ORDERED)
        + ")",
        "",
        "def _strip_" + slug + "(value):",
        "    if not isinstance(value, str):",
        "        return value",
        "    text = value",
    ]
    if parameters.strip_whitespace:
        lines.append("    text = text.strip()")
    lines.extend(
        [
            "    accounting_negative = text.startswith('(') and text.endswith(')')",
            "    if accounting_negative:",
            "        text = text[1:-1]",
        ]
    )
    if parameters.strip_whitespace:
        lines.append("        text = text.strip()")
    if parameters.strip_currency_symbols:
        lines.extend(
            [
                "    text = ''.join(",
                "        character for character in text",
                "        if character not in _currency_" + slug,
                "    )",
            ]
        )
    if parameters.strip_thousands_separators:
        lines.extend(
            [
                "    text = ''.join(",
                "        character for character in text",
                "        if character not in _thousands_" + slug,
                "    )",
            ]
        )
    if parameters.strip_whitespace:
        lines.append("    text = text.strip()")
    lines.extend(
        [
            "    if accounting_negative:",
            "        if re.fullmatch(r'\\+?(?:\\d+(?:\\.\\d*)?|\\.\\d+)', text):",
            "            text = '-' + text.removeprefix('+')",
            "        else:",
            "            text = '(' + text + ')'",
        ]
    )
    lines.append("    return text")
    lines.append("")
    lines.append("for _column in " + _text_list(operation.target_columns) + ":")
    lines.append("    frame[_column] = frame[_column].map(_strip_" + slug + ")")
    return lines


def _impute_missing_lines(
    operation: TransformationOperation,
    slug: str,
) -> list[str]:
    parameters = operation.parameters
    assert isinstance(parameters, ImputeMissingParameters)
    column = operation.target_columns[0]
    before = "_before_" + slug
    lines = [
        before + " = frame[" + _text(column) + "].copy(deep=True)",
        "if not bool(" + before + ".isna().any()):",
        "    raise ValueError("
        + _text("imputation requires at least one missing value")
        + ")",
    ]
    if parameters.strategy is ImputeStrategy.CONSTANT:
        # A literal from the plan, emitted through repr like any other datum.
        lines.append("_fill_" + slug + " = " + repr(parameters.constant_value))
    elif parameters.strategy is ImputeStrategy.MODE:
        lines.extend(
            [
                "_modes_" + slug + " = " + before + ".mode(dropna=True)",
                "if _modes_" + slug + ".empty:",
                "    raise ValueError(" + _text("no mode exists for this column") + ")",
                "_fill_" + slug + " = _modes_" + slug + ".iloc[0]",
            ]
        )
    elif parameters.strategy is ImputeStrategy.MEAN:
        lines.append("_fill_" + slug + " = " + before + ".mean()")
    elif parameters.strategy is ImputeStrategy.MEDIAN:
        lines.append("_fill_" + slug + " = " + before + ".median()")
    else:  # pragma: no cover - the enum makes this unreachable
        raise ValueError("unsupported imputation strategy")
    lines.append(
        "frame[" + _text(column) + "] = " + before + ".fillna(_fill_" + slug + ")"
    )
    return lines


def _compute_column_lines(
    operation: TransformationOperation,
    slug: str,
) -> list[str]:
    del slug
    parameters = operation.parameters
    assert isinstance(parameters, ComputeColumnParameters)
    left = "frame[" + _text(parameters.left_column) + "]"
    right = "frame[" + _text(parameters.right_column) + "]"
    output = "frame[" + _text(parameters.output_column) + "]"
    symbols = {
        ComputeOperator.ADD: "+",
        ComputeOperator.SUBTRACT: "-",
        ComputeOperator.MULTIPLY: "*",
        ComputeOperator.DIVIDE: "/",
    }
    lines: list[str] = []
    if parameters.operator is ComputeOperator.DIVIDE:
        lines.extend(
            [
                "if bool(" + right + ".eq(0).fillna(False).any()):",
                "    raise ValueError(" + _text("division by zero is not allowed") + ")",
            ]
        )
    lines.append(output + " = " + left + " " + symbols[parameters.operator] + " " + right)
    return lines


_OPERATION_RENDERERS = {
    OperationType.TRIM_WHITESPACE: _trim_whitespace_lines,
    OperationType.NORMALIZE_MISSING_TOKENS: _normalize_missing_tokens_lines,
    OperationType.CAST_COLUMN: _cast_column_lines,
    OperationType.RENAME_COLUMN: _rename_column_lines,
    OperationType.DROP_DUPLICATE_ROWS: _drop_duplicate_rows_lines,
    OperationType.DEDUPLICATE_BY_KEYS: _deduplicate_by_keys_lines,
    OperationType.DROP_COLUMN: _drop_column_lines,
    OperationType.IMPUTE_MISSING: _impute_missing_lines,
    OperationType.NORMALIZE_NUMERIC_TEXT: _normalize_numeric_text_lines,
    OperationType.COMPUTE_COLUMN: _compute_column_lines,
}


assert set(_OPERATION_RENDERERS) == set(OperationType)


def _operation_block(
    operation: TransformationOperation,
    position: int,
) -> list[str]:
    """Render one operation, plus the bookkeeping the report prints."""

    slug = "op" + format(position, "03d")
    renderer = _OPERATION_RENDERERS[operation.operation_type]
    columns = ", ".join(operation.target_columns) or "the whole row"
    # Three plan-derived fragments reach this comment, and a column name is
    # upload-controlled text, so every one of them is sanitized. ``position`` is
    # an int we generate.
    lines = [
        "",
        "# "
        + str(position)
        + ". "
        + _comment_safe(operation.operation_id)
        + " - "
        + _comment_safe(operation.operation_type.value)
        + " on "
        + _comment_safe(columns),
        "_rows_before = len(frame)",
    ]
    lines.extend(renderer(operation, slug))
    lines.append(
        "steps.append(("
        + _text(operation.operation_id)
        + ", "
        + _text(operation.operation_type.value)
        + ", _rows_before, len(frame)))"
    )
    return lines


def _verify_syntax(rendered: str) -> None:
    """Check that the rendered text is syntactically valid Python.

    ``ast.parse`` builds a syntax tree and stops there. It produces no bytecode
    and it runs none of the rendered statements, so this stays well inside the
    rule that the product never runs generated source: the text is inspected,
    never invoked. Only the test suite ever runs a rendered script, and it does
    so in a separate interpreter.

    Defence in depth, not the primary control. This catches text that is broken;
    it cannot recognise text that is valid but says more than the plan does.
    ``_comment_safe`` is what prevents the latter. Raising here reaches
    ``build_artifact_set``, which already treats any serializer failure as a
    refusal of the whole bundle, so a script a reader could not trust is never
    shipped as one of seven verified artifacts.
    """

    try:
        ast.parse(rendered)
    except SyntaxError as error:  # pragma: no cover - guarded by _comment_safe
        raise ValueError(
            "rendered pipeline script is not syntactically valid Python"
        ) from error


def render_pipeline_script(plan: TransformationPlan) -> str:
    """Render a standalone runnable pandas script for one approved plan.

    Deterministic by construction: every value comes from the plan, nothing is
    read from the clock, the filesystem, the environment, or a set's iteration
    order. The same plan renders byte-identical text in any process.
    """

    body: list[str] = []
    for position, operation in enumerate(plan.operations, start=1):
        body.extend(_operation_block(operation, position))
    if not body:
        body = ["", "# The approved plan is empty; the table is copied unchanged."]

    indented = [_BODY_INDENT + line if line else "" for line in body]

    header = [
        "#!/usr/bin/env python3",
        _TRIPLE + "Standalone pandas pipeline rendered by DataChef.",
        "",
        "Rendered from a human-approved DataChef transformation plan. It applies the",
        "approved operations in plan order using nothing but pandas, so it keeps",
        "working without DataChef installed.",
        "",
        "Usage:",
        "    python <this script> <input path> <output path>",
        "",
        "A .parquet suffix on either path is read or written as Parquet; anything",
        "else is treated as CSV.",
        _TRIPLE,
        "",
        "# DataChef rendered pipeline",
        # Generated and hash-shaped today, but it is still plan-derived text in
        # a comment, so it takes the same route as everything else here.
        "# plan_id: " + _comment_safe(plan.plan_id),
        "# plan_version: " + str(plan.version),
        "# operation_count: " + str(len(plan.operations)),
        "",
        "import sys",
        "import re",
        "",
        "import pandas as pd",
        "",
        "PLAN_ID = " + _text(plan.plan_id),
        "PLAN_VERSION = " + str(plan.version),
        "",
        "",
        "def read_table(path):",
        "    if path.lower().endswith('.parquet'):",
        "        return pd.read_parquet(path)",
        "    return pd.read_csv(path, index_col=False)",
        "",
        "",
        "def write_table(frame, path):",
        "    if path.lower().endswith('.parquet'):",
        "        frame.to_parquet(path, index=False)",
        "        return",
        "    frame.to_csv(path, index=False, lineterminator='\\n', na_rep='')",
        "",
        "",
        "def apply_plan(frame):",
        "    " + _TRIPLE + "Apply the approved operations in plan order.",
        "",
        "    The frame keeps the index the operations leave behind, including the",
        "    gaps drop_duplicates creates, so this reproduces the approved run.",
        "    " + _TRIPLE,
        "",
        "    steps = []",
    ]

    footer = [
        "",
        "    return frame, steps",
        "",
        "",
        "def main(argv):",
        "    if len(argv) != 3:",
        "        print('usage: python <this script> <input path> <output path>')",
        "        return 2",
        "    frame = read_table(argv[1])",
        "    rows_before, columns_before = frame.shape",
        "    frame, steps = apply_plan(frame)",
        "    write_table(frame, argv[2])",
        "    print(f'plan {PLAN_ID} version {PLAN_VERSION}')",
        "    print(f'applied {len(steps)} operation(s)')",
        "    for operation_id, operation_type, before, after in steps:",
        "        print(f'  {operation_id} ({operation_type}): rows {before} -> {after}')",
        "    print(f'rows {rows_before} -> {frame.shape[0]}')",
        "    print(f'columns {columns_before} -> {frame.shape[1]}')",
        "    return 0",
        "",
        "",
        "if __name__ == '__main__':",
        "    raise SystemExit(main(sys.argv))",
    ]

    rendered = "\n".join([*header, *indented, *footer]) + "\n"
    _verify_syntax(rendered)
    return rendered


def render_pipeline_bytes(plan: TransformationPlan) -> bytes:
    """UTF-8 bytes of the rendered script, for the artifact bundle."""

    return render_pipeline_script(plan).encode("utf-8")


__all__ = [
    "PIPELINE_MEDIA_TYPE",
    "render_pipeline_bytes",
    "render_pipeline_script",
]
