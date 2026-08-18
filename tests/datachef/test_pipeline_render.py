"""The rendered pipeline script must be deterministic and must not drift.

Drift is closed here, in the test suite, and nowhere else. The product renders
the script and hands over its bytes; only these tests ever execute one, and they
do it in a subprocess so a rendered script can never share this interpreter with
the application.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from datachef.application.pipeline_render import (
    PIPELINE_MEDIA_TYPE,
    render_pipeline_bytes,
    render_pipeline_script,
)
from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    DeduplicateByKeysParameters,
    DropDuplicateRowsParameters,
    KeepPolicy,
    NormalizeMissingTokensParameters,
    OperationType,
    RenameColumnParameters,
    RiskLevel,
    TransformationOperation,
    TrimWhitespaceParameters,
)
from datachef.diagnostics import dataframe_fingerprint
from datachef.planning.plan import create_transformation_plan
from datachef.transform.runner import run_allowlisted_plan


REPO_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT = "a" * 64


def _operation(
    operation_id: str,
    operation_type: OperationType,
    target_columns: tuple[str, ...],
    parameters: object,
) -> TransformationOperation:
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=operation_type,
        target_columns=target_columns,
        parameters=parameters,
        diagnostic_issue_ids=("issue-1",),
        rationale="Reported by the diagnosis.",
        expected_effect="Normalizes the affected cells.",
        risk=RiskLevel.LOW,
        requires_human_approval=True,
    )


def _plan(*operations: TransformationOperation):
    return create_transformation_plan(
        dataset_id="dataset-demo",
        dataset_fingerprint=FINGERPRINT,
        version=1,
        operations=tuple(operations),
        summary="Demo plan for the rendered pipeline.",
    )


def _demo_frame() -> pd.DataFrame:
    """Twelve rows carrying every defect the six operations address."""

    return pd.DataFrame(
        {
            "order_id": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
            "category": [
                " alpha ",
                "beta",
                "beta",
                " gamma",
                "delta",
                "epsilon",
                "zeta",
                "eta",
                "theta",
                "iota",
                "kappa",
                "kappa",
            ],
            "measure_text": [
                "10",
                "20",
                "20",
                "30",
                "bad",
                "50",
                "60",
                "70",
                "80",
                "90",
                "100",
                "100",
            ],
            "observed_on": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
                "not-a-date",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
                "2026-01-09",
                "2026-01-10",
                "2026-01-11",
                "2026-01-11",
            ],
            "flag": [
                "yes",
                "no",
                "no",
                "TRUE",
                "weird",
                "false",
                "1",
                "0",
                "yes",
                "no",
                "yes",
                "yes",
            ],
            "note": [
                "NA",
                "missing",
                "missing",
                "ok",
                None,
                "NA",
                "ok",
                "ok",
                "ok",
                "ok",
                "ok",
                "ok",
            ],
        }
    )


_TRIM = _operation(
    "op-001-trim_whitespace",
    OperationType.TRIM_WHITESPACE,
    ("category",),
    TrimWhitespaceParameters(),
)
_NORMALIZE = _operation(
    "op-001-normalize_missing_tokens",
    OperationType.NORMALIZE_MISSING_TOKENS,
    ("note",),
    NormalizeMissingTokensParameters(tokens=("NA", "missing")),
)
_CAST_NUMERIC = _operation(
    "op-001-cast_column",
    OperationType.CAST_COLUMN,
    ("measure_text",),
    CastColumnParameters(
        target_type=CastTarget.NUMERIC,
        errors=CastErrorPolicy.COERCE,
    ),
)
_CAST_STRING = _operation(
    "op-001-cast_column",
    OperationType.CAST_COLUMN,
    ("category",),
    CastColumnParameters(target_type=CastTarget.STRING),
)
_CAST_BOOLEAN = _operation(
    "op-001-cast_column",
    OperationType.CAST_COLUMN,
    ("flag",),
    CastColumnParameters(
        target_type=CastTarget.BOOLEAN,
        errors=CastErrorPolicy.COERCE,
    ),
)
_CAST_DATETIME = _operation(
    "op-001-cast_column",
    OperationType.CAST_COLUMN,
    ("observed_on",),
    CastColumnParameters(
        target_type=CastTarget.DATETIME,
        errors=CastErrorPolicy.COERCE,
    ),
)
_RENAME = _operation(
    "op-001-rename_column",
    OperationType.RENAME_COLUMN,
    ("measure_text",),
    RenameColumnParameters(new_name="measure"),
)
_DROP_DUPLICATES = _operation(
    "op-001-drop_duplicate_rows",
    OperationType.DROP_DUPLICATE_ROWS,
    (),
    DropDuplicateRowsParameters(keep=KeepPolicy.FIRST),
)
_DEDUPLICATE = _operation(
    "op-001-deduplicate_by_keys",
    OperationType.DEDUPLICATE_BY_KEYS,
    ("order_id",),
    DeduplicateByKeysParameters(keys=("order_id",), keep=KeepPolicy.FIRST),
)

# One case per operation type, plus the cast variants, plus a multi-operation
# plan that runs them in sequence the way an agent-proposed plan would.
EQUIVALENCE_CASES: tuple[tuple[str, tuple[TransformationOperation, ...]], ...] = (
    ("TRIM_WHITESPACE", (_TRIM,)),
    ("NORMALIZE_MISSING_TOKENS", (_NORMALIZE,)),
    ("CAST_COLUMN_numeric", (_CAST_NUMERIC,)),
    ("CAST_COLUMN_string", (_CAST_STRING,)),
    ("CAST_COLUMN_boolean", (_CAST_BOOLEAN,)),
    ("CAST_COLUMN_datetime", (_CAST_DATETIME,)),
    ("RENAME_COLUMN", (_RENAME,)),
    ("DROP_DUPLICATE_ROWS", (_DROP_DUPLICATES,)),
    ("DEDUPLICATE_BY_KEYS", (_DEDUPLICATE,)),
)


def _multi_operation_plan():
    """A six-operation plan, renumbered so operation ids stay canonical."""

    return _plan(
        _operation(
            "op-001-trim_whitespace",
            OperationType.TRIM_WHITESPACE,
            ("category", "note"),
            TrimWhitespaceParameters(),
        ),
        _operation(
            "op-002-normalize_missing_tokens",
            OperationType.NORMALIZE_MISSING_TOKENS,
            ("note",),
            NormalizeMissingTokensParameters(tokens=("NA", "missing")),
        ),
        _operation(
            "op-003-cast_column",
            OperationType.CAST_COLUMN,
            ("measure_text",),
            CastColumnParameters(
                target_type=CastTarget.NUMERIC,
                errors=CastErrorPolicy.COERCE,
            ),
        ),
        _operation(
            "op-004-rename_column",
            OperationType.RENAME_COLUMN,
            ("measure_text",),
            RenameColumnParameters(new_name="measure"),
        ),
        _operation(
            "op-005-drop_duplicate_rows",
            OperationType.DROP_DUPLICATE_ROWS,
            (),
            DropDuplicateRowsParameters(keep=KeepPolicy.FIRST),
        ),
        _operation(
            "op-006-deduplicate_by_keys",
            OperationType.DEDUPLICATE_BY_KEYS,
            ("order_id",),
            DeduplicateByKeysParameters(keys=("order_id",), keep=KeepPolicy.FIRST),
        ),
    )


def _environment(seed: str = "0") -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "PYTHONDONTWRITEBYTECODE": "1",
        "DATACHEF_OFFLINE": "true",
    }


def _run_python(code: str, seed: str = "0", timeout: int = 90):
    return subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=REPO_ROOT,
        env=_environment(seed),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _fingerprint_in_subprocess(
    script_path: Path,
    frame_path: Path,
    seed: str = "0",
) -> subprocess.CompletedProcess:
    """Run the rendered script's own ``apply_plan`` in a separate interpreter.

    The frame crosses the boundary as a pickle rather than a CSV so dtypes and
    the index survive intact, and the fingerprint is printed instead of the
    frame so the comparison is exact.
    """

    code = textwrap.dedent(
        f"""
        import runpy
        import pandas as pd
        from datachef.diagnostics import dataframe_fingerprint

        namespace = runpy.run_path({str(script_path)!r})
        frame = pd.read_pickle({str(frame_path)!r})
        result, steps = namespace["apply_plan"](frame)
        print(dataframe_fingerprint(result))
        """
    )
    return _run_python(code, seed=seed)


def test_rendered_script_is_valid_python_and_never_imports_datachef() -> None:
    text = render_pipeline_script(_multi_operation_plan())

    ast.parse(text)
    assert "import datachef" not in text
    assert "from datachef" not in text
    assert text.endswith("\n")
    assert PIPELINE_MEDIA_TYPE == "text/x-python"


def test_rendered_script_carries_the_plan_identity_as_a_comment_header() -> None:
    plan = _multi_operation_plan()

    text = render_pipeline_script(plan)

    assert f"# plan_id: {plan.plan_id}" in text
    assert "# plan_version: 1" in text
    assert "# operation_count: 6" in text


def test_rendered_script_carries_no_nondeterministic_source() -> None:
    """No clock, no uuid, no pid, no randomness, no absolute path."""

    text = render_pipeline_script(_multi_operation_plan())

    for forbidden in (
        "datetime",
        "time.",
        "uuid",
        "random",
        "os.getpid",
        "tempfile",
        str(REPO_ROOT),
    ):
        assert forbidden not in text
    # A set literal's iteration order follows the hash seed, so the renderer
    # must emit ordered tuples and build any set inside the script.
    assert "set((" in text


def test_render_is_byte_identical_across_processes_with_different_hash_seeds() -> None:
    code = textwrap.dedent(
        """
        from hashlib import sha256

        from tests.datachef.test_pipeline_render import _multi_operation_plan
        from datachef.application.pipeline_render import render_pipeline_bytes

        print(sha256(render_pipeline_bytes(_multi_operation_plan())).hexdigest())
        """
    )
    digests = []
    for seed in ("1", "2", "4294967295"):
        completed = _run_python(code, seed=seed)
        assert completed.returncode == 0, completed.stderr
        digests.append(completed.stdout.strip())

    assert digests[0] == digests[1]
    assert digests[0]


def test_render_is_stable_when_called_twice_in_this_process() -> None:
    plan = _multi_operation_plan()

    assert render_pipeline_bytes(plan) == render_pipeline_bytes(plan)


@pytest.mark.parametrize(
    ("name", "operations"),
    EQUIVALENCE_CASES,
    ids=[name for name, _ in EQUIVALENCE_CASES],
)
def test_emitted_script_matches_the_executor_per_operation_type(
    name: str,
    operations: tuple[TransformationOperation, ...],
    tmp_path: Path,
) -> None:
    del name
    plan = _plan(*operations)
    expected = run_allowlisted_plan(_demo_frame(), plan)
    assert expected.success and expected.dataframe is not None

    script_path = tmp_path / "rendered.py"
    script_path.write_bytes(render_pipeline_bytes(plan))
    frame_path = tmp_path / "frame.pkl"
    _demo_frame().to_pickle(frame_path)

    completed = _fingerprint_in_subprocess(script_path, frame_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == dataframe_fingerprint(expected.dataframe)


def test_emitted_script_matches_the_executor_for_a_multi_operation_plan(
    tmp_path: Path,
) -> None:
    plan = _multi_operation_plan()
    expected = run_allowlisted_plan(_demo_frame(), plan)
    assert expected.success and expected.dataframe is not None

    script_path = tmp_path / "rendered.py"
    script_path.write_bytes(render_pipeline_bytes(plan))
    frame_path = tmp_path / "frame.pkl"
    _demo_frame().to_pickle(frame_path)

    completed = _fingerprint_in_subprocess(script_path, frame_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == dataframe_fingerprint(expected.dataframe)


def test_emitted_script_does_not_reset_the_gapped_index_the_executor_leaves(
    tmp_path: Path,
) -> None:
    """``drop_duplicates`` gaps the index and nothing in transform/ resets it."""

    plan = _plan(_DEDUPLICATE)
    expected = run_allowlisted_plan(_demo_frame(), plan)
    assert expected.dataframe is not None
    assert list(expected.dataframe.index) != list(range(len(expected.dataframe)))

    script_path = tmp_path / "rendered.py"
    script_path.write_bytes(render_pipeline_bytes(plan))
    frame_path = tmp_path / "frame.pkl"
    _demo_frame().to_pickle(frame_path)

    code = textwrap.dedent(
        f"""
        import runpy
        import pandas as pd

        namespace = runpy.run_path({str(script_path)!r})
        frame = pd.read_pickle({str(frame_path)!r})
        result, steps = namespace["apply_plan"](frame)
        print(list(result.index))
        """
    )
    completed = _run_python(code)

    assert completed.returncode == 0, completed.stderr
    assert "reset_index" not in render_pipeline_script(plan)
    assert completed.stdout.strip() == str(list(expected.dataframe.index))


def test_emitted_script_reproduces_the_runner_null_key_guard(tmp_path: Path) -> None:
    """The guard lives in the runner, before the drop. The script must match."""

    frame = _demo_frame()
    frame.loc[3, "order_id"] = None
    plan = _plan(_DEDUPLICATE)

    # The committed runner refuses this frame rather than deduplicating it.
    expected = run_allowlisted_plan(frame, plan)
    assert not expected.success
    assert expected.error_code == "PLAN_EXECUTION_ABORTED"

    script_path = tmp_path / "rendered.py"
    script_path.write_bytes(render_pipeline_bytes(plan))
    frame_path = tmp_path / "frame.pkl"
    frame.to_pickle(frame_path)

    completed = _fingerprint_in_subprocess(script_path, frame_path)

    assert completed.returncode != 0
    assert "null keys are unsafe for key deduplication" in completed.stderr


def test_rendered_script_runs_end_to_end_as_a_command_line_program(
    tmp_path: Path,
) -> None:
    plan = _multi_operation_plan()
    script_path = tmp_path / "rendered.py"
    script_path.write_bytes(render_pipeline_bytes(plan))
    source_path = tmp_path / "input.csv"
    _demo_frame().to_csv(source_path, index=False)
    output_path = tmp_path / "output.csv"

    completed = subprocess.run(
        [sys.executable, "-B", str(script_path), str(source_path), str(output_path)],
        cwd=REPO_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    assert f"plan {plan.plan_id} version 1" in completed.stdout
    assert "applied 6 operation(s)" in completed.stdout
    assert "rows 12 -> 10" in completed.stdout
    reloaded = pd.read_csv(output_path, index_col=False)
    assert "measure" in reloaded.columns
    assert len(reloaded) == 10


def test_rendered_script_reports_a_usage_line_without_two_paths(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "rendered.py"
    script_path.write_bytes(render_pipeline_bytes(_multi_operation_plan()))

    completed = subprocess.run(
        [sys.executable, "-B", str(script_path)],
        cwd=REPO_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )

    assert completed.returncode == 2
    assert "usage:" in completed.stdout


def test_an_empty_plan_renders_a_runnable_copy_through(tmp_path: Path) -> None:
    plan = _plan()
    text = render_pipeline_script(plan)

    ast.parse(text)
    assert "# operation_count: 0" in text

    script_path = tmp_path / "rendered.py"
    script_path.write_bytes(render_pipeline_bytes(plan))
    frame_path = tmp_path / "frame.pkl"
    _demo_frame().to_pickle(frame_path)

    completed = _fingerprint_in_subprocess(script_path, frame_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == dataframe_fingerprint(_demo_frame())


def test_cast_error_policy_is_read_from_the_plan_and_never_defaulted() -> None:
    """RAISE is the contract default; the rule-based planner passes COERCE."""

    coerce_text = render_pipeline_script(_plan(_CAST_NUMERIC))
    raising = _operation(
        "op-001-cast_column",
        OperationType.CAST_COLUMN,
        ("measure_text",),
        CastColumnParameters(
            target_type=CastTarget.NUMERIC,
            errors=CastErrorPolicy.RAISE,
        ),
    )
    raise_text = render_pipeline_script(_plan(raising))

    assert "errors='coerce'" in coerce_text
    assert "errors='raise'" in raise_text
    assert "errors='coerce'" not in raise_text


_EXECUTION_MODULES = frozenset({"runpy", "subprocess", "importlib"})
_EXECUTION_BUILTINS = frozenset({"exec", "eval", "compile"})
# ``re.compile`` is a regex, not an execution primitive, so only the bare
# builtin name is treated as one when it appears as an attribute call.
_EXECUTION_ATTRIBUTES = frozenset({"exec", "eval"})


def _execution_offenders(path: Path) -> list[str]:
    """Real imports and calls only.

    Deliberately an AST walk rather than a substring scan: this module's own
    docstring discusses running the script in a subprocess, and prose about a
    risk must not read as the risk itself.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _EXECUTION_MODULES:
                    found.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _EXECUTION_MODULES:
                found.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id in _EXECUTION_BUILTINS:
                found.append(f"{callee.id}(...)")
            elif isinstance(callee, ast.Attribute) and callee.attr in _EXECUTION_ATTRIBUTES:
                found.append(f".{callee.attr}(...)")
    return found


def test_the_product_never_executes_a_rendered_script() -> None:
    """Only the test suite may run rendered text. Guard the product tree."""

    offenders: list[tuple[str, str]] = []
    for tree_root in ("datachef", "ui"):
        for path in (REPO_ROOT / tree_root).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for offender in _execution_offenders(path):
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), offender))

    assert offenders == []


# ---------------------------------------------------------------------------
# Hostile column names.
#
# A column name comes from an uploaded file, so it is untrusted text, and it
# reaches the rendered script twice: as a Python literal on the data path, and
# as prose inside an emitted comment. The literal was always escaped through
# repr. The comment was not, and a newline in a column name used to end the
# comment early -- at best a script that would not parse, at worst a script that
# parsed and carried a statement the plan never contained and no human approved.
# These cases exist because this suite previously had no hostile input at all.
# ---------------------------------------------------------------------------

# Identifier-shaped, so that if a payload ever becomes code again there is an
# ast.Name to find rather than a string that merely looks suspicious.
SENTINEL = "injected_sentinel_call"
INDENTED_PAYLOAD = "amt\n    " + SENTINEL + "()\n    # "

HOSTILE_COLUMN_NAMES: tuple[tuple[str, str], ...] = (
    ("newline", "amt\nlines"),
    ("carriage_return", "amt\rlines"),
    ("crlf", "amt\r\nlines"),
    ("indented_injection", INDENTED_PAYLOAD),
    ("single_quote", "we'ird"),
    ("double_quote", 'say"hi'),
    ("backslash", "back\\slash"),
    ("triple_quote", 'end"""start'),
    ("hash", "a#b"),
    ("non_ascii", "kolonn_é中"),
    ("line_separator", "amt lines"),
    ("paragraph_separator", "amt lines"),
    ("nul", "a\x00b"),
    ("very_long", "z" * 5000),
)


def _hostile_frame(column: str) -> pd.DataFrame:
    return pd.DataFrame({column: ["  x  ", "  x  ", " y", "z"], "keep": [1, 2, 3, 4]})


def _hostile_plan(column: str):
    return _plan(
        _operation(
            "op-001-trim_whitespace",
            OperationType.TRIM_WHITESPACE,
            (column,),
            TrimWhitespaceParameters(),
        )
    )


def _code_shape(text: str) -> tuple[str, ...]:
    """The syntactic shape of a script, ignoring every literal value.

    Two renders of the same plan shape must agree here. An injected statement
    adds nodes, so a shape mismatch is the signal that text became code.
    """

    return tuple(type(node).__name__ for node in ast.walk(ast.parse(text)))


def _identifiers(text: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _comment_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.lstrip().startswith("#")]


@pytest.mark.parametrize(
    ("name", "column"),
    HOSTILE_COLUMN_NAMES,
    ids=[name for name, _ in HOSTILE_COLUMN_NAMES],
)
def test_a_hostile_column_name_still_renders_valid_python(
    name: str,
    column: str,
) -> None:
    del name
    text = render_pipeline_script(_hostile_plan(column))

    ast.parse(text)
    # The comment the name lands in stays exactly one line.
    assert len(_comment_lines(text)) == len(
        _comment_lines(render_pipeline_script(_hostile_plan("benign")))
    )


@pytest.mark.parametrize(
    ("name", "column"),
    HOSTILE_COLUMN_NAMES,
    ids=[name for name, _ in HOSTILE_COLUMN_NAMES],
)
def test_a_hostile_column_name_never_becomes_code(name: str, column: str) -> None:
    """The payload may appear as data. It may never appear as a statement.

    The exact column name has to survive on the data path or the script would
    act on the wrong column, so the honest assertion is not "these bytes are
    absent" but "these bytes are never code". The shape comparison is what
    proves it: an injected statement adds nodes the benign render does not have.
    """

    del name
    text = render_pipeline_script(_hostile_plan(column))
    benign = render_pipeline_script(_hostile_plan("benign"))

    assert _code_shape(text) == _code_shape(benign)
    assert SENTINEL not in _identifiers(text)
    assert not any(item.startswith("injected") for item in _identifiers(text))
    # No emitted comment may carry a line break of any kind.
    for line in _comment_lines(text):
        for terminator in ("\n", "\r", " ", " "):
            assert terminator not in line


@pytest.mark.parametrize(
    ("name", "column"),
    HOSTILE_COLUMN_NAMES,
    ids=[name for name, _ in HOSTILE_COLUMN_NAMES],
)
def test_a_hostile_column_name_still_matches_the_executor(
    name: str,
    column: str,
    tmp_path: Path,
) -> None:
    """Sanitizing the comment must not disturb the data path."""

    del name
    plan = _hostile_plan(column)
    expected = run_allowlisted_plan(_hostile_frame(column), plan)
    assert expected.success and expected.dataframe is not None

    script_path = tmp_path / "rendered.py"
    script_path.write_bytes(render_pipeline_bytes(plan))
    frame_path = tmp_path / "frame.pkl"
    _hostile_frame(column).to_pickle(frame_path)

    completed = _fingerprint_in_subprocess(script_path, frame_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == dataframe_fingerprint(expected.dataframe)


def test_the_comment_sanitizer_keeps_a_safe_set_and_replaces_the_rest() -> None:
    from datachef.application.pipeline_render import _comment_safe

    assert _comment_safe("order_id (2026-01) [x]") == "order_id (2026-01) [x]"
    # Printable non-ASCII survives, so an international label stays readable.
    assert _comment_safe("kolonn_é中") == "kolonn_é中"
    for hostile in ("\n", "\r", "\x00", "\x1f", "\x7f", "\x85", " ",
                    " ", "‮", "​", "\t"):
        assert _comment_safe("a" + hostile + "b") == "a?b", repr(hostile)
    # Every C0 and C1 control goes by construction, not by enumeration.
    controls = "".join(
        chr(code) for code in list(range(0x00, 0x20)) + list(range(0x7F, 0xA0))
    )
    assert set(_comment_safe(controls)) == {"?"}
    assert len(_comment_safe("z" * 5000)) < 200


def test_the_renderer_refuses_to_return_text_that_does_not_parse(monkeypatch) -> None:
    """Fix 2, exercised by disabling Fix 1: the guard is live, not decorative."""

    import datachef.application.pipeline_render as module

    monkeypatch.setattr(module, "_comment_safe", lambda value: value)

    with pytest.raises(ValueError, match="not syntactically valid Python"):
        module.render_pipeline_script(_hostile_plan("amt\nlines"))


def _payload_controller(column: str):
    """Drive the real product path to QA PASS with a crafted column name."""

    from datachef.application import (
        DataChefController,
        JsonRecordsParserOptions,
        RequestedTransformation,
        UploadFormat,
        UploadRequest,
    )
    from datachef.contracts import (
        CastColumnParameters,
        CastTarget,
        DownstreamUse,
        HumanDecision,
        UserIntent,
    )

    records = json.dumps(
        [
            {column: "10", "order_id": 1},
            {column: "20", "order_id": 2},
            {column: "20", "order_id": 2},
            {column: "30", "order_id": 3},
        ]
    ).encode("utf-8")
    controller = DataChefController(
        clock=lambda: datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
    )
    assert controller.load_upload(
        UploadRequest(
            content=records,
            declared_suffix=".json",
            format=UploadFormat.JSON_RECORDS,
            parser_options=JsonRecordsParserOptions(),
        )
    ).changed
    assert controller.diagnose().changed
    controller.submit_intent(
        UserIntent(
            intent_id="intent-payload",
            user_goal="Prepare.",
            downstream_use=DownstreamUse.ANALYSIS,
            selected_key_columns=("order_id",),
            acceptable_row_loss_pct=90,
        ),
        (
            RequestedTransformation(
                request_id="request-cast-payload",
                operation_type=OperationType.CAST_COLUMN,
                target_columns=(column,),
                parameters=CastColumnParameters(target_type=CastTarget.NUMERIC),
            ),
        ),
    )
    assert controller.prepare_plan(command_id="plan").code == "PLAN_AWAITING_APPROVAL"
    assert controller.record_human_decision(
        HumanDecision.APPROVE, command_id="approve"
    ).changed
    assert controller.execute_current_plan(command_id="execute").changed
    return controller


def test_unparseable_render_refuses_the_whole_bundle(monkeypatch) -> None:
    """A broken script is never shipped as one of seven verified artifacts."""

    from datachef.application import ArtifactFailure, ArtifactSet
    import datachef.application.pipeline_render as module

    controller = _payload_controller("amt\nlines")
    assert isinstance(controller.build_artifacts(), ArtifactSet)

    monkeypatch.setattr(module, "_comment_safe", lambda value: value)
    refused = controller.build_artifacts()

    assert isinstance(refused, ArtifactFailure)
    assert refused.code.value == "SERIALIZATION_FAILURE"


def test_a_payload_column_name_reaches_the_bundle_without_becoming_code() -> None:
    """The end-to-end reproduction of the original defect, now clean.

    Upload, diagnose, objective, plan, approval, execution and QA PASS, with a
    column name carrying an indented statement. The shipped artifact must parse,
    and the payload must be data inside it, never a statement.
    """

    from datachef.application import ArtifactSet
    from datachef.contracts import WorkflowStage

    controller = _payload_controller(INDENTED_PAYLOAD)
    runtime = controller.session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    # The approved plan really does target the crafted column.
    assert any(
        any("\n" in target for target in operation.target_columns)
        for operation in runtime.state.transformation_plan.operations
    )

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    script = bundle.pipeline_script.content.decode("utf-8")
    ast.parse(script)
    assert SENTINEL not in _identifiers(script)
    for line in _comment_lines(script):
        assert "\n" not in line and "\r" not in line
    # Still seven artifacts, still schema version 2.
    assert len(bundle.artifacts()) == 7
    assert len(bundle.downloads()) == 6
    manifest = json.loads(bundle.manifest.content.decode("utf-8"))
    assert manifest["artifact_schema_version"] == 2
    assert len(manifest["artifacts"]) == 6
