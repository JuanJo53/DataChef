from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]

# Every test that needs CrewAI runs it in a subprocess. Importing crewai into
# this process would leave it in sys.modules and trip the committed purity
# assertion in test_application_controller.py, which is not ours to relax.
_CONTEXT_SETUP = """
import json, socket, sys
import pandas as pd

attempts = []
_original_connect = socket.socket.connect

def guarded_connect(sock, address):
    host = address[0] if isinstance(address, tuple) and address else None
    if host in {"127.0.0.1", "::1", "localhost"}:
        return _original_connect(sock, address)
    attempts.append(repr(address))
    raise AssertionError("External network access blocked: %r" % (address,))

socket.socket.connect = guarded_connect

import httpx

def blocked_sync_send(_client, request, *args, **kwargs):
    attempts.append(str(request.url))
    raise AssertionError("HTTP blocked: %s" % request.url)

async def blocked_async_send(_client, request, *args, **kwargs):
    attempts.append(str(request.url))
    raise AssertionError("Async HTTP blocked: %s" % request.url)

httpx.Client.send = blocked_sync_send
httpx.AsyncClient.send = blocked_async_send

from datachef.agents.tools import PlanDraft
from datachef.contracts import UserIntent
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.privacy import build_column_alias_map, build_planning_context

def make_context(row_loss=50.0):
    frame = pd.DataFrame({
        "order_id": [1, 2, 2],
        "region": ["N", "S", "S"],
        "amount": [1.0, 2.0, 2.0],
    })
    intent = UserIntent(
        intent_id="intent-agent", user_goal="Prepare for analysis.",
        selected_key_columns=("order_id",), acceptable_row_loss_pct=row_loss,
    )
    report = diagnose_raw_dataframe(frame, selected_key_columns=("order_id",))
    alias_map = build_column_alias_map(report, intent)
    return build_planning_context(report, intent, (), column_alias_map=alias_map)

from crewai import BaseLLM

class FakeLLM(BaseLLM):
    calls: int = 0
    def __init__(self):
        super().__init__(model="datachef-fake", provider="fake")
    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        self.calls += 1
        payload = {"summary": "Deduplicate orders by order_id."}
        if response_model is not None:
            return response_model.model_validate(payload)
        return json.dumps(payload)
"""


def _run_subprocess(body: str) -> subprocess.CompletedProcess:
    script = _CONTEXT_SETUP + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )


def _subprocess_env() -> dict:
    """Inherit the real environment, then strip credentials and isolate storage."""

    import os
    import tempfile

    env = dict(os.environ)
    env.pop("GOOGLE_API_KEY", None)
    env.pop("GEMINI_API_KEY", None)
    storage = tempfile.mkdtemp(prefix="datachef-slice-a-")
    env.update(
        {
            "CREWAI_STORAGE_DIR": storage,
            "LOCALAPPDATA": storage,
            "CREWAI_DISABLE_TELEMETRY": "true",
            "CREWAI_DISABLE_TRACKING": "true",
            "OTEL_SDK_DISABLED": "true",
            "DATACHEF_OFFLINE": "true",
        }
    )
    return env


def test_the_crew_builds_with_a_fake_llm_and_never_touches_the_network() -> None:
    result = _run_subprocess(
        """
        from crewai.events.event_bus import crewai_event_bus
        from datachef.agents.plan_crew import build_planning_crew

        context = make_context()
        draft = PlanDraft(context=context)
        try:
            crew = build_planning_crew(context, draft, FakeLLM())
            names = [tool.name for tool in crew.agents[0].tools]
        finally:
            crewai_event_bus.shutdown(wait=True)

        assert names == [
            "propose_trim_whitespace",
            "propose_normalize_missing_tokens",
            "propose_cast_column",
            "propose_rename_column",
            "propose_drop_duplicate_rows",
            "propose_deduplicate_by_keys",
            "inspect_profile",
            "estimate_current_plan",
            "discard_last_operation",
            "finalize_plan",
        ], names
        assert len(crew.tasks) == 1
        assert crew.tasks[0].guardrail is not None
        assert crew.tasks[0].guardrail_max_retries == 2
        assert attempts == [], attempts
        print("OK")
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_each_tool_exposes_an_args_schema_and_never_an_identity_field() -> None:
    result = _run_subprocess(
        """
        from crewai.events.event_bus import crewai_event_bus
        from datachef.agents.plan_crew import build_crew_tools

        draft = PlanDraft(context=make_context())
        try:
            tools = build_crew_tools(draft)
        finally:
            crewai_event_bus.shutdown(wait=True)

        schemas = {tool.name: tool.args_schema for tool in tools}
        assert "target_columns" in schemas["propose_trim_whitespace"].model_fields
        assert "keys" in schemas["propose_deduplicate_by_keys"].model_fields
        assert "target_type" in schemas["propose_cast_column"].model_fields
        assert "new_name" in schemas["propose_rename_column"].model_fields
        assert "tokens" in schemas["propose_normalize_missing_tokens"].model_fields
        assert "keep" in schemas["propose_drop_duplicate_rows"].model_fields
        assert "summary" in schemas["finalize_plan"].model_fields
        for name, schema in schemas.items():
            for forbidden in ("plan_id", "operation_id", "dataset_id", "dataset_fingerprint", "version"):
                assert forbidden not in schema.model_fields, (name, forbidden)
        assert attempts == [], attempts
        print("OK")
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_a_fake_llm_crew_run_never_opens_a_connection() -> None:
    result = _run_subprocess(
        """
        from datachef.agents.plan_crew import run_planning_crew

        context = make_context()
        try:
            run_planning_crew(context, FakeLLM())
        except Exception:
            pass
        assert attempts == [], attempts
        print("OK")
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_a_credential_alone_does_not_enable_the_live_path() -> None:
    """Offline is the default: a key present is not consent to call out.

    This test deliberately leaves DATACHEF_OFFLINE at its safe default. Flipping
    it to "false" here would make the crew genuinely reach for
    generativelanguage.googleapis.com, which the harness blocks but which this
    suite must never attempt.
    """

    result = _run_subprocess(
        """
        import os
        os.environ["GOOGLE_API_KEY"] = "fake-key-must-never-be-used"
        os.environ["GEMINI_MODEL"] = "gemini-3.1-flash-lite"

        from datachef.agents import AgentPlanner
        from datachef.agents.trace import AgentMode, AgentOutcome
        from datachef.planning import validate_plan

        context = make_context()
        planner = AgentPlanner()
        plan = planner.propose(context, attempt=1)

        assert attempts == [], attempts
        assert planner.trace.mode is AgentMode.OFFLINE
        assert (
            planner.trace.fallback_reason_code
            is AgentOutcome.AGENT_OFFLINE_USING_DETERMINISTIC_PLANNER
        )
        assert validate_plan(context, plan).valid is True
        rendered = planner.trace.model_dump_json()
        assert "fake-key-must-never-be-used" not in rendered
        print("OK")
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_the_guardrail_rejects_an_invalid_draft_with_codes_only() -> None:
    from datachef.agents.plan_crew import build_plan_guardrail
    from datachef.agents.tools import DeduplicateByKeysArgs, PlanDraft, apply_operation_args
    from datachef.contracts import UserIntent
    from datachef.diagnostics import diagnose_raw_dataframe
    from datachef.privacy import build_planning_context

    import pandas as pd

    context = build_planning_context(
        diagnose_raw_dataframe(
            pd.DataFrame({"order_id": [1, 2, 2], "amount": [1.0, 2.0, 2.0]}),
            selected_key_columns=("order_id",),
        ),
        UserIntent(
            intent_id="i",
            user_goal="g",
            selected_key_columns=("order_id",),
            acceptable_row_loss_pct=0.0,
        ),
        (),
    )
    draft = PlanDraft(context=context)

    accepted, _ = build_plan_guardrail(draft)("ok")
    assert accepted is True

    issue = next(
        item.issue_id
        for item in context.diagnostic_report.issues
        if item.kind.value == "DUPLICATE_KEYS"
    )
    apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )

    rejected, message = build_plan_guardrail(draft)("ok")

    assert rejected is False
    assert "ROW_LOSS_THRESHOLD" in message
    assert "Traceback" not in message


def test_importing_the_agent_package_never_loads_crewai() -> None:
    script = (
        "import sys\n"
        "import datachef.agents\n"
        "import datachef.agents.planner\n"
        "import datachef.agents.tools\n"
        "import datachef.agents.plan_crew\n"
        "assert 'crewai' not in sys.modules, sorted(n for n in sys.modules if 'crew' in n)\n"
        "assert 'litellm' not in sys.modules\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_the_application_import_path_still_never_loads_crewai() -> None:
    script = (
        "import sys\n"
        "import datachef.application\n"
        "import datachef.workflow.service\n"
        "forbidden = {n for n in sys.modules if n.split('.')[0] in {'crewai', 'litellm'}}\n"
        "assert not forbidden, sorted(forbidden)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_this_module_never_imports_crewai_into_the_pytest_process() -> None:
    assert "crewai" not in sys.modules


def test_litellm_remains_absent() -> None:
    assert importlib.util.find_spec("litellm") is None
