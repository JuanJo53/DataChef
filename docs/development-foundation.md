# DataChef Phase 0B development foundation

## Reproducibility decision

Four common dependency strategies were considered:

| Strategy | Benefit | Cost or risk |
|---|---|---|
| Broad ranges only | Easy upgrades and library reuse | Teammates can resolve materially different stacks during the hackathon |
| Exact pins in the runtime manifest | Very simple and repeatable | Hides the difference between supported ranges and the event checkpoint |
| Constraints plus direct requirements | Keeps application intent readable while automatically selecting the proven versions | Requires maintaining one small extra file |
| Full environment freeze | Reproduces every installed package | Treats hundreds of transitive packages as application-owned and creates noisy, platform-sensitive diffs |

DataChef uses direct requirements plus `constraints-hackathon.txt`. The
constraints file pins only deliberate runtime dependencies and pytest. It is
referenced from `requirements.txt`, so the normal install command cannot forget
it. This is the smallest approach that prevents Gemini/CrewAI/Pydantic drift
during the four-day Windows hackathon without copying the full package
inventory into source control.

This reproduces the critical direct application stack, not necessarily every
transitive package selected by pip. The complete Phase 0A inventory remains
local baseline evidence rather than an application manifest.

The proven direct versions are:

- CrewAI 1.15.16
- google-genai 2.8.0
- langchain-google-genai 4.3.4
- Pydantic 2.11.10
- Streamlit 1.61.1
- Pandas 3.0.5
- Plotly 6.9.0
- PyArrow 24.0.0
- PySpark 4.2.0
- python-dotenv 1.2.2

PyArrow is direct because CSV/JSON/Parquet input and Parquet output are DataChef
features. PySpark remains installed for backward compatibility but is not part
of the MVP execution path. Pytest is development-only.

## Test classification

| Test | Classification | Reason |
|---|---|---|
| `tests/` | Safe offline pytest suite | Synthetic inputs, temporary outputs, provider traffic blocked |
| `crew/ingestion_agent/test_ingestion.py` | Safe deterministic script | Rule-based Pandas checks, no credential required |
| `crew/dashboard_agent/test_dashboard.py` | Safe deterministic script | Rule-based dashboard checks, no credential required |
| `crew/transformation_agent/test_transformation.py` | External and mutating; excluded | Loads `.env`, can call Gemini, executes generated Python, writes a pipeline |

`pytest.ini` restricts normal collection to `tests/` and also explicitly ignores
the unsafe transformation script. Collection must be inspected before running
the suite in a new environment.

## Installed API evidence

Offline inspection of the installed packages established:

- `google.genai.Client` accepts an `api_key` keyword.
- `client.models.generate_content` receives `model`, `contents`, and optional
  `config` arguments and returns `GenerateContentResponse`.
- `GenerateContentConfig` supports `response_schema` and
  `response_json_schema`; responses expose `.text`, `.parsed`, candidates, and
  usage metadata.
- CrewAI Agent, Task, and Crew are Pydantic models. `Agent.llm` accepts a
  `BaseLLM`; `Task.output_pydantic` accepts a Pydantic model type.
- The installed `BaseLLM` has one abstract method: `call(...)`. A deterministic
  fake implementing it returned a typed Crew output offline.
- CrewAI Flow accepts Pydantic initial state and provides `@start`, `@router`,
  `@listen`, `kickoff`, and `kickoff_async`. Both discovery and skip routes ran
  offline, and state serialized and reconstructed with Pydantic.
- CrewAI import and execution create local storage. Offline tests must isolate
  both `CREWAI_STORAGE_DIR` and `LOCALAPPDATA`, disable telemetry, and cleanly
  shut down the event bus.
- Base CrewAI was installed without LiteLLM. Its optional `google-genai` extra
  declares a 1.x SDK constraint, which does not match DataChef's proven direct
  google-genai 2.8.0 stack. No extra is added until a compatibility decision is
  supported by a live spike.
- `langchain-google-genai` uses the direct `google-genai` SDK and exposes model,
  API-key, timeout, retry, response MIME type, and response schema fields.
- Pydantic 2.11 provides `model_validate`, `model_validate_json`, `model_dump`,
  `model_dump_json`, `model_copy`, and `model_json_schema`.

## Phase 1 Pydantic contract guidance

Contracts should inherit from strict Pydantic models with `extra="forbid"`, use
typed enums for finite states, UTC timestamps, explicit schema/version fields,
and `Field(default_factory=...)` for every mutable collection. They must remain
JSON-serializable and must not contain credentials or raw DataFrames.

| Contract | Recommended representation |
|---|---|
| `DiagnosticReport` | Dataset/report IDs, schema/profile summary, immutable list of typed issues, safe metrics, profiler version |
| `UserIntent` | Target use/audience, authored and selected questions, requirements, required columns, user constraints |
| `TransformationOperation` | Operation ID, allow-listed enum, typed parameter model, issue/requirement references, rationale, risk, pre/postconditions |
| `TransformationPlan` | Plan/version/context hash, ordered operations, explanations, assumptions, planning constraints; no Python source |
| `PlanReview` | APPROVE/REVISE/BLOCK enum, typed findings, revision guidance, reviewed plan/version |
| `ExecutionResult` | Input/output fingerprints and artifact references, applied/skipped operation IDs, row/column metrics, sanitized errors; DataFrame stays outside serialization |
| `QAReport` | Required and warning invariants, pass/fail summary, operation postconditions, diagnostic-comparison reference |
| `DashboardContext` | Dataset/result references, user questions, answerability, semantic roles, warnings, KPI/chart suggestions; pass the live DataFrame separately to the legacy dashboard adapter |

## Test-data policy

- `tests/fixtures/`: tracked, small, synthetic, deterministic, sanitized.
- `data/local/`: ignored private, unknown, large, or colleague-provided data.
- `data/baseline/`: ignored local diagnostic and setup evidence.

Shared application prompts remain trackable. Only `prompts/local/` is reserved
for ignored experiments.
