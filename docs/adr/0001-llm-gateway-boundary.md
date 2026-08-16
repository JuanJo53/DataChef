# ADR 0001: Application-owned LLM gateway boundary

- Status: Accepted for Phase 1 design
- Provider/model verification: Pending the separately approved Phase 0C call

## Context

The repository currently has three unrelated provider paths: the active
transformation module uses `google-genai`, ingestion and dashboard use
`langchain-google-genai` lazily, and `utils/llm_client.py` imports the removed
legacy `google.generativeai` SDK. The legacy helper has no repository caller and
would fail if imported. Existing ingestion fallback can include raw row samples,
and existing transformation code asks the LLM for Python that is passed to
`exec()`.

Offline package evidence shows that CrewAI 1.15.16 accepts a custom `BaseLLM`
object, while its optional Google extra targets a different google-genai major
version than the proven DataChef environment. LiteLLM is not installed.

## Decision

Phase 1 will introduce one application-owned LLM gateway:

1. Gemini is the hackathon provider, implemented with the direct `google-genai`
   SDK behind the gateway.
2. New application modules depend on the gateway interface, not SDK clients.
3. A CrewAI `BaseLLM` adapter delegates to the same gateway; CrewAI does not
   become a second provider configuration system.
4. Deterministic tests inject a fake gateway and make zero network calls.
5. The gateway accepts privacy-safe schema, aggregate diagnostics, sanitized
   examples when policy permits, and typed user intent. Raw rows are excluded by
   default, and `DATACHEF_SEND_ROW_SAMPLES=false` is the default.
6. The LLM returns typed declarative plans. Deterministic allow-listed Python
   operations validate and execute them against a deep copy of the original.
7. Arbitrary generated Python is not the execution contract. The existing path
   remains a migration target and will be disabled behind
   `DATACHEF_ENABLE_EXPERIMENTAL_CODE_EXECUTION=false` when Phase 1 integration
   is authorized.
8. `GOOGLE_API_KEY` is canonical. Secrets are read at the provider edge and are
   never placed in prompts, logs, exceptions, Streamlit state, fixtures, or
   downloadable artifacts.

## Typed value chain

The Phase 1 planning path uses these values at each boundary:

1. **Planner input:** a Pydantic `PlanningContext` containing a
   `SafeDatasetContext`, typed `UserIntent`, selected questions, supported
   operation metadata, and constraints. It contains no raw DataFrame, raw PII,
   credential, or provider-specific object.
2. **Provider request:** the gateway converts that context into Google SDK
   `Content`/text plus `GenerateContentConfig`. The model identifier is a
   separate string, and the config requests JSON with a Pydantic response schema.
3. **Provider response:** `google.genai.types.GenerateContentResponse`, including
   `.parsed`, `.text`, candidates, and usage metadata. This object stays inside
   the provider implementation.
4. **Gateway validated response:** a strict Pydantic result such as
   `TransformationPlan`, `PlanReview`, or `QuestionDiscoveryResult`. Invalid
   provider output becomes a sanitized gateway error, never an unvalidated dict.
5. **`BaseLLM.call()` return:** when CrewAI supplies `response_model`, the adapter
   returns an instance of that exact Pydantic type. Untyped string output is not
   used for planner/reviewer contracts.
6. **CrewAI task output:** `Task.output_pydantic` names the same model type, and
   the caller reads it from `CrewOutput.pydantic` after validating its presence.
7. **Workflow state:** typed Pydantic `WorkflowState` stores the validated domain
   model or its JSON-safe dump/reference. It stores no SDK response, secret, raw
   prompt, or DataFrame and can be reconstructed after a Streamlit rerun.

The Phase 0C live probe must exercise this complete chain—real CrewAI Agent and
Task, DataChef `BaseLLM` adapter, DataChef gateway, and `google-genai` client—with
exactly one provider invocation, zero automatic retries, synthetic schema only,
and no execution of returned content.

## Consequences

- Agent, Task, Crew, and Flow judging requirements can be met while data
  processing remains deterministic.
- Provider migration and fake testing have one seam.
- Existing ingestion/dashboard call sites need compatibility adapters in a
  later authorized phase; they are not changed in Phase 0B.
- `utils/llm_client.py` must be replaced or retired in Phase 1, not imported.

## Open questions for Phase 0C

- Which exact Gemini model identifier is available to the contributor account
  and works with google-genai 2.8.0 structured Pydantic output?
- Does the application-owned CrewAI `BaseLLM` adapter preserve typed output and
  usage metadata with that live model?
- What timeout and retry policy gives a reliable demonstration without duplicate
  Streamlit calls?
- Can `langchain-google-genai` be removed after legacy call sites migrate, or is
  a short compatibility bridge needed through the hackathon?
- Does the installed CrewAI native Google extra offer any benefit sufficient to
  justify its incompatible declared SDK range? It is not selected by default.
