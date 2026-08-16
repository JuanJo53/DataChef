from __future__ import annotations

from importlib import metadata
import importlib.util
import inspect
import json
from pathlib import Path
import socket


def test_proven_package_versions_are_installed() -> None:
    expected_versions = {
        "crewai": "1.15.16",
        "google-genai": "2.8.0",
        "langchain-google-genai": "4.3.4",
        "pydantic": "2.11.10",
        "pandas": "3.0.5",
        "pyarrow": "24.0.0",
    }

    installed = {
        distribution: metadata.version(distribution)
        for distribution in expected_versions
    }

    assert installed == expected_versions


def test_google_sdk_exposes_structured_output_contract() -> None:
    from google import genai
    from google.genai import types

    client_parameters = inspect.signature(genai.Client).parameters
    generation_parameters = inspect.signature(
        genai.models.Models.generate_content
    ).parameters

    assert "api_key" in client_parameters
    assert {"model", "contents", "config"} <= generation_parameters.keys()
    assert "response_schema" in types.GenerateContentConfig.model_fields
    assert "response_json_schema" in types.GenerateContentConfig.model_fields
    assert "parsed" in types.GenerateContentResponse.model_fields
    assert isinstance(types.GenerateContentResponse.text, property)


def test_legacy_sdk_and_litellm_are_not_implicit_dependencies() -> None:
    assert importlib.util.find_spec("google.generativeai") is None
    assert importlib.util.find_spec("litellm") is None


def test_crewai_fake_llm_typed_task_and_flow_are_offline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_path = tmp_path / "crewai-storage"
    storage_path.mkdir()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(storage_path))
    monkeypatch.setenv("LOCALAPPDATA", str(storage_path))
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("CREWAI_DISABLE_TRACKING", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    network_attempts: list[str] = []
    original_connect = socket.socket.connect

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) and address else None
        if host in {"127.0.0.1", "::1", "localhost"}:
            return original_connect(sock, address)
        network_attempts.append(repr(address))
        raise AssertionError(f"External network access blocked: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    import httpx

    def blocked_sync_send(_client, request, *args, **kwargs):
        network_attempts.append(str(request.url))
        raise AssertionError(f"HTTP blocked: {request.url}")

    async def blocked_async_send(_client, request, *args, **kwargs):
        network_attempts.append(str(request.url))
        raise AssertionError(f"Async HTTP blocked: {request.url}")

    monkeypatch.setattr(httpx.Client, "send", blocked_sync_send)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked_async_send)

    from pydantic import BaseModel
    from crewai import Agent, BaseLLM, Crew, Process, Task
    from crewai.events.event_bus import crewai_event_bus
    from crewai.flow.flow import Flow, listen, router, start

    class TinyResult(BaseModel):
        verdict: str
        explanation: str

    class FakeLLM(BaseLLM):
        calls: int = 0

        def __init__(self) -> None:
            super().__init__(model="datachef-fake", provider="fake")

        def call(
            self,
            messages,
            tools=None,
            callbacks=None,
            available_functions=None,
            from_task=None,
            from_agent=None,
            response_model=None,
        ):
            self.calls += 1
            payload = {
                "verdict": "APPROVE",
                "explanation": "Synthetic offline result.",
            }
            if response_model is not None:
                return response_model.model_validate(payload)
            return json.dumps(payload)

    class RouteState(BaseModel):
        discover: bool = False
        route: str = ""
        completed: str = ""

    class RouteFlow(Flow[RouteState]):
        @start()
        def begin(self):
            return self.state.discover

        @router(begin)
        def select_route(self):
            self.state.route = "discover" if self.state.discover else "skip"
            return self.state.route

        @listen("discover")
        def discovery(self):
            self.state.completed = "discovery"
            return self.state.completed

        @listen("skip")
        def skipped(self):
            self.state.completed = "skip"
            return self.state.completed

    try:
        fake = FakeLLM()
        agent = Agent(
            role="Offline compatibility reviewer",
            goal="Return the requested typed result",
            backstory="A deterministic fake used only for tests.",
            llm=fake,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description="Return an APPROVE verdict for this synthetic check.",
            expected_output="A typed verdict and explanation.",
            agent=agent,
            output_pydantic=TinyResult,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
            tracing=False,
        )
        result = crew.kickoff()

        assert isinstance(result.pydantic, TinyResult)
        assert result.pydantic.verdict == "APPROVE"
        assert fake.calls == 1

        for discover, expected_route in ((True, "discovery"), (False, "skip")):
            flow = RouteFlow(
                initial_state=RouteState(discover=discover),
                tracing=False,
                suppress_flow_events=True,
            )
            assert flow.kickoff() == expected_route
            restored = RouteState.model_validate_json(flow.state.model_dump_json())
            assert restored.completed == expected_route
    finally:
        crewai_event_bus.shutdown(wait=True)

    assert network_attempts == []
