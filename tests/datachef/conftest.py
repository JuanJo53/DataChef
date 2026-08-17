from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from datachef.contracts import DownstreamUse, UserIntent


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.setenv("DATACHEF_OFFLINE", "true")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("CREWAI_DISABLE_TRACKING", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    import httpx
    import socket

    original_connect = socket.socket.connect

    def blocked_connect(sock, address):
        host = address[0] if isinstance(address, tuple) and address else None
        if host in {"127.0.0.1", "::1", "localhost"}:
            return original_connect(sock, address)
        raise AssertionError("External network access is forbidden in Phase 1A tests")

    def blocked_send(*args, **kwargs):
        del args, kwargs
        raise AssertionError("HTTP access is forbidden in Phase 1A tests")

    async def blocked_async_send(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Async HTTP access is forbidden in Phase 1A tests")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(httpx.Client, "send", blocked_send)
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked_async_send)


@pytest.fixture
def raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [101, 101, 202, 303],
            "category": [" North ", " North ", "South", "West"],
            "amount_text": ["10", "10", "20", "30"],
            "observed_on": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
            "email": [
                "fictional.one@example.test",
                "fictional.one@example.test",
                "fictional.two@example.test",
                "fictional.three@example.test",
            ],
            "phone": [
                "+1 555 010 1000",
                "+1 555 010 1000",
                "+1 555 010 2000",
                "+1 555 010 3000",
            ],
        }
    )


@pytest.fixture
def user_intent() -> UserIntent:
    return UserIntent(
        intent_id="intent-fixture",
        user_goal="",
        downstream_use=DownstreamUse.ANALYSIS,
        selected_key_columns=("customer_id",),
        protected_columns=("email", "phone"),
        required_columns=("customer_id", "amount_text"),
        acceptable_row_loss_pct=30.0,
        questions=(),
    )
