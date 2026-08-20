"""Model construction driven entirely by GEMINI_MODEL.

No model string is hardcoded here. The three legacy crew agents pin a model
literal and have all gone stale; this module reads the configured name through
``inspect_runtime_configuration``, which also decides whether the live path is
permitted at all and never returns the credential value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

from utils.runtime_config import ModelStatus, ProviderStatus, inspect_runtime_configuration


@dataclass(frozen=True, slots=True)
class LiveModelDecision:
    """Whether a live crew may run, and under which configured model."""

    permitted: bool
    model_name: str | None
    model_configured: bool


def decide_live_model(
    environment: Mapping[str, str] | None = None,
) -> LiveModelDecision:
    """Live mode requires offline off, a credential present, and a model set."""

    env = os.environ if environment is None else environment
    check = inspect_runtime_configuration(env)
    model_configured = check.model_status is not ModelStatus.NOT_CONFIGURED
    permitted = (
        not check.offline
        and check.provider_status is ProviderStatus.CONFIGURED
        and check.model_status is ModelStatus.UNVERIFIED
    )
    name = (env.get("GEMINI_MODEL", "") or "").strip() or None
    return LiveModelDecision(
        permitted=bool(permitted and name),
        model_name=name if permitted else None,
        model_configured=model_configured,
    )


def build_llm(model_name: str):
    """Construct the CrewAI LLM. Imported inside the function by design."""

    from crewai import LLM

    return LLM(model=f"gemini/{model_name}")


__all__ = ["LiveModelDecision", "build_llm", "decide_live_model"]
