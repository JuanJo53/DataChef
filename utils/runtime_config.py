"""Sanitized runtime-configuration inspection for DataChef.

This module deliberately reports only whether a credential exists. It never
returns or serializes the credential value, so its result is safe to log or
show in a startup diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import os
import re

from pydantic import BaseModel, ConfigDict, Field


class ProviderStatus(str, Enum):
    """Provider readiness without exposing provider credentials."""

    OFFLINE = "OFFLINE"
    MISSING_CREDENTIAL = "MISSING_CREDENTIAL"
    CONFIGURED = "CONFIGURED"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"


class ModelStatus(str, Enum):
    """Local model validation status; live availability is checked separately."""

    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNVERIFIED = "UNVERIFIED"
    INVALID = "INVALID"


class CredentialSource(str, Enum):
    """Safe identifier for the environment variable that supplied a key."""

    NONE = "NONE"
    GOOGLE_API_KEY = "GOOGLE_API_KEY"
    LEGACY_GEMINI_API_KEY = "GEMINI_API_KEY"


class ConfigurationCheck(BaseModel):
    """A log-safe snapshot of DataChef provider configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_status: ProviderStatus
    model_status: ModelStatus
    credential_present: bool
    credential_source: CredentialSource
    model_configured: bool
    offline: bool
    send_row_samples: bool
    experimental_code_execution: bool
    ready_for_approved_live_probe: bool
    messages: list[str] = Field(default_factory=list)


_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_PLACEHOLDER_MODEL_RE = re.compile(r"^(?:change[-_ ]?me|your[-_ ]?model|<.*>)$", re.I)


def _parse_bool(environment: Mapping[str, str], name: str, default: bool) -> bool:
    raw_value = environment.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be true or false")


def _model_status(model_name: str | None) -> ModelStatus:
    if model_name is None:
        return ModelStatus.NOT_CONFIGURED
    if any(character.isspace() for character in model_name):
        return ModelStatus.INVALID
    if _PLACEHOLDER_MODEL_RE.fullmatch(model_name):
        return ModelStatus.INVALID
    return ModelStatus.UNVERIFIED


def inspect_runtime_configuration(
    environment: Mapping[str, str] | None = None,
) -> ConfigurationCheck:
    """Inspect environment configuration without performing a network call.

    ``GOOGLE_API_KEY`` is canonical. ``GEMINI_API_KEY`` is recognized only as
    a temporary compatibility alias for existing repository modules.
    """

    env = os.environ if environment is None else environment
    messages: list[str] = []

    try:
        offline = _parse_bool(env, "DATACHEF_OFFLINE", default=True)
        send_row_samples = _parse_bool(
            env, "DATACHEF_SEND_ROW_SAMPLES", default=False
        )
        experimental_code_execution = _parse_bool(
            env,
            "DATACHEF_ENABLE_EXPERIMENTAL_CODE_EXECUTION",
            default=False,
        )
    except ValueError as exc:
        return ConfigurationCheck(
            provider_status=ProviderStatus.INVALID_CONFIGURATION,
            model_status=ModelStatus.NOT_CONFIGURED,
            credential_present=False,
            credential_source=CredentialSource.NONE,
            model_configured=False,
            offline=True,
            send_row_samples=False,
            experimental_code_execution=False,
            ready_for_approved_live_probe=False,
            messages=[str(exc)],
        )

    canonical_key = env.get("GOOGLE_API_KEY", "").strip()
    legacy_key = env.get("GEMINI_API_KEY", "").strip()
    if canonical_key:
        credential_source = CredentialSource.GOOGLE_API_KEY
        credential_present = True
        if legacy_key:
            messages.append(
                "Both key aliases are set; GOOGLE_API_KEY takes precedence."
            )
    elif legacy_key:
        credential_source = CredentialSource.LEGACY_GEMINI_API_KEY
        credential_present = True
        messages.append(
            "GEMINI_API_KEY is a legacy alias; migrate local setup to GOOGLE_API_KEY."
        )
    else:
        credential_source = CredentialSource.NONE
        credential_present = False

    raw_model = env.get("GEMINI_MODEL", "").strip()
    model_name = raw_model or None
    model_status = _model_status(model_name)

    if offline:
        provider_status = ProviderStatus.OFFLINE
        messages.append("Offline mode prevents provider calls.")
    elif not credential_present:
        provider_status = ProviderStatus.MISSING_CREDENTIAL
        messages.append("GOOGLE_API_KEY is required when offline mode is disabled.")
    else:
        provider_status = ProviderStatus.CONFIGURED

    if model_status is ModelStatus.NOT_CONFIGURED:
        messages.append("GEMINI_MODEL is not configured.")
    elif model_status is ModelStatus.INVALID:
        messages.append("GEMINI_MODEL has an invalid local format or placeholder.")
    else:
        messages.append("GEMINI_MODEL is configured but has not been verified live.")

    ready_for_approved_live_probe = (
        provider_status is ProviderStatus.CONFIGURED
        and model_status is ModelStatus.UNVERIFIED
        and not send_row_samples
        and not experimental_code_execution
    )

    return ConfigurationCheck(
        provider_status=provider_status,
        model_status=model_status,
        credential_present=credential_present,
        credential_source=credential_source,
        model_configured=model_name is not None,
        offline=offline,
        send_row_samples=send_row_samples,
        experimental_code_execution=experimental_code_execution,
        ready_for_approved_live_probe=ready_for_approved_live_probe,
        messages=messages,
    )
