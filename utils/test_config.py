"""Tests for the centralised model configuration (utils.config)."""

import pytest

from utils.config import DEFAULT_MODELS, MODEL_ENV_BY_ROLE, api_key, model_for

ALL_ROLES = sorted(DEFAULT_MODELS)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from a known state, whatever the developer's .env says."""
    monkeypatch.delenv("DATACHEF_MODEL", raising=False)
    for var in MODEL_ENV_BY_ROLE.values():
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("role", ALL_ROLES)
def test_falls_back_to_the_builtin_default(role):
    assert model_for(role) == DEFAULT_MODELS[role]


@pytest.mark.parametrize("role", ALL_ROLES)
def test_shared_variable_overrides_every_role(monkeypatch, role):
    monkeypatch.setenv("DATACHEF_MODEL", "team-model")

    assert model_for(role) == "team-model"


@pytest.mark.parametrize("role", ALL_ROLES)
def test_role_variable_wins_over_the_shared_one(monkeypatch, role):
    monkeypatch.setenv("DATACHEF_MODEL", "team-model")
    monkeypatch.setenv(MODEL_ENV_BY_ROLE[role], "my-model")

    assert model_for(role) == "my-model"


def test_a_role_override_does_not_leak_into_other_roles(monkeypatch):
    # The point of the whole exercise: one teammate pinning the transform model
    # must not silently change the dashboard's.
    monkeypatch.setenv("DATACHEF_MODEL_TRANSFORM", "his-model")

    assert model_for("transform") == "his-model"
    assert model_for("insights") == DEFAULT_MODELS["insights"]
    assert model_for("chat") == DEFAULT_MODELS["chat"]


def test_blank_or_whitespace_variables_are_ignored(monkeypatch):
    # A variable left empty in .env must not blank out the model name.
    monkeypatch.setenv("DATACHEF_MODEL", "   ")
    monkeypatch.setenv("DATACHEF_MODEL_CHAT", "")

    assert model_for("chat") == DEFAULT_MODELS["chat"]


def test_values_are_stripped(monkeypatch):
    monkeypatch.setenv("DATACHEF_MODEL_CHAT", "  spaced-model  ")

    assert model_for("chat") == "spaced-model"


def test_unknown_role_is_rejected_loudly():
    with pytest.raises(ValueError) as excinfo:
        model_for("nope")

    assert "nope" in str(excinfo.value)


def test_defaults_keep_transform_on_its_own_quota_pool():
    # Free-tier quota is per model, so the critical path must not share a model
    # with the optional calls that fire on every dashboard render.
    assert DEFAULT_MODELS["transform"] != DEFAULT_MODELS["insights"]
    assert DEFAULT_MODELS["transform"] != DEFAULT_MODELS["chat"]


def test_every_role_has_an_env_variable():
    assert set(MODEL_ENV_BY_ROLE) == set(DEFAULT_MODELS)


def test_api_key_prefers_google_and_accepts_the_legacy_alias(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert api_key() is None

    monkeypatch.setenv("GEMINI_API_KEY", "legacy")
    assert api_key() == "legacy"

    monkeypatch.setenv("GOOGLE_API_KEY", "canonical")
    assert api_key() == "canonical"
