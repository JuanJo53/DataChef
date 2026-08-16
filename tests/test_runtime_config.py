from utils.runtime_config import (
    CredentialSource,
    ModelStatus,
    ProviderStatus,
    inspect_runtime_configuration,
)


def test_offline_is_the_safe_default() -> None:
    check = inspect_runtime_configuration({})

    assert check.provider_status is ProviderStatus.OFFLINE
    assert check.model_status is ModelStatus.NOT_CONFIGURED
    assert check.credential_present is False
    assert check.ready_for_approved_live_probe is False
    assert check.send_row_samples is False
    assert check.experimental_code_execution is False


def test_missing_credential_is_distinct_from_offline() -> None:
    check = inspect_runtime_configuration({"DATACHEF_OFFLINE": "false"})

    assert check.provider_status is ProviderStatus.MISSING_CREDENTIAL
    assert check.credential_source is CredentialSource.NONE
    assert check.ready_for_approved_live_probe is False


def test_configured_provider_keeps_model_unverified() -> None:
    secret = "synthetic-test-secret"
    check = inspect_runtime_configuration(
        {
            "DATACHEF_OFFLINE": "false",
            "GOOGLE_API_KEY": secret,
            "GEMINI_MODEL": "synthetic-model-name",
        }
    )

    assert check.provider_status is ProviderStatus.CONFIGURED
    assert check.model_status is ModelStatus.UNVERIFIED
    assert check.credential_source is CredentialSource.GOOGLE_API_KEY
    assert check.ready_for_approved_live_probe is True
    assert secret not in check.model_dump_json()


def test_legacy_key_is_only_a_compatibility_alias() -> None:
    check = inspect_runtime_configuration(
        {
            "DATACHEF_OFFLINE": "false",
            "GEMINI_API_KEY": "synthetic-legacy-secret",
        }
    )

    assert check.provider_status is ProviderStatus.CONFIGURED
    assert check.credential_source is CredentialSource.LEGACY_GEMINI_API_KEY
    assert any("legacy alias" in message for message in check.messages)


def test_invalid_model_placeholder_is_reported() -> None:
    check = inspect_runtime_configuration(
        {
            "DATACHEF_OFFLINE": "false",
            "GOOGLE_API_KEY": "synthetic-test-secret",
            "GEMINI_MODEL": "<your-model>",
        }
    )

    assert check.model_status is ModelStatus.INVALID
    assert check.ready_for_approved_live_probe is False


def test_invalid_boolean_is_a_sanitized_configuration_error() -> None:
    check = inspect_runtime_configuration({"DATACHEF_OFFLINE": "sometimes"})

    assert check.provider_status is ProviderStatus.INVALID_CONFIGURATION
    assert check.credential_present is False
    assert check.ready_for_approved_live_probe is False
    assert check.messages == ["DATACHEF_OFFLINE must be true or false"]
