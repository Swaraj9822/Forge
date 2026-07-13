"""Unit tests for Forge startup validation (``forge.app``).

These tests run fully offline and exercise the two *fatal* startup checks that
run before any network or credential call:

* :func:`forge.app.validate_required_config` — raises :class:`StartupError`
  directing the user to run ``forge init`` when the required GCP project ID or
  region is absent, blank, or still the ``forge init`` placeholder
  (Req 2.4, 12.3).
* :func:`forge.app.check_adc` — probes Application Default Credentials and
  raises :class:`StartupError` whose message names the
  ``gcloud auth application-default login`` command when ADC are unavailable
  (Req 2.3).

The ADC probe is driven through the module-level
``forge.app._google_auth_default`` callable (the guarded ``google.auth.default``
import), which is monkeypatched per test so no real credential lookup occurs.
"""

from __future__ import annotations

import pytest

from forge import app as app_module
from forge.app import StartupError, check_adc, validate_required_config
from forge.config import PROJECT_PLACEHOLDER, REGION_PLACEHOLDER, Config
from forge.vertex import CredentialsError

# The exact command the ADC error message must name (Req 2.3).
ADC_COMMAND = "gcloud auth application-default login"


# ---------------------------------------------------------------------------
# validate_required_config — missing project/region direct to `forge init`
# ---------------------------------------------------------------------------


def test_validate_passes_when_project_and_region_present() -> None:
    """A fully-configured Config does not raise (Req 2.4, 12.3)."""
    config = Config(project="my-project", region="us-central1")

    # Should not raise.
    assert validate_required_config(config) is None


@pytest.mark.parametrize(
    "project, region",
    [
        (None, "us-central1"),  # project absent
        ("", "us-central1"),  # project blank
        ("   ", "us-central1"),  # project whitespace-only
        (PROJECT_PLACEHOLDER, "us-central1"),  # project still the placeholder
    ],
)
def test_validate_raises_when_project_missing(project, region) -> None:
    """A missing/blank/placeholder project directs the user to ``forge init``."""
    config = Config(project=project, region=region)

    with pytest.raises(StartupError) as excinfo:
        validate_required_config(config)

    message = excinfo.value.message
    assert "forge init" in message
    assert "GCP project ID" in message
    assert excinfo.value.exit_code != 0


@pytest.mark.parametrize(
    "project, region",
    [
        ("my-project", None),  # region absent
        ("my-project", ""),  # region blank
        ("my-project", "   "),  # region whitespace-only
        ("my-project", REGION_PLACEHOLDER),  # region still the placeholder
    ],
)
def test_validate_raises_when_region_missing(project, region) -> None:
    """A missing/blank/placeholder region directs the user to ``forge init``."""
    config = Config(project=project, region=region)

    with pytest.raises(StartupError) as excinfo:
        validate_required_config(config)

    message = excinfo.value.message
    assert "forge init" in message
    assert "GCP region" in message
    assert excinfo.value.exit_code != 0


def test_validate_raises_when_both_missing_names_both_values() -> None:
    """When both required values are missing, both are named (Req 2.4, 12.3)."""
    config = Config(project=None, region=None)

    with pytest.raises(StartupError) as excinfo:
        validate_required_config(config)

    message = excinfo.value.message
    assert "forge init" in message
    assert "GCP project ID" in message
    assert "GCP region" in message


def test_validate_default_config_directs_to_forge_init() -> None:
    """The default Config (no project/region) directs the user to ``forge init``."""
    with pytest.raises(StartupError) as excinfo:
        validate_required_config(Config())

    assert "forge init" in excinfo.value.message


# ---------------------------------------------------------------------------
# check_adc — missing ADC names `gcloud auth application-default login`
# ---------------------------------------------------------------------------


def test_check_adc_passes_with_valid_credentials(monkeypatch) -> None:
    """Valid credentials returned by the auth library do not raise (Req 2.3)."""
    credential = object()

    def fake_default():
        return credential, "resolved-project"

    monkeypatch.setattr(app_module, "_google_auth_default", fake_default)

    # Should not raise.
    assert check_adc() is None


def test_check_adc_raises_when_auth_library_raises(monkeypatch) -> None:
    """An auth failure surfaces a StartupError naming the gcloud command (Req 2.3)."""

    def fake_default():
        raise RuntimeError("no application default credentials found")

    monkeypatch.setattr(app_module, "_google_auth_default", fake_default)

    with pytest.raises(StartupError) as excinfo:
        check_adc()

    message = excinfo.value.message
    assert ADC_COMMAND in message
    # The message reuses the Vertex client's canonical ADC text.
    assert message == CredentialsError.DEFAULT_MESSAGE


def test_check_adc_raises_when_no_credentials_returned(monkeypatch) -> None:
    """No credentials (``None``) surfaces a StartupError naming the command (Req 2.3)."""

    def fake_default():
        return None, "resolved-project"

    monkeypatch.setattr(app_module, "_google_auth_default", fake_default)

    with pytest.raises(StartupError) as excinfo:
        check_adc()

    message = excinfo.value.message
    assert ADC_COMMAND in message
    assert message == CredentialsError.DEFAULT_MESSAGE


def test_check_adc_skipped_when_auth_library_unavailable(monkeypatch) -> None:
    """When the auth library is absent the probe is skipped and does not raise."""
    monkeypatch.setattr(app_module, "_google_auth_default", None)

    # The check defers to the VertexClient's request-time credential check.
    assert check_adc() is None


def test_validate_required_config_anthropic_and_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Anthropic passes when key present in env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    config_ant = Config(provider_type="anthropic")
    assert validate_required_config(config_ant) is None

    # Raises when missing
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(StartupError) as excinfo:
        validate_required_config(config_ant)
    assert "ANTHROPIC_API_KEY" in excinfo.value.message

    # 2. OpenAI passes when key present in env
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    config_oa = Config(provider_type="openai")
    assert validate_required_config(config_oa) is None

    # Raises when missing
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(StartupError) as excinfo:
        validate_required_config(config_oa)
    assert "OPENAI_API_KEY" in excinfo.value.message
