"""Property tests for forge.memory.redact_secrets."""

from __future__ import annotations

import re

from forge.memory import redact_secrets


class TestRedactSecrets:
    """Tests for the redact_secrets function."""

    def test_api_key_pattern(self) -> None:
        """API keys are redacted."""
        result = redact_secrets("api_key=sk-1234567890abcdef")
        assert "sk-1234567890abcdef" not in result
        assert "«redacted»" in result

    def test_secret_token_pattern(self) -> None:
        """Secret tokens are redacted."""
        result = redact_secrets("secret: my_secret_value_here")
        assert "my_secret_value_here" not in result
        assert "«redacted»" in result

    def test_aws_key_pattern(self) -> None:
        """AWS access keys are redacted."""
        result = redact_secrets("AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "«redacted»" in result

    def test_pem_key_pattern(self) -> None:
        """PEM private keys are redacted."""
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
        result = redact_secrets(text)
        assert "MIIEpAIBAAKCAQEA" not in result
        assert "«redacted»" in result

    def test_long_hex_string(self) -> None:
        """Long hex strings (64+ chars) are redacted."""
        hex_str = "a" * 64
        result = redact_secrets(f"hash={hex_str}")
        assert hex_str not in result
        assert "«redacted»" in result

    def test_regular_prose_unchanged(self) -> None:
        """Regular prose without secrets is unchanged."""
        text = "The quick brown fox jumps over the lazy dog."
        assert redact_secrets(text) == text

    def test_password_pattern(self) -> None:
        """Password values are redacted."""
        result = redact_secrets("password=hunter2")
        assert "hunter2" not in result
        assert "«redacted»" in result

    def test_authorization_header(self) -> None:
        """Authorization headers are redacted."""
        result = redact_secrets("authorization: Bearer eyJhbGciOiJIUzI1NiIs...")
        assert "eyJhbGciOiJIUzI1NiIs" not in result
        assert "«redacted»" in result
