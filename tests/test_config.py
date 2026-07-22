"""Tests for the configuration system in ``src.core.config``."""

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from src.core.config import (
    AppConfig,
    GuardrailPolicyConfig,
    ModelConfig,
    default_config,
    load_config,
)


class DefaultConfigTests(unittest.TestCase):
    def test_default_config_has_expected_models(self) -> None:
        cfg = default_config()
        self.assertIn("gpt-4o", cfg.models)
        self.assertIn("claude-sonnet", cfg.models)
        self.assertIn("local", cfg.models)
        self.assertEqual(cfg.default_model, "gpt-4o")

    def test_api_key_is_not_stored_only_env_name(self) -> None:
        cfg = default_config()
        gpt = cfg.get_model("gpt-4o")
        self.assertIsNotNone(gpt)
        self.assertEqual(gpt.api_key_env, "OPENAI_API_KEY")
        # No literal key anywhere in the resolved config dict.
        serialized = cfg.model_dump()
        self.assertNotIn("sk-", str(serialized))

    def test_get_api_key_resolves_from_env(self) -> None:
        cfg = default_config()
        with mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "secret-value"}
        ):
            self.assertEqual(cfg.get_api_key("gpt-4o"), "secret-value")
        # Unset -> None, never a hardcoded fallback.
        self.assertIsNone(cfg.get_api_key("gpt-4o"))

    def test_local_model_has_no_api_key_env(self) -> None:
        cfg = default_config()
        self.assertIsNone(cfg.get_api_key("local"))


class LoadConfigTests(unittest.TestCase):
    def _write_temp(self, text: str) -> Path:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        return Path(path)

    def test_load_config_with_env_substitution(self) -> None:
        yaml_text = textwrap.dedent(
            """
            default_model: my-model
            models:
              my-model:
                name: my-model
                provider: openai
                api_key_env: MY_KEY
                base_url: ${MY_BASE_URL:-https://example.com/v1}
                default_temperature: 0.5
                max_tokens: 1024
            guardrail_policies:
              production:
                name: production
                block_pii: true
                max_toxicity: 0.7
                block_injection: true
            """
        )
        path = self._write_temp(yaml_text)
        try:
            with mock.patch.dict(
                os.environ, {"MY_KEY": "k", "MY_BASE_URL": "https://custom/v1"}
            ):
                cfg = load_config(str(path))
            self.assertEqual(cfg.default_model, "my-model")
            model = cfg.get_model("my-model")
            assert model is not None
            self.assertEqual(model.base_url, "https://custom/v1")
            self.assertEqual(model.default_temperature, 0.5)
            self.assertEqual(model.max_tokens, 1024)
            policy = cfg.get_policy("production")
            self.assertIsInstance(policy, GuardrailPolicyConfig)
            assert policy is not None
            self.assertTrue(policy.block_pii)
        finally:
            path.unlink()

    def test_env_substitution_default_value(self) -> None:
        yaml_text = textwrap.dedent(
            """
            models:
              m:
                name: m
                provider: openai
                api_key_env: MISSING_KEY
                base_url: ${MISSING_URL:-https://fallback/v1}
            """
        )
        path = self._write_temp(yaml_text)
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                cfg = load_config(str(path))
            model = cfg.get_model("m")
            assert model is not None
            self.assertEqual(model.base_url, "https://fallback/v1")
        finally:
            path.unlink()

    def test_load_config_missing_file_returns_default(self) -> None:
        cfg = load_config("/nonexistent/path/config.yaml")
        self.assertIsInstance(cfg, AppConfig)
        self.assertEqual(cfg.default_model, "gpt-4o")


class ModelConfigValidationTests(unittest.TestCase):
    def test_rejects_unknown_provider(self) -> None:
        with self.assertRaises(Exception):
            ModelConfig(name="x", provider="banana")  # type: ignore[arg-type]

    def test_rejects_invalid_temperature(self) -> None:
        with self.assertRaises(Exception):
            ModelConfig(
                name="x", provider="openai", default_temperature=3.0
            )

    def test_numeric_env_string_coerced(self) -> None:
        # Simulates an env-substituted string value becoming a float/int.
        model = ModelConfig(
            name="x",
            provider="openai",
            default_temperature="0.7",  # type: ignore[arg-type]
            max_tokens="2048",  # type: ignore[arg-type]
        )
        self.assertEqual(model.default_temperature, 0.7)
        self.assertEqual(model.max_tokens, 2048)


if __name__ == "__main__":
    unittest.main()
