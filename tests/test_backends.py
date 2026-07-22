"""Tests for the unified model backend interface in ``src.backends.base``."""

import unittest
from unittest.mock import MagicMock

from src.backends.base import (
    AnthropicBackend,
    LocalBackend,
    ModelBackend,
    OpenAIBackend,
    get_backend,
)
from src.core.config import AppConfig, ModelConfig, load_config


class BackendInterfaceTests(unittest.TestCase):
    def test_model_backend_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            ModelBackend("x")  # type: ignore[abstract]

    def test_generate_returns_str_from_openai(self) -> None:
        backend = OpenAIBackend(model_name="gpt-4o", api_key="test-key")
        fake_message = MagicMock()
        fake_message.content = "Hello there!"
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        backend.client = mock_client

        out = backend.generate("Hi", system_prompt="Be brief", temperature=0.2)
        self.assertIsInstance(out, str)
        self.assertEqual(out, "Hello there!")
        # Verify the client was called with the expected payload shape.
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertEqual(kwargs["messages"][1]["content"], "Hi")
        self.assertEqual(kwargs["temperature"], 0.2)

    def test_generate_returns_str_from_anthropic(self) -> None:
        backend = AnthropicBackend(model_name="claude-sonnet", api_key="test-key")
        block = MagicMock()
        block.type = "text"
        block.text = "Anthropic reply."
        fake_response = MagicMock()
        fake_response.content = [block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_response
        backend.client = mock_client

        out = backend.generate("Hi", system_prompt="sys")
        self.assertIsInstance(out, str)
        self.assertEqual(out, "Anthropic reply.")
        _, kwargs = mock_client.messages.create.call_args
        self.assertEqual(kwargs["system"], "sys")

    def test_local_backend_uses_injected_pipeline(self) -> None:
        backend = LocalBackend(model_name="local", pipeline=lambda p: f"echo:{p}")
        out = backend.generate("hello")
        self.assertEqual(out, "echo:hello")

    def test_local_backend_prefaces_system_prompt(self) -> None:
        captured: dict[str, str] = {}

        def pipe(prompt: str) -> str:
            captured["p"] = prompt
            return "ok"

        backend = LocalBackend(model_name="local", pipeline=pipe)
        backend.generate("question", system_prompt="You are safe.")
        self.assertTrue(captured["p"].startswith("You are safe."))


class FactoryTests(unittest.TestCase):
    def _config(self) -> AppConfig:
        return AppConfig(
            models={
                "gpt-4o": ModelConfig(
                    name="gpt-4o",
                    provider="openai",
                    api_key_env="OPENAI_API_KEY",
                ),
                "claude-sonnet": ModelConfig(
                    name="claude-sonnet",
                    provider="anthropic",
                    api_key_env="ANTHROPIC_API_KEY",
                ),
                "local": ModelConfig(
                    name="local", provider="local", model_path="Qwen/Test"
                ),
            },
            default_model="gpt-4o",
        )

    def test_factory_returns_openai_backend(self) -> None:
        backend = get_backend("gpt-4o", config=self._config())
        self.assertIsInstance(backend, OpenAIBackend)
        self.assertEqual(backend.model_name, "gpt-4o")

    def test_factory_returns_anthropic_backend(self) -> None:
        backend = get_backend("claude-sonnet", config=self._config())
        self.assertIsInstance(backend, AnthropicBackend)

    def test_factory_returns_local_backend(self) -> None:
        backend = get_backend("local", config=self._config())
        self.assertIsInstance(backend, LocalBackend)

    def test_factory_infers_provider_from_name(self) -> None:
        # Unknown model name -> provider inferred, no config needed.
        backend = get_backend("gpt-4-turbo")
        self.assertIsInstance(backend, OpenAIBackend)
        backend2 = get_backend("claude-opus")
        self.assertIsInstance(backend2, AnthropicBackend)
        backend3 = get_backend("my-local-model")
        self.assertIsInstance(backend3, LocalBackend)

    def test_factory_uses_default_config_when_none_passed(self) -> None:
        # Should not raise and should resolve against shipped config/defaults.
        backend = get_backend("gpt-4o")
        self.assertIsInstance(backend, OpenAIBackend)


if __name__ == "__main__":
    unittest.main()
