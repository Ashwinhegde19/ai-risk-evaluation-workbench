"""Tests for backend routing logic (Kilo gateway vs Modal endpoint)."""

import os
import unittest
from unittest.mock import patch

from src.backends.base import ClineBackend, get_backend


class BackendRoutingTest(unittest.TestCase):
    """Test slug-based routing to Kilo (frontier) vs Modal (open-source)."""

    def test_frontier_model_routes_to_kilo(self):
        """openai/gpt-5 should route to KILO_BASE_URL."""
        with patch.dict(
            os.environ,
            {"KILO_BASE_URL": "https://kilo.example.com/v1", "KILO_API_KEY": "test-key"},
            clear=False,
        ):
            backend = get_backend("openai/gpt-5")
            self.assertEqual(backend.base_url, "https://kilo.example.com/v1")
            self.assertEqual(backend.api_key, "test-key")
            self.assertEqual(backend.model_name, "openai/gpt-5")

    def test_open_source_model_routes_to_modal(self):
        """qwen3-8b should route to OPEN_MODEL_BASE_URL."""
        with patch.dict(
            os.environ,
            {
                "OPEN_MODEL_BASE_URL": "https://modal.example.com/v1",
                "OPEN_MODEL_API_KEY": "modal-key",
            },
            clear=False,
        ):
            backend = get_backend("qwen3-8b")
            self.assertEqual(backend.base_url, "https://modal.example.com/v1")
            self.assertEqual(backend.api_key, "modal-key")
            self.assertEqual(backend.model_name, "qwen3-8b")

    def test_frontier_model_fallback_to_openai(self):
        """If KILO_BASE_URL is unset, fall back to OPENAI_BASE_URL."""
        with patch.dict(
            os.environ,
            {"OPENAI_BASE_URL": "https://api.openai.com/v1", "OPENAI_API_KEY": "openai-key"},
            clear=False,
        ):
            # Remove KILO vars if present
            env = os.environ.copy()
            env.pop("KILO_BASE_URL", None)
            env.pop("KILO_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                backend = get_backend("openai/gpt-5")
                self.assertEqual(backend.base_url, "https://api.openai.com/v1")
                self.assertEqual(backend.api_key, "openai-key")

    def test_frontier_model_missing_base_url_raises(self):
        """If no base URL is set and MOCK != 1, raise ValueError."""
        with patch.dict(os.environ, {"MOCK": "0"}, clear=False):
            env = os.environ.copy()
            env.pop("KILO_BASE_URL", None)
            env.pop("OPENAI_BASE_URL", None)
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError) as ctx:
                    get_backend("openai/gpt-5")
                self.assertIn("KILO_BASE_URL", str(ctx.exception))
                self.assertIn("OPENAI_BASE_URL", str(ctx.exception))

    def test_open_source_model_missing_base_url_raises(self):
        """If OPEN_MODEL_BASE_URL is unset and MOCK != 1, raise ValueError."""
        with patch.dict(os.environ, {"MOCK": "0"}, clear=False):
            env = os.environ.copy()
            env.pop("OPEN_MODEL_BASE_URL", None)
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError) as ctx:
                    get_backend("qwen3-8b")
                self.assertIn("OPEN_MODEL_BASE_URL", str(ctx.exception))

    def test_mock_mode_allows_missing_base_url(self):
        """If MOCK=1, missing base URLs should not raise."""
        with patch.dict(os.environ, {"MOCK": "1"}, clear=True):
            # Should not raise even with no base URLs set
            backend = get_backend("openai/gpt-5")
            self.assertIsNotNone(backend)
            backend2 = get_backend("qwen3-8b")
            self.assertIsNotNone(backend2)

    def test_non_namespaced_model_uses_config(self):
        """Non-namespaced models (e.g., 'gpt-4o') should use config.yaml."""
        # This should load from config.yaml and not trigger routing logic
        backend = get_backend("gpt-4o")
        self.assertIsNotNone(backend)
        self.assertEqual(backend.model_name, "gpt-4o")

    def test_cline_free_glm_5_2_routes_to_kilo(self):
        """cline-free/glm-5.2 now routes through the Kilo gateway (frontier lane)."""
        with patch.dict(os.environ, {"MOCK": "0", "KILO_BASE_URL": "https://kilo.example.com/v1", "KILO_API_KEY": "test-key"}, clear=False):
            backend = get_backend("cline-free/glm-5.2")
            self.assertIsInstance(backend, OpenAIBackend)
            self.assertEqual(backend.model_name, "cline-free/glm-5.2")

    def test_cline_namespace_still_routes_to_cline(self):
        """cline/ prefix models still route to ClineBackend."""
        with patch.dict(os.environ, {"MOCK": "0"}, clear=False):
            backend = get_backend("cline/some-model")
            self.assertIsInstance(backend, ClineBackend)

    def test_cline_namespace_mock_mode_returns_openai_backend(self):
        """In mock mode, Cline slugs return a mock OpenAIBackend."""
        with patch.dict(os.environ, {"MOCK": "1"}, clear=True):
            backend = get_backend("cline/some-model")
            self.assertIsNotNone(backend)
            self.assertEqual(backend.model_name, "cline/some-model")


if __name__ == "__main__":
    unittest.main()
