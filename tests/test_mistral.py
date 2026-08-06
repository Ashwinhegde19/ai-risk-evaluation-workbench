"""Tests for the Mistral/Shieldstral backend and guardrail integration."""

import unittest
from unittest.mock import MagicMock, patch

from src.backends.base import (
    MistralBackend,
    MistralShieldstralBackend,
    OpenAIBackend,
    get_backend,
)


class MistralBackendTests(unittest.TestCase):
    """Tests for MistralBackend (Mistral AI API)."""

    def test_mistral_backend_generate(self) -> None:
        backend = MistralBackend(model_name="mistral-small-latest", api_key="test-key")
        fake_message = MagicMock()
        fake_message.content = "Mistral response."
        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = fake_response
        backend.client = mock_client

        out = backend.generate("Hello", temperature=0.3)
        self.assertEqual(out, "Mistral response.")
        _, kwargs = mock_client.chat.complete.call_args
        self.assertEqual(kwargs["model"], "mistral-small-latest")
        self.assertEqual(kwargs["temperature"], 0.3)

    def test_mistral_backend_raises_without_api_key(self) -> None:
        backend = MistralBackend(model_name="mistral-small-latest")
        with self.assertRaises(ValueError):
            backend.generate("test")


class MistralShieldstralBackendTests(unittest.TestCase):
    """Tests for MistralShieldstralBackend (self-deployed on Modal)."""

    def _make_mock_httpx_client(self, response_text: str) -> MagicMock:
        """Create a mock httpx client that returns the given response text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": response_text},
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": "yes" if "yes" in response_text else "no", "logprob": -0.01},
                                    {"token": "no" if "no" in response_text else "yes", "logprob": -5.0},
                                ]
                            }
                        ]
                    },
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        return mock_client

    def test_classify_safety_unsafe(self) -> None:
        backend = MistralShieldstralBackend()
        backend.client = self._make_mock_httpx_client("yes")

        result = backend.classify_safety(
            query="Does this promote violence?",
            document="Here's how to hurt someone...",
        )
        self.assertEqual(result["verdict"], "unsafe")
        self.assertGreater(result["unsafe_score"], 0.9)

    def test_classify_safety_safe(self) -> None:
        backend = MistralShieldstralBackend()
        backend.client = self._make_mock_httpx_client("no")

        result = backend.classify_safety(
            query="Does this promote violence?",
            document="The weather is nice today.",
        )
        self.assertEqual(result["verdict"], "safe")
        self.assertGreater(result["safe_score"], 0.9)

    def test_classify_safety_uncertain(self) -> None:
        backend = MistralShieldstralBackend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "maybe"},
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": "perhaps", "logprob": -1.0},
                                    {"token": "possibly", "logprob": -1.5},
                                ]
                            }
                        ]
                    },
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        backend.client = mock_client

        result = backend.classify_safety(
            query="Does this promote violence?",
            document="Something ambiguous.",
        )
        self.assertEqual(result["verdict"], "uncertain")

    def test_generate_delegates_to_inner(self) -> None:
        backend = MistralShieldstralBackend()
        backend.client = self._make_mock_httpx_client("response")

        out = backend.generate("test prompt")
        self.assertEqual(out, "response")


class MistralRoutingTests(unittest.TestCase):
    """Tests for backend routing to Mistral lanes."""

    @patch("src.backends.base._is_open_model", return_value=False)
    @patch("src.backends.base._is_cline_model", return_value=False)
    @patch("src.backends.base._is_frontier_slug", return_value=False)
    @patch("src.backends.base.os.environ.get")
    def test_shieldstral_slug_routes_to_self_deployed(
        self, mock_environ: MagicMock, *_: MagicMock
    ) -> None:
        """mistral-shieldstral slug routes to MistralShieldstralBackend."""
        mock_environ.side_effect = lambda key, default=None: {
            "MISTRAL_MODEL_BASE_URL": "https://test.modal.run/v1",
        }.get(key, default)

        backend = get_backend("mistral-shieldstral")
        self.assertIsInstance(backend, MistralShieldstralBackend)

    @patch("src.backends.base._is_open_model", return_value=False)
    @patch("src.backends.base._is_cline_model", return_value=False)
    @patch("src.backends.base._is_frontier_slug", return_value=False)
    def test_mistral_api_slug_routes_to_mistral_backend(self, *_: MagicMock) -> None:
        """mistral/mistral-small slug routes to MistralBackend."""
        with patch("src.backends.base._is_mock_enabled", return_value=True):
            backend = get_backend("mistral/mistral-small")
            # In mock mode, falls back to OpenAIBackend
            self.assertIsInstance(backend, OpenAIBackend)

    @patch("src.backends.base._is_open_model", return_value=False)
    @patch("src.backends.base._is_cline_model", return_value=False)
    @patch("src.backends.base._is_frontier_slug", return_value=False)
    @patch("src.backends.base.os.environ.get")
    def test_shieldstral_missing_base_url_raises(
        self, mock_environ: MagicMock, *_: MagicMock
    ) -> None:
        """Missing MISTRAL_MODEL_BASE_URL raises in real mode."""
        mock_environ.side_effect = lambda key, default=None: {
            "MISTRAL_MODEL_BASE_URL": None,
        }.get(key, default)

        with patch("src.backends.base._is_mock_enabled", return_value=False):
            with self.assertRaises(ValueError):
                get_backend("mistral-shieldstral")


class ShieldstralGuardrailTests(unittest.TestCase):
    """Tests for the ShieldstralGuardrail wrapper."""

    def test_scan_delegates_to_backend(self) -> None:
        from src.guardrails.shieldstral import ShieldstralGuardrail

        mock_backend = MagicMock(spec=MistralShieldstralBackend)
        mock_backend.classify_safety.return_value = {
            "safe_score": 1.0,
            "unsafe_score": 0.0,
            "verdict": "safe",
        }

        guardrail = ShieldstralGuardrail(
            policy="Does this promote violence?",
            backend=mock_backend,
        )
        result = guardrail.scan("The weather is nice.")
        self.assertFalse(result.triggered)
        self.assertEqual(result.check_type, "shieldstral")

    def test_scan_unsafe_triggers(self) -> None:
        from src.guardrails.shieldstral import ShieldstralGuardrail

        mock_backend = MagicMock(spec=MistralShieldstralBackend)
        mock_backend.classify_safety.return_value = {
            "safe_score": 0.0,
            "unsafe_score": 1.0,
            "verdict": "unsafe",
        }

        guardrail = ShieldstralGuardrail(
            policy="Does this promote violence?",
            strictness="high",
            backend=mock_backend,
        )
        result = guardrail.scan("Harmful content here.")
        self.assertTrue(result.triggered)

    def test_scan_backend_failure_returns_safe(self) -> None:
        from src.guardrails.shieldstral import ShieldstralGuardrail

        mock_backend = MagicMock(spec=MistralShieldstralBackend)
        mock_backend.classify_safety.side_effect = RuntimeError("Connection failed")

        guardrail = ShieldstralGuardrail(
            policy="Does this promote violence?",
            backend=mock_backend,
        )
        result = guardrail.scan("Some text.")
        # Fail open (don't block) when backend is unavailable
        self.assertFalse(result.triggered)
        self.assertIn("Connection failed", result.details)

    def test_scan_with_threshold(self) -> None:
        from src.guardrails.shieldstral import ShieldstralGuardrail

        mock_backend = MagicMock(spec=MistralShieldstralBackend)
        mock_backend.classify_safety.return_value = {
            "safe_score": 0.2,
            "unsafe_score": 0.8,
            "verdict": "unsafe",
        }

        guardrail = ShieldstralGuardrail(
            policy="Does this contain PII?",
            backend=mock_backend,
        )
        result = guardrail.scan_with_threshold("Contact: test@email.com", threshold=0.5)
        self.assertTrue(result.triggered)


if __name__ == "__main__":
    unittest.main()
