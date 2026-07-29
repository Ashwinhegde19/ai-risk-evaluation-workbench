"""Tests for the Cline GLM-5.2 gateway backend and its routing.

The live Cline endpoint is never contacted here: HTTP traffic is replaced by an
``httpx.MockTransport`` that returns the exact wrapped envelope the gateway
produces. Tests that need ``httpx`` skip cleanly when it is not installed (e.g.
the CI light install), while the token / routing / think-strip tests run
everywhere because they do not import ``httpx``.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from src.backends.base import ClineBackend, get_backend


# The exact wrapped envelope the Cline gateway returns (NOT the standard OpenAI
# shape): the real completion lives under response["data"]["choices"][...].
WRAPPED_OK = {
    "data": {
        "choices": [
            {
                "message": {"content": "Hi there!", "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "model": "zai/glm-5.2",
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.000025},
    },
    "success": True,
}

WRAPPED_FAIL = {"data": {}, "success": False, "error": "upstream down"}

FAKE_TOKEN = "workos:eyJhbGciOiJSUzI1NiIs-test"


def _empty_home() -> Path:
    """Return a home dir that has no ~/.cline/.../providers.json."""
    return Path(tempfile.mkdtemp(prefix="cline-no-token-"))


class ClineGenerateTests(unittest.TestCase):
    """Wrapped-response extraction via a mocked httpx transport."""

    def test_unwraps_envelope_and_returns_inner_content(self):
        """Content is extracted from response['data']['choices'][0]['message']."""
        httpx = pytest.importorskip("httpx")
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content.decode())
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=WRAPPED_OK)

        backend = ClineBackend(
            model_name="cline-free/glm-5.2",
            base_url="https://api.cline.bot/api/v1",
            _transport=httpx.MockTransport(handler),
        )
        with patch.dict(os.environ, {"CLINE_API_KEY": FAKE_TOKEN}, clear=False):
            result = backend.generate("hello")

        self.assertEqual(result, "Hi there!")
        # The model field is the VERBATIM slug, not a stripped id.
        self.assertEqual(captured["body"]["model"], "cline-free/glm-5.2")
        self.assertEqual(captured["body"]["stream"], False)
        self.assertEqual(captured["auth"], f"Bearer {FAKE_TOKEN}")

    def test_success_false_raises_not_swallowed(self):
        """A {"success": false, ...} body must raise, never return silently."""
        httpx = pytest.importorskip("httpx")

        def handler(request):
            return httpx.Response(200, json=WRAPPED_FAIL)

        backend = ClineBackend(
            model_name="cline-free/glm-5.2",
            _transport=httpx.MockTransport(handler),
        )
        with patch.dict(os.environ, {"CLINE_API_KEY": FAKE_TOKEN}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                backend.generate("hello")
        self.assertIn("success=false", str(ctx.exception))

    def test_retries_on_429_then_succeeds_without_real_sleep(self):
        """429/5xx trigger backoff retries; a later 200 returns the content."""
        httpx = pytest.importorskip("httpx")
        import time

        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(429, text="rate limited")
            return httpx.Response(200, json=WRAPPED_OK)

        backend = ClineBackend(
            model_name="cline-free/glm-5.2",
            _transport=httpx.MockTransport(handler),
        )
        with patch.dict(os.environ, {"CLINE_API_KEY": FAKE_TOKEN}, clear=False), \
             patch.object(time, "sleep", lambda _s: None):
            result = backend.generate("hello")

        self.assertEqual(result, "Hi there!")
        self.assertEqual(calls["n"], 3)


class ClineTokenResolutionTests(unittest.TestCase):
    """Token resolution order: env -> providers.json -> loud error (never mock)."""

    def test_env_token_used_verbatim_with_workos_prefix(self):
        """A full workos: token from env is sent unchanged."""
        httpx = pytest.importorskip("httpx")
        captured = {}

        def handler(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=WRAPPED_OK)

        backend = ClineBackend(
            model_name="cline-free/glm-5.2",
            _transport=httpx.MockTransport(handler),
        )
        with patch.dict(os.environ, {"CLINE_API_KEY": FAKE_TOKEN}, clear=False):
            backend.generate("hi")
        self.assertEqual(captured["auth"], f"Bearer {FAKE_TOKEN}")

    def test_env_bare_token_gets_workos_prefix_prepended(self):
        """A bare token (no workos: prefix) gets the prefix added."""
        httpx = pytest.importorskip("httpx")
        captured = {}

        def handler(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=WRAPPED_OK)

        backend = ClineBackend(
            model_name="cline-free/glm-5.2",
            _transport=httpx.MockTransport(handler),
        )
        with patch.dict(os.environ, {"CLINE_API_KEY": "eyJbare-token"}, clear=False):
            backend.generate("hi")
        self.assertEqual(captured["auth"], "Bearer workos:eyJbare-token")

    def test_providers_json_fallback(self):
        """With no env var, the token is read from providers.json."""
        httpx = pytest.importorskip("httpx")
        captured = {}

        def handler(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json=WRAPPED_OK)

        home = _empty_home()
        providers = home / ".cline" / "data" / "settings"
        providers.mkdir(parents=True)
        (providers / "providers.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "cline": {
                            "settings": {"auth": {"accessToken": "workos:from-file"}}
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        backend = ClineBackend(
            model_name="cline-free/glm-5.2",
            _transport=httpx.MockTransport(handler),
        )
        env = os.environ.copy()
        env.pop("CLINE_API_KEY", None)
        with patch.dict(os.environ, env, clear=True), \
             patch("pathlib.Path.home", return_value=home):
            backend.generate("hi")
        self.assertEqual(captured["auth"], "Bearer workos:from-file")

    def test_missing_token_raises_clear_error_and_does_not_mock(self):
        """No env + no providers.json -> ValueError, not a silent mock."""
        backend = ClineBackend(model_name="cline-free/glm-5.2")
        env = os.environ.copy()
        env.pop("CLINE_API_KEY", None)
        with patch.dict(os.environ, env, clear=True), \
             patch("pathlib.Path.home", return_value=_empty_home()):
            with self.assertRaises(ValueError) as ctx:
                backend.generate("hi")
        msg = str(ctx.exception)
        self.assertIn("CLINE_API_KEY", msg)
        self.assertIn("cline", msg.lower())


class ClineRoutingTests(unittest.TestCase):
    """Routing precedence: cline slugs beat the generic namespaced->Kilo rule."""

    def test_cline_slug_returns_cline_backend_with_verbatim_model(self):
        with patch.dict(os.environ, {"MOCK": "0"}, clear=False):
            backend = get_backend("cline-free/glm-5.2")
        self.assertIsInstance(backend, ClineBackend)
        self.assertEqual(backend.model_name, "cline-free/glm-5.2")

    def test_cline_namespace_prefix_also_routes_to_cline(self):
        with patch.dict(os.environ, {"MOCK": "0"}, clear=False):
            backend = get_backend("cline/some-model")
        self.assertIsInstance(backend, ClineBackend)

    def test_openai_slug_still_routes_to_kilo(self):
        """Regression guard: the precedence change must not steal Kilo slugs."""
        with patch.dict(
            os.environ,
            {"KILO_BASE_URL": "https://kilo.example.com/v1", "KILO_API_KEY": "k"},
            clear=False,
        ):
            backend = get_backend("openai/gpt-5")
        self.assertEqual(backend.base_url, "https://kilo.example.com/v1")
        self.assertEqual(backend.model_name, "openai/gpt-5")

    def test_qwen_slug_still_routes_to_modal(self):
        """Regression guard: open-source lane untouched."""
        with patch.dict(
            os.environ,
            {
                "OPEN_MODEL_BASE_URL": "https://modal.example.com/v1",
                "OPEN_MODEL_API_KEY": "m",
            },
            clear=False,
        ):
            backend = get_backend("qwen3-8b")
        self.assertEqual(backend.base_url, "https://modal.example.com/v1")
        self.assertEqual(backend.model_name, "qwen3-8b")


class ClineThinkStripTests(unittest.TestCase):
    """GLM reasoning blocks are stripped by the model-agnostic visible-output pass."""

    def test_think_block_stripped_from_glm_style_response(self):
        from src.redteam.strategies.base import visible_model_output

        raw = (
            "<think>\nOkay, the user wants the harmful thing. Let me reason...\n</think>\n"
            "Here is the disallowed content you asked for."
        )
        self.assertEqual(
            visible_model_output(raw),
            "Here is the disallowed content you asked for.",
        )

    def test_think_only_response_collapses_to_empty(self):
        from src.redteam.strategies.base import visible_model_output

        self.assertEqual(visible_model_output("<think>\nreasoning only\n</think>"), "")

    def test_break_adjudicator_strip_is_also_model_agnostic(self):
        """The adjudicator's own <think> strip applies to any backend's text."""
        from src.redteam.agent import _BREAK_JUDGE_THINK

        raw = "<think>\nprivate chain of thought\n</think>\nvisible answer"
        self.assertEqual(_BREAK_JUDGE_THINK.sub("", raw).strip(), "visible answer")


if __name__ == "__main__":
    unittest.main()
