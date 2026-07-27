"""Smoke test for the self-deployed Modal Qwen3-8B endpoint.

This is the gate to run before any real evaluation against the open-source
target. It verifies that the Modal L4 endpoint is actually serving Qwen3-8B
(not a mock or a misconfigured proxy) by:

1. Reading ``OPEN_MODEL_BASE_URL`` from the environment (exit 1 if missing).
2. ``GET {base}/models`` and asserting ``qwen3-8b`` is listed.
3. Sending ONE chat completion -- ``"Reply exactly: MODAL_L4_OK"`` -- and
   asserting the response contains ``MODAL_L4_OK``.
4. Printing the model name, response, token usage and latency.

Exits ``0`` only if every check passes; exits ``1`` on any failure with a hint.

Usage::

    python scripts/modal_smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# The exact marker the model is asked to echo back. A real Qwen3-8B endpoint
# will comply; a mock/canned backend or a broken proxy will not.
EXPECTED_MARKER = "MODAL_L4_OK"
MODEL_NAME = "qwen3-8b"
REQUEST_TIMEOUT = 120.0  # seconds; long generations at 4K context can be slow


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Perform an HTTP GET/POST and parse the JSON body.

    Args:
        url: The request URL.
        payload: Optional JSON body; when provided a POST is issued.

    Returns:
        The parsed JSON response as a dict.

    Raises:
        urllib.error.HTTPError: On a non-2xx response.
        urllib.error.URLError: On a connection failure.
    """
    data = None
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPEN_MODEL_API_KEY", "none")
    if api_key and api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_models(base_url: str) -> bool:
    """Assert that ``qwen3-8b`` is listed by the endpoint's ``/models`` route.

    Args:
        base_url: The OpenAI-compatible base URL (ending in ``/v1``).

    Returns:
        ``True`` if the model is listed, ``False`` otherwise.
    """
    models_url = f"{base_url}/models"
    print(f"[smoke] GET {models_url}")
    body = _http_json(models_url)
    listed = [m.get("id") for m in body.get("data", [])]
    print(f"[smoke] models listed: {listed}")
    if MODEL_NAME not in listed:
        print(
            f"[smoke] FAIL: '{MODEL_NAME}' not in /v1/models. "
            "Is the Modal endpoint serving Qwen3-8B with --served-model-name qwen3-8b?",
            file=sys.stderr,
        )
        return False
    return True


def check_chat_completion(base_url: str) -> bool:
    """Send one chat completion and assert the marker is echoed back.

    Args:
        base_url: The OpenAI-compatible base URL (ending in ``/v1``).

    Returns:
        ``True`` if the response contains ``MODAL_L4_OK``, ``False`` otherwise.
    """
    completions_url = f"{base_url}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": f"Reply exactly: {EXPECTED_MARKER}"}],
        "temperature": 0.0,
        "max_tokens": 32,
    }
    print(f"[smoke] POST {completions_url}")
    start = time.monotonic()
    body = _http_json(completions_url, payload)
    latency = time.monotonic() - start

    try:
        content = body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        print(f"[smoke] FAIL: unexpected response shape: {body}", file=sys.stderr)
        return False

    usage = body.get("usage", {})
    model = body.get("model", MODEL_NAME)
    print(f"[smoke] model:      {model}")
    print(f"[smoke] response:   {content!r}")
    print(f"[smoke] usage:      {usage}")
    print(f"[smoke] latency:    {latency:.2f}s")

    if EXPECTED_MARKER not in content:
        print(
            f"[smoke] FAIL: response does not contain '{EXPECTED_MARKER}'. "
            "This usually means a mock/canned backend or a broken proxy, not a "
            "live Qwen3-8B endpoint.",
            file=sys.stderr,
        )
        return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    """Run the smoke test.

    Args:
        argv: Unused; accepted for CLI symmetry.

    Returns:
        ``0`` if all checks pass, ``1`` otherwise.
    """
    base_url = os.getenv("OPEN_MODEL_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        print(
            "[smoke] FAIL: OPEN_MODEL_BASE_URL is not set. Deploy the Modal "
            "endpoint (modal deploy modal_deploy.py) and export its /v1 URL.",
            file=sys.stderr,
        )
        return 1

    print(f"[smoke] target base_url: {base_url}")
    try:
        if not check_models(base_url):
            return 1
        if not check_chat_completion(base_url):
            return 1
    except urllib.error.HTTPError as exc:
        print(
            f"[smoke] FAIL: HTTP {exc.code} from endpoint. Hint: check the Modal "
            "app is deployed and not cold-starting / OOM.",
            file=sys.stderr,
        )
        return 1
    except urllib.error.URLError as exc:
        print(f"[smoke] FAIL: could not reach endpoint: {exc.reason}", file=sys.stderr)
        return 1

    print("[smoke] PASS: Modal L4 endpoint is serving Qwen3-8B correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
