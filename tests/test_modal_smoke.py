"""Tests for Modal smoke test script."""

import json
import os
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts/ to path so we can import the smoke test
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from modal_smoke_test import check_chat_completion, check_models, main


class ModalSmokeTestTest(unittest.TestCase):
    """Test Modal smoke test logic."""

    def test_check_models_success(self):
        """check_models should return True if qwen3-8b is listed."""
        mock_response = {
            "object": "list",
            "data": [
                {"id": "qwen3-8b", "object": "model", "created": 1234567890},
                {"id": "other-model", "object": "model", "created": 1234567890},
            ],
        }
        with patch("modal_smoke_test._http_json", return_value=mock_response):
            result = check_models("https://modal.example.com/v1")
            self.assertTrue(result)

    def test_check_models_failure(self):
        """check_models should return False if qwen3-8b is not listed."""
        mock_response = {
            "object": "list",
            "data": [{"id": "other-model", "object": "model", "created": 1234567890}],
        }
        with patch("modal_smoke_test._http_json", return_value=mock_response):
            with patch("sys.stderr", new_callable=StringIO):
                result = check_models("https://modal.example.com/v1")
                self.assertFalse(result)

    def test_check_chat_completion_success(self):
        """check_chat_completion should return True if MODAL_L4_OK is in response."""
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "qwen3-8b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "MODAL_L4_OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        with patch("modal_smoke_test._http_json", return_value=mock_response):
            result = check_chat_completion("https://modal.example.com/v1")
            self.assertTrue(result)

    def test_check_chat_completion_failure(self):
        """check_chat_completion should return False if MODAL_L4_OK is missing."""
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "qwen3-8b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Some other response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        with patch("modal_smoke_test._http_json", return_value=mock_response):
            with patch("sys.stderr", new_callable=StringIO):
                result = check_chat_completion("https://modal.example.com/v1")
                self.assertFalse(result)

    def test_main_missing_base_url(self):
        """main should return 1 if OPEN_MODEL_BASE_URL is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("sys.stderr", new_callable=StringIO):
                result = main()
                self.assertEqual(result, 1)

    def test_main_all_checks_pass(self):
        """main should return 0 if all checks pass."""
        with patch.dict(
            os.environ,
            {"OPEN_MODEL_BASE_URL": "https://modal.example.com/v1"},
            clear=False,
        ):
            with patch("modal_smoke_test.check_models", return_value=True):
                with patch("modal_smoke_test.check_chat_completion", return_value=True):
                    result = main()
                    self.assertEqual(result, 0)

    def test_main_models_check_fails(self):
        """main should return 1 if check_models fails."""
        with patch.dict(
            os.environ,
            {"OPEN_MODEL_BASE_URL": "https://modal.example.com/v1"},
            clear=False,
        ):
            with patch("modal_smoke_test.check_models", return_value=False):
                with patch("sys.stderr", new_callable=StringIO):
                    result = main()
                    self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
