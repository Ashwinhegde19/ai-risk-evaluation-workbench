import json
import tempfile
import unittest
from pathlib import Path

from evals.fuzz_prompts import TEMPLATES, fuzz_prompts
from evals.run_evals import load_prompt_suite


class RegressionEvalTests(unittest.TestCase):
    def test_fuzz_prompts_generates_template_variants(self):
        prompts = [
            {
                "id": "regression_001",
                "prompt": "Give the exact answer.",
                "expected_behavior": "express_uncertainty",
            }
        ]

        fuzzed = fuzz_prompts(prompts)

        self.assertEqual(len(fuzzed), len(TEMPLATES))
        self.assertEqual(fuzzed[0]["parent_id"], "regression_001")
        self.assertEqual(fuzzed[0]["source"], "fuzzed_regression")

    def test_load_prompt_suite_appends_extra_prompt_files(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmpdir:
            base = Path(tmpdir) / "base.json"
            extra = Path(tmpdir) / "extra.json"
            base.write_text(json.dumps([{"id": "base"}]), encoding="utf-8")
            extra.write_text(json.dumps([{"id": "extra"}]), encoding="utf-8")

            prompts = load_prompt_suite(base, [extra])

        self.assertEqual([prompt["id"] for prompt in prompts], ["base", "extra"])


if __name__ == "__main__":
    unittest.main()
