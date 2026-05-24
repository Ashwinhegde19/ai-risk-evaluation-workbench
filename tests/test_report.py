import unittest

import pandas as pd

from reports.generate_report import ensure_report_columns, summarize


class ReportCompatibilityTests(unittest.TestCase):
    def test_summarize_accepts_legacy_eval_rows(self):
        df = pd.DataFrame(
            [
                {
                    "model_label": "Open Source Assistant",
                    "prompt_id": "p1",
                    "passed": 1,
                    "hallucination_flag": 0,
                    "unsafe_flag": 0,
                    "bias_risk": 0,
                    "over_refusal": 0,
                    "risk_score": 0,
                    "latency_ms": 100,
                    "cost_per_1k_requests_usd": 0.0,
                }
            ]
        )

        summary = summarize(ensure_report_columns(df))

        self.assertEqual(summary.loc[0, "under_refusal_rate"], 0)


if __name__ == "__main__":
    unittest.main()
