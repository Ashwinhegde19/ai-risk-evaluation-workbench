"""Tests for the Streamlit dashboard package (``src.dashboard``)."""

import importlib.util
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.core.models import AttackTree, ComplianceReport, EvalResult
from src.dashboard import DashboardData, components
from src.dashboard.data_loader import discover_data
from src.dashboard.sample_data import DEMO_MODELS, generate_dashboard_data


def _have_plotly() -> bool:
    """Return ``True`` if ``plotly`` is importable in this environment."""
    return importlib.util.find_spec("plotly") is not None


class SampleDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = generate_dashboard_data()

    def test_returns_dashboard_data(self) -> None:
        self.assertIsInstance(self.data, DashboardData)

    def test_contains_expected_models(self) -> None:
        self.assertEqual(set(self.data.models), set(DEMO_MODELS))
        for result in self.data.eval_results:
            self.assertIn(result.model_name, DEMO_MODELS)

    def test_eval_results_valid_and_in_range(self) -> None:
        self.assertTrue(self.data.eval_results)
        dims_seen = set()
        for result in self.data.eval_results:
            self.assertIsInstance(result, EvalResult)
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)
            dims_seen.add(result.dimension)
        # All seven canonical dimensions should be covered.
        self.assertEqual(len(dims_seen), 7)

    def test_attack_trees_valid(self) -> None:
        self.assertTrue(self.data.attack_trees)
        for tree in self.data.attack_trees:
            self.assertIsInstance(tree, AttackTree)
            self.assertTrue(0.0 <= tree.final_score <= 1.0)

    def test_reports_one_per_model(self) -> None:
        self.assertEqual(set(self.data.reports.keys()), set(DEMO_MODELS))
        for report in self.data.reports.values():
            self.assertIsInstance(report, ComplianceReport)
            self.assertEqual(report.model_name, report.model_name)

    def test_history_shape(self) -> None:
        # 6 timestamps * 3 models = 18 run records.
        self.assertEqual(len(self.data.history), 18)
        for run in self.data.history:
            self.assertIn("timestamp", run)
            self.assertIn("model", run)
            self.assertIn("scores", run)
            self.assertEqual(len(run["scores"]), 7)

    def test_deterministic(self) -> None:
        other = generate_dashboard_data()
        self.assertEqual(self.data, other)


class DataLoaderTests(unittest.TestCase):
    def _write_artifacts(self, directory: Path) -> DashboardData:
        data = generate_dashboard_data()
        (directory / "eval_results.json").write_text(
            json.dumps([r.model_dump() for r in data.eval_results]), encoding="utf-8"
        )
        (directory / "attack_trees.json").write_text(
            json.dumps([t.model_dump() for t in data.attack_trees]), encoding="utf-8"
        )
        # The file-based path carries a single compliance report.
        report = next(iter(data.reports.values()))
        (directory / "compliance_report.json").write_text(
            report.model_dump_json(), encoding="utf-8"
        )
        (directory / "scores_history.json").write_text(
            json.dumps(data.history), encoding="utf-8"
        )
        return data

    def test_discover_loads_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            expected = self._write_artifacts(directory)
            loaded = discover_data(directory)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.source, str(directory))
            self.assertEqual(len(loaded.eval_results), len(expected.eval_results))
            self.assertEqual(len(loaded.attack_trees), len(expected.attack_trees))
            report = next(iter(expected.reports.values()))
            self.assertEqual(set(loaded.reports.keys()), {report.model_name})
            self.assertEqual(len(loaded.history), len(expected.history))

    def test_discover_only_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data = generate_dashboard_data()
            (directory / "scores_history.json").write_text(
                json.dumps(data.history), encoding="utf-8"
            )
            loaded = discover_data(directory)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.eval_results, [])
            self.assertEqual(loaded.history, data.history)

    def test_discover_missing_dir_returns_none(self) -> None:
        loaded = discover_data(Path("/nonexistent/path/xyz"))
        self.assertIsNone(loaded)

    def test_discover_empty_dir_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = discover_data(Path(tmp))
            self.assertIsNone(loaded)


class ComponentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = generate_dashboard_data()
        self.results = self.data.eval_results

    def test_severity_from_score_boundaries(self) -> None:
        from src.core.models import Severity

        self.assertEqual(components.severity_from_score(0.9), Severity.LOW)
        self.assertEqual(components.severity_from_score(0.8), Severity.MEDIUM)
        self.assertEqual(components.severity_from_score(0.6), Severity.HIGH)
        self.assertEqual(components.severity_from_score(0.2), Severity.CRITICAL)

    def test_aggregate_dimension_scores(self) -> None:
        agg = components.aggregate_dimension_scores(self.results)
        self.assertEqual(set(agg.keys()), set(components.DIMENSION_ORDER))
        for score in agg.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_aggregate_by_model(self) -> None:
        by_model = components.aggregate_by_model(self.results)
        self.assertEqual(set(by_model.keys()), set(DEMO_MODELS))
        for dims in by_model.values():
            self.assertEqual(set(dims.keys()), set(components.DIMENSION_ORDER))

    def test_model_comparison_rows(self) -> None:
        rows = components.model_comparison_rows(self.results)
        self.assertEqual(len(rows), len(self.results))
        for row in rows:
            self.assertIn("model", row)
            self.assertIn("dimension", row)
            self.assertIn("score", row)
            self.assertIn("severity", row)

    def test_finding_rows(self) -> None:
        report = next(iter(self.data.reports.values()))
        rows = components.finding_rows(report.findings)
        self.assertEqual(len(rows), len(report.findings))
        for row in rows:
            self.assertIn("framework", row)
            self.assertIn("control_id", row)
            self.assertIn("evidence", row)

    def test_attack_tree_dot(self) -> None:
        dot = components.attack_tree_dot(self.data.attack_trees[0])
        self.assertIn("digraph", dot)
        self.assertIn("ROOT", dot)

    def test_to_csv_roundtrip(self) -> None:
        rows = components.model_comparison_rows(self.results)
        csv_text = components.to_csv(rows)
        self.assertTrue(csv_text.strip())
        import csv

        reader = csv.DictReader(csv_text.splitlines())
        parsed = list(reader)
        self.assertEqual(len(parsed), len(rows))
        self.assertIn("model", reader.fieldnames)
        self.assertIn("score", reader.fieldnames)

    def test_to_csv_empty(self) -> None:
        self.assertEqual(components.to_csv([]), "")

    def test_to_json(self) -> None:
        report = next(iter(self.data.reports.values()))
        text = components.to_json(report)
        # Should be valid JSON that round-trips to the same model.
        reparsed = ComplianceReport.model_validate_json(text)
        self.assertEqual(reparsed.model_name, report.model_name)

    def test_radar_figure(self) -> None:
        if not _have_plotly():
            self.skipTest("plotly not installed")
        agg = components.aggregate_dimension_scores(self.results)
        fig = components.radar_figure(agg, title="Test")
        self.assertEqual(len(fig.data), 1)
        self.assertTrue(hasattr(fig, "to_json"))

    def test_radar_figure_multi(self) -> None:
        if not _have_plotly():
            self.skipTest("plotly not installed")
        by_model = components.aggregate_by_model(self.results)
        fig = components.radar_figure_multi(by_model)
        self.assertEqual(len(fig.data), len(by_model))

    def test_trend_figure(self) -> None:
        if not _have_plotly():
            self.skipTest("plotly not installed")
        fig = components.trend_figure(self.data.history)
        # One trace per model in the history.
        self.assertEqual(len(fig.data), len(set(r["model"] for r in self.data.history)))

    def test_compliance_pdf_bytes(self) -> None:
        report = next(iter(self.data.reports.values()))
        pdf = components.compliance_pdf_bytes(report)
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_export_compliance_pdf_writes_file(self) -> None:
        report = next(iter(self.data.reports.values()))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.pdf"
            path = components.export_compliance_pdf(report, out)
            self.assertTrue(Path(path).exists())
            self.assertTrue(Path(path).read_bytes().startswith(b"%PDF"))


class AppPageTests(unittest.TestCase):
    """Exercise the Streamlit page functions with a mocked ``st`` module."""

    def setUp(self) -> None:
        import src.dashboard.app as app

        self.app = app
        self.data = generate_dashboard_data()
        self.mock_st = unittest.mock.MagicMock()
        # columns(n) must return an unpackable tuple of exactly n mocks so
        # that pages using 2 or 4 columns both work.
        self.mock_st.columns.side_effect = (
            lambda n=1: tuple(unittest.mock.MagicMock() for _ in range(int(n)))
        )
        # Default widget returns for pages that read them.
        self.mock_st.multiselect.return_value = list(self.data.models)
        self.mock_st.selectbox.return_value = self.data.dimensions[0]
        app.st = self.mock_st

    def tearDown(self) -> None:
        # Restore the real streamlit module binding.
        import streamlit as st

        self.app.st = st

    def test_page_overview(self) -> None:
        self.app.page_overview(self.data)
        self.mock_st.plotly_chart.assert_called()

    def test_page_model_comparison(self) -> None:
        self.app.page_model_comparison(self.data)
        self.mock_st.plotly_chart.assert_called()

    def test_page_redteam(self) -> None:
        self.mock_st.selectbox.return_value = 0
        self.app.page_redteam(self.data)
        self.mock_st.graphviz_chart.assert_called()

    def test_page_compliance(self) -> None:
        model = next(iter(self.data.reports.keys()))
        self.mock_st.selectbox.return_value = model
        self.app.page_compliance(self.data)
        # PDF + JSON download buttons should have been offered.
        self.assertGreaterEqual(self.mock_st.download_button.call_count, 2)

    def test_page_trends(self) -> None:
        self.mock_st.multiselect.return_value = list(self.data.models)
        self.app.page_trends(self.data)
        self.mock_st.plotly_chart.assert_called()

    def test_main_dispatches_overview(self) -> None:
        self.app.main(page="Overview", data_dir="results", use_demo=True)
        self.mock_st.plotly_chart.assert_called()

    def test_main_unknown_page_errors(self) -> None:
        self.app.main(page="Nope", data_dir="results", use_demo=True)
        self.mock_st.error.assert_called()


if __name__ == "__main__":
    unittest.main()
