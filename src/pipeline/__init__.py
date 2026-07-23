"""CI/CD evaluation pipeline for the AI Risk Evaluation Workbench.

This package provides the automation layer that wires the workbench's
evaluation, red-teaming, compliance reporting, regression detection, and
certificate generation into a single continuous-integration workflow. The
GitHub Actions workflow (``.github/workflows/eval.yml``) drives
:mod:`src.pipeline.run`, while each sub-module is independently testable.
"""

from src.pipeline.certificate import (
    CertificateError,
    CertificateStatus,
    ComplianceCertificate,
    aggregate_scores,
    all_checks_pass,
    build_certificate,
    generate_certificate,
    try_generate_certificate,
    write_certificate,
)
from src.pipeline.pr_comment import (
    format_eval_markdown,
    post_pr_comment,
    post_pr_comment_from_results,
)
from src.pipeline.regression import (
    CRITICAL_DIMENSIONS,
    CRITICAL_THRESHOLD,
    DEFAULT_HISTORY_PATH,
    REGRESSION_THRESHOLD,
    RegressionFinding,
    RegressionReport,
    detect_regressions,
    load_history,
    record_run,
    save_history,
)
from src.pipeline.run import (
    MockBackend,
    PipelineConfig,
    run_eval_suite,
    run_pipeline,
    run_redteam,
)

__all__ = [
    # regression
    "REGRESSION_THRESHOLD",
    "CRITICAL_THRESHOLD",
    "CRITICAL_DIMENSIONS",
    "DEFAULT_HISTORY_PATH",
    "RegressionFinding",
    "RegressionReport",
    "load_history",
    "save_history",
    "record_run",
    "detect_regressions",
    # pr_comment
    "format_eval_markdown",
    "post_pr_comment",
    "post_pr_comment_from_results",
    # certificate
    "CertificateStatus",
    "ComplianceCertificate",
    "CertificateError",
    "aggregate_scores",
    "all_checks_pass",
    "build_certificate",
    "generate_certificate",
    "try_generate_certificate",
    "write_certificate",
    # run
    "PipelineConfig",
    "MockBackend",
    "run_eval_suite",
    "run_redteam",
    "run_pipeline",
]
