"""LLM-as-Judge ensemble for the AI Risk Evaluation Workbench.

Public API:
    - :mod:`src.judge.rubrics` -- scoring rubrics and judge I/O helpers.
    - :mod:`src.judge.ensemble` -- multi-model judge aggregation.
    - :mod:`src.judge.calibration` -- judge bias detection.
"""

from src.core.models import JudgeScore
from src.judge.calibration import (
    BiasDetector,
    BiasFinding,
    BiasReport,
    DEFAULT_BIAS_THRESHOLD,
    run_position_bias,
    run_self_preference,
    run_verbosity_bias,
    severity_from_delta,
)
from src.judge.ensemble import (
    DEFAULT_DISAGREEMENT_THRESHOLD,
    DEFAULT_JUDGE_MODELS,
    DEFAULT_MAX_CONCURRENCY,
    EnsembleResult,
    JudgeEnsemble,
)
from src.judge.rubrics import (
    RiskDimension,
    Rubric,
    RubricLevel,
    build_judge_prompt,
    get_rubric,
    judge_response_schema,
    ParsedJudgeResponse,
    parse_judge_response,
    rating_to_score,
    rubric_to_judge_score,
)

__all__ = [
    # shared model
    "JudgeScore",
    # rubrics
    "RiskDimension",
    "Rubric",
    "RubricLevel",
    "rating_to_score",
    "judge_response_schema",
    "get_rubric",
    "build_judge_prompt",
    "ParsedJudgeResponse",
    "parse_judge_response",
    "rubric_to_judge_score",
    # ensemble
    "DEFAULT_JUDGE_MODELS",
    "DEFAULT_DISAGREEMENT_THRESHOLD",
    "DEFAULT_MAX_CONCURRENCY",
    "EnsembleResult",
    "JudgeEnsemble",
    # calibration
    "DEFAULT_BIAS_THRESHOLD",
    "severity_from_delta",
    "BiasFinding",
    "BiasReport",
    "run_position_bias",
    "run_verbosity_bias",
    "run_self_preference",
    "BiasDetector",
]
