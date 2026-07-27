"""Production guardrails for the AI Risk Evaluation Workbench.

This package provides input/output guardrails:

* :mod:`src.guardrails.pii` — PII detection (Presidio, regex fallback).
* :mod:`src.guardrails.toxicity` — toxicity scoring (Detoxify, lexicon fallback).
* :mod:`src.guardrails.injection` — prompt-injection detection.
* :mod:`src.guardrails.policies` — policy-driven guardrail pipeline.
"""

from src.guardrails.injection import (
    InjectionDetector,
    InjectionFinding,
    InjectionResult,
)
from src.guardrails.pii import PiiDetector, PiiFinding, PiiType
from src.guardrails.policies import (
    GuardrailPipeline,
    GuardrailPipelineResult,
    build_default_pipeline,
    build_production_pipeline,
    build_testing_pipeline,
)
from src.guardrails.toxicity import ToxicityScores, ToxicityScorer

__all__ = [
    "PiiType",
    "PiiFinding",
    "PiiDetector",
    "ToxicityScores",
    "ToxicityScorer",
    "InjectionFinding",
    "InjectionResult",
    "InjectionDetector",
    "GuardrailPipelineResult",
    "GuardrailPipeline",
    "build_default_pipeline",
    "build_production_pipeline",
    "build_testing_pipeline",
]
