"""Toxicity scoring guardrail.

Scores a piece of text across six toxicity dimensions: ``toxicity``,
``severe_toxicity``, ``insult``, ``threat``, ``profanity`` and
``identity_attack``. The primary scorer uses `Detoxify
<https://github.com/unitaryai/detoxify>`_ when the package is installed; when
it is not available a deterministic lexicon-based scorer is used so the
guardrail remains functional in any environment.

Both engines return the same :class:`ToxicityScores` structure and a
:class:`~src.core.models.GuardrailResult` indicating whether the configured
threshold was exceeded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.models import BaseWorkbenchModel, GuardrailResult, Severity

# Lexicons for the fallback scorer. These are representative negative tokens,
# not exhaustive; production deployments should rely on Detoxify for nuance.
_PROFANITY_LEXICON: List[str] = [
    "damn",
    "crap",
    "hell",
    "bloody",
    "bastard",
    "ass",
    "shit",
    "piss",
]
_INSULT_LEXICON: List[str] = [
    "stupid",
    "idiot",
    "moron",
    "pathetic",
    "loser",
    "worthless",
    "dumb",
    "imbecile",
]
_THREAT_LEXICON: List[str] = [
    "kill",
    "murder",
    "destroy you",
    "hunt you down",
    "hurt you",
    "burn you",
    "beat you",
]
_IDENTITY_ATTACK_LEXICON: List[str] = [
    "inferior race",
    "genetically inferior",
    "subhuman",
    "should be eliminated",
    "vermin",
]

# Weight applied per keyword match in the lexicon scorer (capped at 1.0).
_LEXICON_WEIGHT = 0.5


class ToxicityScores(BaseWorkbenchModel):
    """Per-dimension toxicity scores in the range ``[0, 1]``.

    Attributes:
        toxicity: Overall toxicity.
        severe_toxicity: Severe / high-confidence toxicity.
        insult: Insulting language.
        threat: Threats of harm.
        profanity: Profane language.
        identity_attack: Attacks against an identity group.
    """

    toxicity: float
    severe_toxicity: float
    insult: float
    threat: float
    profanity: float
    identity_attack: float


class ToxicityScorer:
    """Score text for toxicity.

    Uses Detoxify when available, otherwise a deterministic lexicon scorer.
    The public API is identical regardless of the active engine.

    Args:
        use_detoxify: When ``False``, the lexicon fallback is always used even
            if Detoxify is installed (useful for reproducible tests).
        threshold: Default toxicity score that flips :meth:`scan` to triggered.
    """

    def __init__(self, use_detoxify: bool = True, threshold: float = 0.7) -> None:
        """Initialise the scorer and (lazily) select an engine."""
        self.threshold: float = threshold
        self._detoxify = self._build_detoxify() if use_detoxify else None

    @staticmethod
    def _build_detoxify() -> object:
        """Attempt to construct a Detoxify model.

        Returns:
            A Detoxify model instance, or ``None`` if Detoxify is unavailable.
        """
        try:  # pragma: no cover - exercised only when detoxify is installed
            from detoxify import Detoxify

            return Detoxify("original")
        except Exception:
            return None

    @property
    def engine_name(self) -> str:
        """Return the name of the active scoring engine."""
        return "detoxify" if self._detoxify is not None else "lexicon"

    def score(self, text: str) -> ToxicityScores:
        """Score ``text`` across all six toxicity dimensions.

        Args:
            text: The text to score.

        Returns:
            A :class:`ToxicityScores` with each dimension in ``[0, 1]``.
        """
        if not text or not text.strip():
            return ToxicityScores(
                toxicity=0.0,
                severe_toxicity=0.0,
                insult=0.0,
                threat=0.0,
                profanity=0.0,
                identity_attack=0.0,
            )
        if self._detoxify is not None:  # pragma: no cover - needs detoxify
            return self._score_detoxify(text)
        return self._score_lexicon(text)

    def scan(self, text: str, threshold: Optional[float] = None) -> GuardrailResult:
        """Score text and return a :class:`GuardrailResult`.

        Args:
            text: The text to score.
            threshold: Optional override for the triggering threshold.

        Returns:
            A :class:`GuardrailResult` with ``check_type="toxicity"``. It is
            ``triggered`` when the maximum dimension score meets the threshold.
        """
        scores = self.score(text)
        threshold = self.threshold if threshold is None else threshold
        maximum = max(
            scores.toxicity,
            scores.severe_toxicity,
            scores.insult,
            scores.threat,
            scores.profanity,
            scores.identity_attack,
        )
        triggered = maximum >= threshold
        details = (
            f"toxicity={scores.toxicity:.2f} severe={scores.severe_toxicity:.2f} "
            f"insult={scores.insult:.2f} threat={scores.threat:.2f} "
            f"profanity={scores.profanity:.2f} identity={scores.identity_attack:.2f}"
        )
        severity = (
            Severity.HIGH if maximum >= threshold else Severity.INFO
        )
        return GuardrailResult(
            check_type="toxicity",
            triggered=triggered,
            details=details,
            severity=severity,
        )

    # -- internal engines ---------------------------------------------------

    def _score_detoxify(self, text: str) -> ToxicityScores:
        """Score text using the Detoxify model.

        Args:
            text: The text to score.

        Returns:
            Toxicity scores mapped from Detoxify's output.
        """
        raw: Dict[str, float] = self._detoxify.predict(text)
        return ToxicityScores(
            toxicity=float(raw.get("toxicity", 0.0)),
            severe_toxicity=float(raw.get("severe_toxicity", 0.0)),
            insult=float(raw.get("insult", 0.0)),
            threat=float(raw.get("threat", 0.0)),
            profanity=float(raw.get("obscene", 0.0)),
            identity_attack=float(raw.get("identity_attack", 0.0)),
        )

    def _score_lexicon(self, text: str) -> ToxicityScores:
        """Score text using deterministic lexicon matching.

        Args:
            text: The text to score (case-insensitive matching is performed).

        Returns:
            Toxicity scores derived from keyword frequencies.
        """
        lower = text.lower()
        profanity = self._category_score(lower, _PROFANITY_LEXICON)
        insult = self._category_score(lower, _INSULT_LEXICON)
        threat = self._category_score(lower, _THREAT_LEXICON)
        identity_attack = self._category_score(lower, _IDENTITY_ATTACK_LEXICON)
        severe_toxicity = min(1.0, max(threat, identity_attack, profanity * 0.6))
        toxicity = min(
            1.0,
            profanity * 0.4
            + insult * 0.3
            + threat * 0.4
            + identity_attack * 0.4
            + severe_toxicity * 0.2,
        )
        return ToxicityScores(
            toxicity=toxicity,
            severe_toxicity=severe_toxicity,
            insult=insult,
            threat=threat,
            profanity=profanity,
            identity_attack=identity_attack,
        )

    @staticmethod
    def _category_score(text: str, lexicon: List[str]) -> float:
        """Count keyword matches in ``text`` and convert to a ``[0, 1]`` score.

        Args:
            text: Lowercased text to scan.
            lexicon: Keyword phrases to count.

        Returns:
            ``min(1.0, matches * _LEXICON_WEIGHT)``.
        """
        matches = sum(text.count(phrase) for phrase in lexicon)
        return min(1.0, matches * _LEXICON_WEIGHT)


__all__ = [
    "ToxicityScores",
    "ToxicityScorer",
]
