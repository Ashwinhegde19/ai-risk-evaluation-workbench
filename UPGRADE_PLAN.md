# UPGRADE PLAN — AI Compliance & Red-Team Platform

> Execute phases in order. Each phase = one Claude Code session.
> After each phase: run `pytest tests/ -v`, then commit.

---

## Phase 1: Foundation Refactor (1 session)

### Task 1.1: Pydantic v2 Data Models
Create `src/core/models.py` with:
- `EvalResult`: model_name, dimension, score (0-1), severity, raw_response, judge_scores
- `AttackTurn`: turn_number, attacker_prompt, model_response, strategy_used, escalation_level
- `AttackTree`: root_prompt, turns[], final_score, strategy_chain[], success (bool)
- `ComplianceFinding`: framework (eu_ai_act|nist_rmf|iso_42001), control_id, risk_tier, description, evidence, severity
- `ComplianceReport`: model_name, timestamp, findings[], overall_risk_tier, gaps[]
- `JudgeScore`: judge_model, dimension, score, reasoning, confidence
- `GuardrailResult`: check_type, triggered (bool), details, severity

### Task 1.2: Config System
Create `src/core/config.py`:
- YAML-based config for models, API keys (from env vars), eval suites, guardrail policies
- `config.yaml` template with all options documented

### Task 1.3: Unified Backend Interface
Create `src/backends/base.py`:
- Abstract `ModelBackend` class: `generate(prompt, system_prompt, temperature) -> str`
- `OpenAIBackend`, `AnthropicBackend`, `LocalBackend` implementations
- Factory function: `get_backend(model_name) -> ModelBackend`

### Task 1.4: Tests
- `tests/test_models.py`: Validate all Pydantic models
- `tests/test_backends.py`: Mock backend tests
- `tests/test_config.py`: Config loading tests

**Commit:** `refactor: Phase 1 — foundation architecture`

---

## Phase 2: Multi-Turn Red-Team Agent (2 sessions)

### Task 2.1: Attack Strategy Framework
Create `src/redteam/strategies/` with 8 strategies, each implementing:
```python
class AttackStrategy(ABC):
    name: str
    def generate_prompt(self, turn: int, history: list[AttackTurn]) -> str
    def should_escalate(self, response: str) -> bool
    def get_escalation_prompt(self, turn: int, history: list[AttackTurn]) -> str
```

Strategies:
1. `dan_jailbreak.py` — DAN / jailbreak chains
2. `roleplay.py` — Role-play escalation (grandma, fictional scenario)
3. `encoding.py` — Base64, ROT13, leetspeak, unicode tricks
4. `multilingual.py` — Language switching to bypass filters
5. `context_overflow.py` — Flood context window then inject attack
6. `tool_exploit.py` — Trick model into calling dangerous tools
7. `rag_poison.py` — Inject malicious docs into knowledge base
8. `memory_manip.py` — Exploit conversation memory

### Task 2.2: Adaptive Orchestrator
Create `src/redteam/agent.py`:
- `RedTeamAgent` class that:
  - Takes a target model + strategy list
  - Runs multi-turn conversations (configurable max turns, default 5)
  - Adapts strategy based on model responses (if one fails, switch)
  - Tracks escalation level per turn
  - Builds an `AttackTree` for each attack
  - Returns list of `AttackTree` results

### Task 2.3: Attack Tree Visualization
Create `src/redteam/visualize.py`:
- Text-based tree rendering
- Optional: Graphviz DOT export

### Task 2.4: Tests
- `tests/test_strategies.py`: Each strategy generates valid prompts
- `tests/test_agent.py`: Mock model, verify multi-turn flow
- `tests/test_attack_tree.py`: Tree construction and scoring

**Commit:** `feat: Phase 2 — multi-turn red-team agent`

---

## Phase 3: LLM-as-Judge Ensemble (1 session)

### Task 3.1: Scoring Rubrics
Create `src/judge/rubrics.py`:
- Rubric per dimension: hallucination, bias, toxicity, jailbreak_resistance, privacy, ip_theft, harmful_content
- Each rubric: 5-point scale with clear criteria per level
- Structured output format (JSON schema for judge response)

### Task 3.2: Judge Ensemble
Create `src/judge/ensemble.py`:
- `JudgeEnsemble` class:
  - Uses 3 judge models (configurable, default: GPT-4o, Claude Sonnet, Gemini Pro)
  - Each judge scores independently using rubrics
  - Aggregation: median score + flag disagreements
  - Returns `JudgeScore` per judge + `EnsembleResult`

### Task 3.3: Bias Detection
Create `src/judge/calibration.py`:
- Position bias test: swap order of responses, check if scores change
- Verbosity bias test: pad response with filler, check if scores change
- Self-preference test: judge scores own model's output higher
- Returns bias report with severity

### Task 3.4: Tests
- `tests/test_rubrics.py`: Rubric validation
- `tests/test_ensemble.py`: Mock judges, verify aggregation
- `tests/test_calibration.py`: Bias detection logic

**Commit:** `feat: Phase 3 — calibrated judge ensemble`

---

## Phase 4: Compliance Mapping (1 session)

### Task 4.1: EU AI Act Mapper
Create `src/compliance/eu_ai_act.py`:
- Map each eval dimension + score to EU AI Act risk tier:
  - Unacceptable Risk (Art. 5): social scoring, manipulation, real-time biometric
  - High Risk (Art. 6): employment, credit, law enforcement, education
  - Limited Risk (Art. 50): chatbots, deepfakes, emotion recognition
  - Minimal Risk: everything else
- `classify_risk_tier(eval_results) -> list[ComplianceFinding]`

### Task 4.2: NIST AI RMF Mapper
Create `src/compliance/nist_rmf.py`:
- Map findings to NIST AI RMF functions:
  - GOVERN: policies, accountability
  - MAP: context, risk identification
  - MEASURE: assessment, testing (this is where evals map)
  - MANAGE: mitigation, monitoring
- Each finding gets a control ID (e.g., "MEASURE-2.6")

### Task 4.3: ISO 42001 Mapper
Create `src/compliance/iso_42001.py`:
- Map to ISO 42001 controls (Annex A)
- Focus on: A.7 (AI system impact assessment), A.8 (AI system lifecycle)

### Task 4.4: Compliance Report Generator
Create `src/reports/compliance.py`:
- Takes all eval results + compliance findings
- Generates structured JSON report
- Generates PDF report (use `reportlab` or `weasyprint`)
- Includes: executive summary, per-framework findings, gap analysis, recommendations

### Task 4.5: Tests
- `tests/test_eu_ai_act.py`: Risk tier classification
- `tests/test_nist_rmf.py`: Control mapping
- `tests/test_reports.py`: Report generation

**Commit:** `feat: Phase 4 — compliance mapping`

---

## Phase 5: Real Guardrails (1 session)

### Task 5.1: PII Detection
Create `src/guardrails/pii.py`:
- Use Microsoft Presidio for PII detection
- Detect: names, emails, phones, SSNs, credit cards, addresses
- Return `GuardrailResult` with PII type and location

### Task 5.2: Toxicity Scoring
Create `src/guardrails/toxicity.py`:
- Use `detoxify` library or Perspective API
- Score: toxicity, severe_toxicity, insult, threat, profanity, identity_attack
- Threshold-based triggering

### Task 5.3: Prompt Injection Detection
Create `src/guardrails/injection.py`:
- Pattern-based detection (known injection patterns)
- LLM-based detection (use a small model to classify)
- Return confidence score

### Task 5.4: Guardrail Policies
Create `src/guardrails/policies.py`:
- YAML-configurable policies per deployment tier
- Example: "production" tier blocks all PII + toxicity > 0.7
- Example: "testing" tier logs but doesn't block
- `GuardrailPipeline`: chain multiple checks

### Task 5.5: Tests
- `tests/test_pii.py`: PII detection accuracy
- `tests/test_toxicity.py`: Toxicity scoring
- `tests/test_injection.py`: Injection detection
- `tests/test_policies.py`: Policy enforcement

**Commit:** `feat: Phase 5 — production guardrails`

---

## Phase 6: CI/CD Pipeline (1 session)

### Task 6.1: GitHub Actions Workflow
Create `.github/workflows/eval.yml`:
- Trigger: push to main, PR, manual dispatch
- Steps: install deps → run eval suite → run red-team → generate report
- Upload report as artifact
- Post summary as PR comment

### Task 6.2: Regression Detection
Create `src/pipeline/regression.py`:
- Store historical scores in `data/scores_history.json`
- Compare current run vs last run
- Flag if any dimension drops > 5%
- Fail CI if critical regression detected

### Task 6.3: PR Comment Bot
Create `src/pipeline/pr_comment.py`:
- Format eval results as markdown table
- Post as PR comment via GitHub API
- Include: scores, regressions, compliance status

### Task 6.4: Compliance Certificate
Create `src/pipeline/certificate.py`:
- If all checks pass → generate compliance certificate JSON
- Include: model, timestamp, scores, frameworks checked, validity period

**Commit:** `feat: Phase 6 — CI/CD eval pipeline`

---

## Phase 7: Streamlit Dashboard (1-2 sessions)

### Task 7.1: Main Dashboard
Create `src/dashboard/app.py`:
- Page 1: Overview — radar chart of safety scores per dimension
- Page 2: Model Comparison — side-by-side with response diffs
- Page 3: Red-Team Results — attack tree visualization, drill-down
- Page 4: Compliance — EU AI Act / NIST / ISO findings, gap analysis
- Page 5: Trends — historical score tracking over time

### Task 7.2: Components
- Radar chart (plotly)
- Attack tree renderer (streamlit + graphviz)
- Compliance table with expandable details
- Export buttons (PDF, JSON, CSV)

**Commit:** `feat: Phase 7 — interactive dashboard`

---

## Phase 8: Polish & README (1 session)

### Task 8.1: Professional README
- Architecture diagram (mermaid)
- Screenshots / demo GIFs
- Quick start guide
- Feature list with examples
- Compliance framework coverage table
- Comparison with existing tools (DeepTeam, etc.)

### Task 8.2: Packaging
- `pyproject.toml` with all dependencies
- `Makefile` with common commands
- Docker support (optional)

### Task 8.3: Final Tests
- Ensure >80% test coverage
- Integration test: full pipeline run with mock models
- `pytest tests/ -v --cov=src --cov-report=html`

### Task 8.4: Demo Data
- Pre-generated eval results for demo purposes
- Sample compliance report PDF
- Sample attack tree visualization

**Commit:** `docs: Phase 8 — polish and presentation`

---

## Execution Rules
1. One phase per Claude Code session
2. Run `pytest tests/ -v` after every phase
3. Commit after every phase with the specified message
4. If stuck, re-read this file and continue from the specific task
5. Never skip tests
6. Never hardcode API keys
