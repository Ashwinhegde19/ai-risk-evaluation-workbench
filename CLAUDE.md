# CLAUDE.md — AI Risk Evaluation Workbench

## Project Overview
AI Compliance & Red-Team Platform that evaluates LLMs against EU AI Act,
NIST AI RMF, and ISO 42001. Runs multi-turn adversarial red-teaming,
calibrated LLM-as-Judge scoring, and generates audit-ready compliance reports.

## Tech Stack
- Python 3.11+
- Pydantic v2 (all data models)
- FastAPI (API layer)
- Streamlit (dashboard)
- pytest (testing)
- GitHub Actions (CI/CD)

## Repo Structure
```
ai-risk-evaluation-workbench/
├── src/
│   ├── core/              # Pydantic models, schemas, config
│   ├── backends/          # Model backends (OpenAI, Anthropic, local)
│   ├── redteam/           # Multi-turn red-team agent
│   │   ├── strategies/    # Attack strategies (DAN, role-play, encoding, etc.)
│   │   └── agent.py       # Orchestrator
│   ├── judge/             # LLM-as-Judge ensemble
│   │   ├── rubrics.py     # Scoring rubrics per risk dimension
│   │   ├── ensemble.py    # Multi-model judge orchestration
│   │   └── calibration.py # Judge bias detection
│   ├── compliance/        # Regulatory mapping
│   │   ├── eu_ai_act.py   # EU AI Act risk tier mapping
│   │   ├── nist_rmf.py    # NIST AI RMF control mapping
│   │   └── iso_42001.py   # ISO 42001 control mapping
│   ├── guardrails/        # Input/output guardrails
│   │   ├── pii.py         # PII detection (Presidio)
│   │   ├── toxicity.py    # Toxicity scoring
│   │   └── injection.py   # Prompt injection detection
│   ├── pipeline/          # CI/CD eval pipeline
│   ├── dashboard/         # Streamlit app
│   └── reports/           # Report generators (PDF, JSON)
├── tests/
├── data/
│   ├── eval_suites/       # Test datasets
│   └── knowledge_base/    # RAG poisoning test docs
├── .github/workflows/     # CI/CD
├── CLAUDE.md
├── UPGRADE_PLAN.md
└── pyproject.toml
```

## Commands
```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run eval suite
python -m src.pipeline.run --model gpt-4o --suite full

# Run red-team
python -m src.redteam.agent --model gpt-4o --turns 5 --strategy all

# Generate compliance report
python -m src.reports.generate --format pdf --framework eu_ai_act

# Start dashboard
streamlit run src/dashboard/app.py
```

## Coding Standards
- All models use Pydantic v2 with strict validation
- Every function has type hints
- Every module has a docstring
- Tests required for all new code (pytest)
- No hardcoded API keys — use environment variables
- Commit messages: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`

## Current Phase
Executing UPGRADE_PLAN.md — follow phase order strictly.
