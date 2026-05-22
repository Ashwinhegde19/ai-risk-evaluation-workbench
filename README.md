---
title: AI Assistant Risk Evaluation Workbench
sdk: docker
pinned: false
---

# AI Assistant Risk Evaluation Workbench

This project compares an open-source assistant and a frontier-model assistant through an AI risk lens: hallucination, bias, harmful outputs, jailbreak resistance, latency, and cost.

The goal is to simulate how an AI vendor's assistant could be evaluated before enterprise deployment. Both assistants share the same UI, system prompt, short-term memory, guardrails, logging, and evaluation suite. Only the model backend changes.

## Assistants

| Assistant | Backend | Default model |
|---|---|---|
| Open Source Assistant | Hugging Face Transformers | `Qwen/Qwen2.5-0.5B-Instruct` |
| Frontier Assistant | Kilo Gateway / OpenAI-compatible API | `deepseek/deepseek-v4-flash` |

## Features

- Multi-turn chat with sliding-window short-term memory.
- OSS and frontier model clients behind the same interface.
- Lightweight input and output guardrails for harmful, privacy, jailbreak, and bias-sensitive prompts.
- JSONL chat logs for observability.
- Custom eval suite across factual, hallucination-trap, jailbreak, harmful, bias, privacy, and business-risk prompts.
- CSV evaluation results with pass rate, hallucination flags, unsafe output flags, refusal behavior, bias risk, latency, and estimated cost.
- One-page PDF report generator with comparison charts and a recommendation.
- Chainlit chat interface for a polished assistant demo.

## Architecture

```txt
Chainlit UI
  -> RiskAwareAssistant
      -> SlidingWindowMemory
      -> Guardrails
      -> ModelClient
          -> HuggingFaceOSSClient
          -> FrontierGatewayClient
      -> JSONL logger
  -> Eval Runner
      -> Prompt Dataset
      -> Heuristic Scoring
      -> Optional LLM Judge
      -> CSV Results
      -> PDF Report
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For frontier model support, set:

```bash
FRONTIER_PROVIDER=kilo
KILO_API_KEY=your_kilo_api_key
KILO_BASE_URL=https://api.kilo.ai/api/gateway
FRONTIER_MODEL=deepseek/deepseek-v4-flash
```

`KILOCODE_MODE` is optional. Use it only when you intentionally choose a `kilo-auto/*` model and want Kilo Gateway to route by mode, for example `plan`, `build`, or `general`.

The frontier client also supports direct OpenAI-compatible providers:

```bash
FRONTIER_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=
FRONTIER_MODEL=gpt-4.1-mini
```

For OSS model selection, set:

```bash
OSS_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

## Run The App

```bash
chainlit run app.py
```

The app exposes:

- Assistant selector for OSS vs frontier models.
- Temperature, max-token, and guardrail settings.
- Side-panel request traces with latency and safety metadata.
- Actions to reset memory, run a 5-prompt smoke eval, and generate the PDF report.

## Run Evals From CLI

Run a small smoke eval:

```bash
python -m evals.run_evals --models "Frontier Assistant" --limit 5
```

Run both models:

```bash
python -m evals.run_evals --models "Open Source Assistant" "Frontier Assistant"
```

Run with pre-model blocking enabled:

```bash
python -m evals.run_evals --block-unsafe-inputs
```

Generate the PDF report:

```bash
python reports/generate_report.py
```

## Evaluation Method

The eval set in [evals/prompts.json](evals/prompts.json) contains prompts across:

- `factual_accuracy`
- `hallucination_trap`
- `jailbreak_resistance`
- `harmful_request`
- `bias_sensitive`
- `data_privacy`
- `business_risk`

The scorer records:

- `passed`
- `hallucination_flag`
- `unsafe_flag`
- `correct_refusal`
- `over_refusal`
- `bias_risk`
- `risk_score`
- `latency_ms`
- `cost_per_1k_requests_usd`

An optional LLM-as-judge path can be enabled with `--use-judge` when either `KILO_API_KEY` or `OPENAI_API_KEY` is configured.

## Cost And Latency

The eval runner records latency per prompt and includes estimated cost per 1,000 requests.

| Deployment | Cost input | Notes |
|---|---:|---|
| OSS local / Hugging Face Space | `OSS_COST_PER_1K_REQUESTS_USD` | Defaults to `$0.00`; update this with hosting cost assumptions. |
| Frontier gateway/API | `FRONTIER_COST_PER_1K_REQUESTS_USD` | Defaults to `$0.17`; approximate DeepSeek V4 Flash estimate assuming around 500 input and 500 output tokens per request. |

For the final report, run the eval suite after deployment and use the measured `avg_latency_ms` values from `results/eval_results.csv`.

## Hugging Face Spaces Deployment

1. Create a new Hugging Face Space with `Docker`.
2. Push this repository to the Space.
3. Set Space secrets for any optional API keys:
   - `KILO_API_KEY`
   - `FRONTIER_PROVIDER`
   - `FRONTIER_MODEL`
   - `OSS_MODEL_ID`
4. Use `Qwen/Qwen2.5-0.5B-Instruct` for the public OSS demo to keep memory needs manageable.

## Tradeoffs

- The guardrails are intentionally lightweight and rule-based. This makes the behavior transparent, but it is not a replacement for a production moderation system.
- The OSS model is small enough for a public demo, but it will be less capable than larger OSS or frontier models.
- The heuristic scorer is reproducible and fast, but nuanced safety and hallucination assessment benefits from manual review or LLM-as-judge scoring.
- Sliding-window memory is simple and predictable, but it does not provide long-term user memory or retrieval.

## Improvements With More Time

- Add a stronger safety classifier and policy-specific refusal evaluator.
- Add retrieval grounding for factual/business-policy questions.
- Add prompt versioning, eval run IDs, and comparison across model versions.
- Add OpenTelemetry or Langfuse-style tracing for production observability.
- Add a richer dashboard with per-category drilldowns and failed-case review.
- Deploy larger OSS models on Modal, RunPod, or Replicate and compare cost/latency.

## Submission Checklist

- GitHub repository with complete source code.
- Public OSS demo link from Hugging Face Spaces.
- `reports/evaluation_report.pdf` generated from a real eval run.
- Optional screenshots or Loom walkthrough.
