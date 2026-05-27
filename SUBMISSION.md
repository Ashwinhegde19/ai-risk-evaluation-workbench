# Submission Guide

## What This Project Demonstrates

This is a lightweight AI risk evaluation workbench for comparing an open-source assistant and a frontier assistant under the same product surface.

It focuses on the risks an AI vendor would need to understand before enterprise deployment:

- hallucination and unsupported factual claims
- jailbreak and prompt-injection resistance
- bias and discriminatory output risk
- harmful request refusal behavior
- privacy and data disclosure risk
- business liability in regulated or customer-facing workflows
- deterministic tool use for calculator and AI-risk checklist requests
- policy inference for prompts without manual labels
- retrieval grounding and claim verification for factual questions
- latency and cost tradeoffs across model backends

## What To Review First

1. Open the live Hugging Face demo or run the Chainlit app locally.
2. Switch between `Use OSS` and `Use Frontier`.
3. Try `calculate: (42 * 17) / 3` or `create an AI risk checklist for an insurance assistant` to verify tool use.
4. Open `View Logs` to inspect model, latency, safety labels, tool calls, and pre-model blocking.
5. Run the eval suite and generate the one-page PDF report.
6. Review notable eval cases in the PDF to see concrete model failures.

## Architecture

```txt
Chainlit UI
  -> RiskAwareAssistant
      -> SlidingWindowMemory
      -> Guardrails
      -> AssistantTools
      -> EvidenceRetriever
          -> Local knowledge base
          -> Optional web search
      -> ModelClient
          -> HuggingFaceOSSClient
          -> ModalEndpointClient
          -> FrontierGatewayClient
      -> JSONL logger
  -> Eval Runner
      -> Prompt Dataset
      -> Policy Inference
      -> Retrieval / Claim Verification
      -> Heuristic Scoring
      -> Optional LLM Judge
      -> CSV Results
      -> PDF Report
```

Both assistants share the same system prompt, memory, guardrails, tool layer, logging, UI, and eval suite. Only the model backend changes.

## Model Choices

Open-source default:

```txt
Qwen/Qwen2.5-0.5B-Instruct
```

This model was chosen because it is small enough for local testing and public Hugging Face Spaces CPU deployment. A stronger optional local-quality model is documented:

```txt
Qwen/Qwen2.5-1.5B-Instruct
```

The OSS backend can also be routed through an optional Modal endpoint:

```txt
OSS_BACKEND=modal
MODAL_OSS_ENDPOINT=<hosted endpoint>
```

The scaffold lives in:

```txt
modal_app/oss_endpoint.py
```

Frontier default:

```txt
deepseek/deepseek-v4-flash
```

This is routed through Kilo Gateway using an OpenAI-compatible client. The gateway abstraction keeps the app provider-flexible while preserving one assistant interface.

## Live Demo

```txt
https://ashwinhegde19-ai-risk-evaluation-workbench.hf.space
```

## Key Commands

```bash
source .venv/bin/activate
chainlit run app.py --host 127.0.0.1 --port 7860
```

```bash
python -m evals.run_evals --models "Open Source Assistant" "Frontier Assistant" --block-unsafe-inputs --max-tokens 256
python reports/generate_report.py
```

```bash
python -m evals.run_unlabelled_evals --models "Open Source Assistant" "Frontier Assistant" --enable-retrieval --verify-claims --block-unsafe-inputs --max-tokens 256
python reports/generate_report.py --results-path results/unlabelled_eval_results.csv --output-path reports/unlabelled_evaluation_report.pdf
```

## Current Tradeoffs

- The guardrails are transparent and lightweight, but not a replacement for a production moderation classifier.
- The policy inference router is a v1 implementation that can be replaced by Prompt Guard, Llama Guard, or Modal-hosted classifiers.
- The claim verifier is conservative and catches unsupported evidence patterns, but it is not a full groundedness model.
- Tool use is deterministic and narrow so it is auditable, but it is not a full agent planner.
- The OSS model is deployable on free CPU hardware, but its quality is weaker than the frontier model.
- The heuristic scorer is reproducible and fast, but high-stakes safety evaluation should include human review and stronger LLM-as-judge calibration.
- Sliding-window memory is simple and predictable, but does not provide durable user memory or retrieval.

## Production Improvements

- Add a policy-specific safety classifier and refusal-quality evaluator.
- Add stronger retrieval grounding for factual, policy, legal, and support workflows.
- Add Patronus Lynx-style groundedness verification on Modal.
- Add prompt versioning, eval run history, and per-category regression dashboards.
- Add OpenTelemetry or Langfuse-style traces for production observability.
- Add human review queues for high-risk categories such as medical, legal, financial, privacy, and policy commitments.
- Compare larger OSS deployments on Modal, RunPod, Replicate, or paid Hugging Face hardware.
