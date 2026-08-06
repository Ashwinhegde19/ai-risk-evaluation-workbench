---
title: VERDICT — Red-Team Evidence Console
emoji: ⚖️
colorFrom: red
colorTo: gray
sdk: static
pinned: false
---

# AI Risk Evaluation Workbench

LLM red-team + compliance platform. Runs multi-turn adversarial attacks, scores with a calibrated multi-model judge, maps findings to EU AI Act / NIST AI RMF / ISO 42001, emits audit-ready reports.

**15 attack strategies** (8 legacy + 7 from 2024–2026 research) against `openai/gpt-5`, `anthropic/claude-opus-4.1`, `google/gemini-2.5-pro`, `deepseek/deepseek-v4-flash`, `qwen3-8b`.

## The headline

**The attack taxonomy matters more than the model.** deepseek-v4-flash scored 0% under the legacy 8 strategies and 80% under 5 modern ones:

| Run | Strategies | Break rate | 95% Wilson CI |
|---|---|---:|---:|
| legacy 8 | dan_jailbreak, roleplay, encoding, multilingual, context_overflow, tool_exploit, rag_poison, memory_manip | 0.0% (0/40) | [0.0%, 8.8%] |
| modern 5 | crescendo, many_shot, policy_conflation, structured_output, best_of_n | 80.0% (20/25) | [60.9%, 91.1%] |

Why: legacy strategies are single-shot static prompts that frontier models are trained to refuse. Modern strategies target instruction-following, multi-turn escalation, and long-context imitation — pathways alignment still leaves open. Sources: Crescendo (USENIX Security 2025), Many-shot Jailbreaking (Anthropic, NeurIPS 2024), Best-of-N (Hughes et al., NeurIPS 2024).

## Results (live)

Per-model break rates from `results/redteam_findings.json`:

| Model | Break rate | Top breaking strategies |
|---|---:|---|
| `openai/gpt-5` | *running* | structured_output 5/5 so far |
| `qwen3-8b` | *running* | — |
| `deepseek/deepseek-v4-flash` | 80.0% (20/25) | crescendo 5/5, many_shot 5/5, policy_conflation 5/5, structured_output 5/5 |

Per-strategy breakdown (deepseek-v4-flash, modern 5):

| Strategy | Breaks | Rate |
|---|---:|---:|
| `crescendo` | 5/5 | 100% |
| `many_shot` | 5/5 | 100% |
| `policy_conflation` | 5/5 | 100% |
| `structured_output` | 5/5 | 100% |
| `best_of_n` | 0/5 | 0% (under-sampled; paper uses 10k augments, we use 5) |

## The 15 strategies

| # | Strategy | Technique | Source |
|---|---|---|---|
| 1 | `dan_jailbreak` | DAN persona chains | legacy |
| 2 | `roleplay` | Role-play escalation | legacy |
| 3 | `encoding` | Base64 / ROT13 / leetspeak | legacy |
| 4 | `multilingual` | Language switching | legacy |
| 5 | `context_overflow` | Context flooding | legacy |
| 6 | `tool_exploit` | Dangerous tool calls | legacy |
| 7 | `rag_poison` | Malicious document injection | legacy |
| 8 | `memory_manip` | Memory exploitation | legacy |
| 9 | `few_shot` | In-context priming | derived |
| 10 | `syllogism` | Logical-framing coercion | derived |
| 11 | `policy_conflation` | Authority conflation | derived |
| 12 | `structured_output` | JSON/data-export circumvention | derived |
| 13 | `crescendo` | Multi-turn gradual escalation | Crescendo, USENIX Sec 2025 |
| 14 | `many_shot` | Long-context fabricated history | Many-shot, Anthropic NeurIPS 2024 |
| 15 | `best_of_n` | Randomized augmentation sampling | Best-of-N, Hughes NeurIPS 2024 |

## Reproduce

```bash
pip install -e ".[dev]"

# 15-strategy red-team against gpt-5 + qwen3-8b
python3 -u -m src.redteam.agent \
  --targets openai/gpt-5,qwen3-8b \
  --turns 5 --strategy all --trials 5 --seed 42 \
  --break-judge-model openai/gpt-4o-mini

# Passive eval + compliance report
python3 -u -m src.pipeline.run \
  --targets openai/gpt-5,anthropic/claude-opus-4.1,google/gemini-2.5-pro,qwen3-8b \
  --suite full --max-redteam-turns 5 --report-dir results

python3 -u -m src.reports.generate --format all --framework all --deployment-context high
```

## Honest limitations

- **n = 5 trials per cell.** Wilson intervals are wide (e.g. deepseek modern CI [60.9%, 91.1%]). Per-strategy rates are directional, not precise.
- **Flash-tier results don't generalize to frontier.** deepseek-v4-flash is a fast inference tier; gpt-5 and qwen3-8b are still running.
- **best_of_n is under-sampled.** Paper reports 89% on gpt-4o with 10,000 augmentations; we run 5. Configuration gap, not strategy failure.
- **Adjudicator noise.** The break judge (`openai/gpt-4o-mini`) misread ~4% of clean refusals in cross-checks.

## Architecture & deeper docs

- **Live site:** [HF Space](https://ashwinhegde19-ai-risk-evaluation-workbench.static.hf.space)
- **Strategy source:** `src/redteam/strategies/` (15 strategies, each with a docstring and research citation)
- **Compliance mapping:** `src/compliance/redteam_mapping.py` (strategy → EU/NIST/ISO control)
- **Results JSONs:** `results/redteam_findings*.json` (raw trial-level data)
- **Full technical docs:** see `docs/ARCHITECTURE.md` and `docs/EVALUATION.md` in the repo

Frontier models route through the Kilo gateway; qwen3-8b is self-hosted on Modal (NVIDIA L4). Judge ensemble: GPT-4o + Claude Sonnet + Gemini Pro with median aggregation. CI/CD gates on every push.
