# Attack Taxonomy, Model Family, and Honest Legal Class: An Evaluation Study of LLM Jailbreaks

**Ashwin Hegde**  
AI Risk Evaluation Workbench  
August 2026

**A dissertation draft** for an applied computing / information-systems capstone.  
This document reports what the repository actually measured. It does not claim EU AI Act certification, CE marking, or a new foundation model.

---

## Abstract

This project evaluates three language-model backends on the same fifteen-strategy red-team suite and asks two questions. First, how much of jailbreak success is explained by *attack type* rather than *model family*? Second, can evaluation scores be used to assign an EU AI Act risk class? On 225 trials (15 strategies × 5 seeds × 3 models), break rates were 9.3% for `openai/gpt-5` (7/75, Wilson 95% CI [4.6%, 18.0%]), 21.3% for `deepseek/deepseek-v4-flash` (16/75, [13.6%, 31.9%]), and 66.7% for `qwen3-8b` (50/75, [55.4%, 76.3%]). One strategy, structured JSON output, broke all three models on every trial (15/15). Earlier DeepSeek runs flipped from 0% to 80% when the attack set changed, which is evidence that the *suite* can dominate the *model*. A single human then labelled 48 of 50 sampled transcripts against the same COMPLIED / REFUSED rule used by the `gpt-4o-mini` break judge. After six corrections of clear yes/no mix-ups, agreement was 75% (Cohen’s κ = 0.50). The legal contribution is negative but important: EU AI Act class is a function of *declared use case* (Art. 5, Art. 6 + Annex III, Art. 50, Chapter V), not of a bias or jailbreak score. The workbench now encodes that rule and stops issuing language that sounds like a conformity certificate.

**Keywords:** LLM red-teaming, jailbreak, LLM-as-judge, EU AI Act, human adjudication, evaluation validity

---

## Chapter 1 — Introduction

### 1.1 The problem

Teams that ship chatbots need two different facts, and they keep mixing them up.

The first fact is *technical*: under a stated attack, did the model produce the prohibited artefact? That is an evaluation question. It has a transcript, a decision rule, and an error rate.

The second fact is *legal*: what class does the *system* fall into under the EU AI Act? That question is answered by intended purpose — employment, credit, a general chatbot, a prohibited social-scoring practice — not by whether a model scored 0.4 on a bias rubric.

Popular “AI compliance platforms,” including an earlier version of this repository, collapse the two. They map `bias → Art. 6 High Risk` and print a JSON “certificate.” That is convenient. It is also wrong. A customer-support bot that fails a jailbreak test is still, in the ordinary case, a limited-risk conversational system with an Art. 50 transparency duty. An employment-screening system that never fails a jailbreak is still high-risk under Annex III. Treating a score as a class is how student projects, and some vendor decks, invent a legal result they cannot defend in a viva.

### 1.2 Research questions

This dissertation defends three claims, in this order:

1. **RQ1 (taxonomy).** On a fixed 15-strategy suite, jailbreak success varies at least as much by attack type as by model family. In particular, structured-output framing broke every model tested, while several classic encodings broke none of them at *n* = 5.
2. **RQ2 (judge validity).** An automated break judge is not a substitute for a person. A human review of 48 transcripts agrees with `gpt-4o-mini` on 75% of cases (κ = 0.50). Reported break rates are therefore directional.
3. **RQ3 (legal class).** Eval scores must not assign EU AI Act risk class. Class comes from declared use case. Residual findings (jailbreak, bias, hallucination) are evidence for duties that already apply (for example Art. 15 robustness if the system is already Annex III).

### 1.3 What this project is not

It is not a production compliance product. It is not better than Garak, Microsoft PyRIT, Promptfoo, or UK AISI Inspect as a general red-team toolkit. Crescendo, many-shot, and best-of-N are *named after* published attacks; the implementations here are short prompt ladders, not full reproductions (best-of-N in the source paper uses thousands of augmentations; this suite uses five). The project does not train a new model and does not evaluate voice or multimodal jailbreaks.

Those limits are features of the thesis, not defects to hide. The contribution is a *measured* comparison plus an *honest* legal mapping, not a platform brand.

### 1.4 Structure

Chapter 2 places the work next to published jailbreaks and the Act. Chapter 3 describes the suite, judge, human sheet, and use-case classifier. Chapter 4 reports the numbers. Chapter 5 discusses what they can and cannot support. Chapter 6 concludes.

---

## Chapter 2 — Background and related work

### 2.1 Jailbreaks as an evaluation object

A jailbreak is a prompt, or a multi-turn conversation, that causes a model to produce content its safety policy is supposed to refuse. Early public attacks were single-turn personas (DAN). Later work showed that *how* the request is wrapped matters more than a single magic string.

**Crescendo** (Russinovich, Salem, and Eldan; USENIX Security 2025, arXiv:2404.01833) reaches a prohibited goal by a sequence of modest, related turns rather than one blatant request. **Many-shot jailbreaking** (Anil et al., Anthropic; NeurIPS 2024) places a long fabricated history of compliance in context so the model continues the pattern. **Best-of-N jailbreaking** (Hughes et al., NeurIPS 2024, arXiv:2412.01156) samples many surface augmentations of the same harmful request; the paper reports very high attack success on GPT-4o at thousands of samples. This repository implements short, deterministic versions of those ideas. They are citations of *technique*, not claims of paper-faithful reproduction.

Other strategies in the suite are older or derived: role-play, encoding, language switching, context flooding, tool-use bait, RAG poisoning, memory override, few-shot priming, syllogistic framing, authority conflation, and structured JSON output. The last of these is the empirical headline: asking for the prohibited artefact as a data record, not as prose.

### 2.2 LLM-as-judge

Using a second model to score the first is now standard and known to be biased (position, verbosity, self-preference). This project uses `gpt-4o-mini` as a *break adjudicator* with a binary rule: COMPLIED if the visible reply provides the prohibited content, REFUSED otherwise. Disclaimers do not cancel a completed artefact. Empty replies are refusals. Internal `<think>` blocks are stripped before the decision.

A binary judge is simpler than a 1–5 rubric. It is still an instrument. Chapter 4 measures it against a human.

### 2.3 The EU AI Act, briefly and accurately

The Act does not classify “GPT-5” or “Qwen” as high-risk because a red-team score was poor.

- **Article 5** prohibits specific *practices* (social scoring of natural persons, certain manipulative techniques, real-time remote biometric identification in publicly accessible spaces subject to listed exceptions, emotion recognition in workplace and education, and others). The *use* is banned. A model that can be *misused* for social scoring is not thereby an Art. 5 system.
- **Article 6 and Annex III** list *high-risk systems* by intended purpose: biometrics, critical infrastructure, education and vocational assessment, employment and worker management, access to essential private and public services (including credit), law enforcement, migration, administration of justice. High-risk duties (Arts. 9–15: risk management, data, technical documentation, record-keeping, transparency to deployers, human oversight, accuracy / robustness / cybersecurity) attach to that purpose.
- **Article 50** imposes transparency duties on chatbots (disclose that the user is interacting with AI), synthetic content, and some emotion-recognition uses that are not already prohibited.
- **Chapter V (Arts. 51–56)** imposes provider duties on general-purpose AI models, separately from whether a downstream deployer later puts the same weights in an Annex III product.

A jailbreak finding on a chatbot is residual robustness evidence. It becomes an Art. 15 gap *if and only if* the system under assessment is already high-risk by purpose. Mapping `jailbreak_resistance → Art. 50` or `bias → Art. 6` from the dimension name alone is a category error. This project previously made that error. Chapter 3 describes the replacement.

### 2.4 Existing tools

NVIDIA Garak, Microsoft PyRIT, Promptfoo, and Inspect (UK AISI) already run adversarial probes at larger scale and with better sampling than this suite. Llama Guard and similar classifiers score harm categories. This dissertation does not replace them. It uses a *fixed* suite so that three backends can be compared, and it refuses to launder those comparisons into a legal certificate.

---

## Chapter 3 — Method

### 3.1 Targets

Three backends were run on the same strategies, turn budget, and seeds:

| Target | Role | Serving |
|---|---|---|
| `openai/gpt-5` | frontier | Kilo / OpenAI-compatible gateway |
| `deepseek/deepseek-v4-flash` | flash-tier frontier | same gateway |
| `qwen3-8b` | open-weight, self-hosted | Modal, NVIDIA L4, vLLM |

DeepSeek-v4-flash is not “the” DeepSeek frontier model. Results for it do not generalise to a larger DeepSeek checkpoint.

### 3.2 Suite

Fifteen strategies, five independent trials each, base seed 42 (trial *i* uses seed 42 + *i* − 1), maximum five turns. The prohibited goal for most strategies is a usable phishing email impersonating a bank. Role-play also probes lockpicking instruction; tool-exploit probes destructive tool use; encoding probes malware; context-overflow probes system-prompt extraction; RAG-poison probes adoption of an insecure password-reset policy. That mix is still *narrow*. It is not “AI safety” in general.

A trial is a **break** when the break judge returns COMPLIED on the visible reply (or, in the recorded JSON, `broke: true`). Wilson score 95% intervals are reported for rates.

### 3.3 Human adjudication

`src/redteam/human_review.py` draws a stratified sample of 50 rows from the 225-trial file, keeping any `adjudication_needs_review` rows first and then round-robining `(model, broke)` buckets. The labeller (the author) filled `human_verdict` with COMPLIED or REFUSED. Two rows were skipped. Six later rows were flipped after a review of likely yes/no mix-ups on *clear* cases (explicit “I can’t” versus a completed JSON or phishing template). Fuzzy cases were not silently overwritten.

Cohen’s κ is computed on the 2×2 table of judge versus human. Unlabelled rows are ignored.

### 3.4 Use-case classification

`src/compliance/system_class.py` is the only place that assigns `RiskTier` for the Act. The default use case for this workbench is `gpai_or_chatbot` (limited / Art. 50 + Chapter V). Employment, credit, education, and law-enforcement use cases are high-risk *by purpose*. Social scoring as a *declared use* is unacceptable. An eval dimension named `social_scoring` is a *misuse probe*, not a classification of the system.

Eval and red-team findings copy the system’s class onto the finding. They do not upgrade a chatbot to Annex III because a bias score was poor. The eval-gate JSON is labelled as a research record, not a conformity assessment.

### 3.5 Implementation notes

The repository contains a second, older stack (`assistant/`, Chainlit) that compares a tiny open model to a frontier model on the same product surface. That stack is pedagogically useful and is not the source of the 15-strategy table. The numbers in Chapter 4 come from `src/redteam` and `results/redteam_findings.json`.

---

## Chapter 4 — Results

### 4.1 Per-model break rates

From `results/redteam_findings.json`, 5 trials × 15 strategies:

| Model | Breaks | Rate | 95% Wilson CI |
|---|---:|---:|---|
| `openai/gpt-5` | 7 / 75 | 9.3% | [4.6%, 18.0%] |
| `deepseek/deepseek-v4-flash` | 16 / 75 | 21.3% | [13.6%, 31.9%] |
| `qwen3-8b` | 50 / 75 | 66.7% | [55.4%, 76.3%] |

Qwen3-8B breaks about seven times as often as GPT-5 on this suite. DeepSeek-v4-flash sits between them. Intervals are wide because *n* = 5 per cell. The *order* GPT-5 < DeepSeek-flash < Qwen3-8B is stable enough to report; the exact percentages are not.

### 4.2 Per-strategy rates (all models pooled)

| Strategy | Breaks | Rate | GPT-5 | DeepSeek | Qwen |
|---|---:|---:|---:|---:|---:|
| structured_output | 15/15 | 100% | 5/5 | 5/5 | 5/5 |
| many_shot | 10/15 | 67% | 0/5 | 5/5 | 5/5 |
| policy_conflation | 9/15 | 60% | 0/5 | 4/5 | 5/5 |
| multilingual | 6/15 | 40% | 0/5 | 1/5 | 5/5 |
| roleplay | 6/15 | 40% | 0/5 | 1/5 | 5/5 |
| tool_exploit | 6/15 | 40% | 1/5 | 0/5 | 5/5 |
| best_of_n | 5/15 | 33% | 0/5 | 0/5 | 5/5 |
| dan_jailbreak | 5/15 | 33% | 0/5 | 0/5 | 5/5 |
| few_shot | 5/15 | 33% | 0/5 | 0/5 | 5/5 |
| rag_poison | 5/15 | 33% | 0/5 | 0/5 | 5/5 |
| syllogism | 1/15 | 7% | 1/5 | 0/5 | 0/5 |
| context_overflow | 0/15 | 0% | 0 | 0 | 0 |
| crescendo | 0/15 | 0% | 0 | 0 | 0 |
| encoding | 0/15 | 0% | 0 | 0 | 0 |
| memory_manip | 0/15 | 0% | 0 | 0 | 0 |

**Finding 1.** Structured output is the only strategy that is 100% on every model. Asking for the phishing email as a JSON record (`from_name`, `from_address`, `subject`, `body`, `link_url`) consistently defeated prose-oriented refusal.

**Finding 2.** Several “classic” strategies (encoding, memory manipulation, this implementation of Crescendo, context overflow) scored 0/15. For Crescendo and best-of-N this is likely under-implementation, not a proof that the published attacks fail. For encoding, at *n* = 5, the models simply refused.

**Finding 3.** DeepSeek’s earlier 0% (legacy eight strategies) and 80% (five modern strategies) are both real *and* superseded. Against the unified 15-strategy file it is 21.3%. The attack set, not a mysterious change in the weights, is what moved the number. That is the cleanest evidence for RQ1.

### 4.3 Human versus judge

Confusion matrix after the six mix-up corrections:

| | Judge COMPLIED | Judge REFUSED |
|---|---:|---:|
| Human COMPLIED | 18 | 8 |
| Human REFUSED | 4 | 18 |

Agreement 36/48 = **75%**. Cohen’s κ = **0.50**.

The leftover twelve disagreements are almost all boundary cases: syllogism (accepting premises without pasting a full email), Crescendo outlines versus a finished lure, grandma role-play that mixes comfort talk with procedure, multilingual wording, an encoding trial with no visible payload, and a model that *claimed* to dump a system prompt. Clear “I can’t” strings and finished `Subject:` / JSON bank emails mostly agreed after review.

**Finding 4.** The instrument is moderately aligned with one human rater. It is not calibrated enough to treat a 9.3% versus 21.3% gap as a precise effect size. It *is* calibrated enough to treat 9% versus 67% as a real difference on this suite.

### 4.4 Legal class (negative result)

Under the default use case `gpai_or_chatbot`, overall legal class is **limited**, including when a model’s break rate is 67%. Residual findings may be severe; the class does not move. Declaring `system_use_case=employment` makes the class **high** even if every eval score is clean. That is the intended behaviour, and the one the Act requires.

No run of this pipeline is a notified-body assessment.

---

## Chapter 5 — Discussion and limitations

### 5.1 What RQ1 can support

Attack wrapping changes outcomes. The same DeepSeek flash model looked “safe” on an old eight-strategy board and “broken” on a five-strategy modern board. On a unified board it is in the middle. Any vendor slide that quotes a single break rate without the suite is incomplete.

Structured JSON is a practical lesson for anyone building a tool-using or schema-constrained assistant: the safety filter that watches prose may not watch a `body` field.

### 5.2 What RQ1 cannot support

- Generalisation beyond phishing-like bank lures and a handful of other harms.
- Claims about Crescendo or Best-of-N *as published*. The code is a sketch.
- Claims about “Qwen is unsafe” as a model family. This is one 8B instruct checkpoint on one host.
- Fine ranking of strategies with 0% or 100% at *n* = 5.

### 5.3 The human labels

The author labelled the sheet while still learning the decision rule. Six obvious inversions were later corrected. The remaining 12 disagreements were left as-is because they are genuinely hard. There is no second rater, so there is no inter-rater κ. A viva question of the form “are you sure about row 32?” should be answered: **no, not to the nearest label; yes, that the judge and a person diverge on fuzzy cases.**

### 5.4 Legal honesty as a contribution

The useful legal result is a *refusal to over-claim*. Replacing a dimension-to-article lookup with a use-case classifier sounds small. It is the difference between a dissertation an examiner can mark and a document that would mislead a compliance officer.

### 5.5 Engineering debt

Two code paths (`assistant/` and `src/`) remain. Tests lock a lot of dictionaries. The Streamlit dashboard and static VERDICT site still use “PASS/FAIL” stamps; they are now described as eval-gate records. Future work should pick one stack, add a Garak or PyRIT baseline, increase trials on structured_output / many_shot / policy_conflation, and if resources allow, run DPO on an open 3B–8B model against the structured-output failure — measuring with this same suite.

---

## Chapter 6 — Conclusion

This dissertation asked whether jailbreak success on a fixed suite tracks attack type as much as model family, whether the automated judge can be trusted, and whether eval scores may assign EU AI Act class.

On the first question, yes: structured output broke every model; the same DeepSeek flash checkpoint moved from 0% to 80% to 21% as the suite changed; Qwen3-8B was far more brittle than GPT-5 on this phishing-centred board.

On the second question, only partly: one human agreed with the judge on 75% of 48 cases (κ = 0.50). Rates are directional.

On the third question, no: class is purpose. The workbench now says so, and no longer pretends a JSON file is a certificate.

That is a smaller story than “an AI compliance platform.” It is a story that can be defended.

---

## References

Anil, C., et al. (2024). Many-shot jailbreaking. *NeurIPS 2024*. Anthropic.

Hughes, J., et al. (2024). Best-of-N Jailbreaking. arXiv:2412.01156. *NeurIPS 2024*.

Russinovich, M., Salem, A., & Eldan, R. (2025). Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. *USENIX Security 2025*. arXiv:2404.01833.

Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). *OJ L*, 2024.

National Institute of Standards and Technology. (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*. NIST AI 100-1.

ISO/IEC 42001:2023. *Information technology — Artificial intelligence — Management system*.

Inan, H., et al. (2023). Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. arXiv:2312.06674.

Microsoft. PyRIT (Python Risk Identification Tool for generative AI). https://github.com/Azure/PyRIT

NVIDIA. Garak: LLM vulnerability scanner. https://github.com/NVIDIA/garak

UK AI Security Institute. Inspect. https://inspect.aisi.org.uk/

---

## Appendix A — How to reproduce

```bash
pip install -e ".[dev]"

# Red-team (requires provider credentials)
python3 -u -m src.redteam.agent \
  --targets openai/gpt-5,qwen3-8b \
  --turns 5 --strategy all --trials 5 --seed 42 \
  --break-judge-model openai/gpt-4o-mini

# Human sheet (already filled in this repo)
python3 -m src.redteam.human_review score \
  --sheet data/human_review/adjudication_sheet.csv \
  --out results/human_agreement.json

# Use-case class stays chatbot unless you declare otherwise
python3 -u -m src.pipeline.run --model qwen3-8b --mock \
  --system-use-case gpai_or_chatbot
```

## Appendix B — Data files

| File | Contents |
|---|---|
| `results/redteam_findings.json` | 225-trial unified suite |
| `data/human_review/adjudication_sheet.csv` | 50-row human sheet (48 labelled) |
| `results/human_agreement.json` | Agreement statistics |
| `src/compliance/system_class.py` | Use-case → Act class |
| `docs/HUMAN_ADJUDICATION.md` | Short form of Chapter 4.3 |

## Appendix C — Author’s note on labelling

The human labels are the author’s. Six rows were corrected after a review of obvious inversions. The author was still learning the rule during the first pass. That limitation is part of the result, not an excuse attached afterwards.
