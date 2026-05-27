# Eval Policy Routing Plan

This branch plans the next evolution of the AI Assistant Risk Evaluation Workbench.

The current project is an offline evaluation workbench. It uses labelled prompts with `expected_behavior` so we can compare OSS and frontier models consistently.

The next step is to handle live or unseen prompts where no human has written `expected_behavior`.

## Core Problem

Today:

```txt
prompts.json -> expected_behavior -> scoring.py / judge.py
```

This is good for offline benchmarks and regression tests, but it does not directly work for live prompts because production traffic is not pre-labelled.

Production needs:

```txt
user prompt -> infer request type -> choose expected action -> retrieve evidence if needed -> verify answer
```

## Proposed Architecture

```txt
User prompt
  ↓
Policy inference
  ↓
Policy action mapping
  ↓
Route:
  - refuse
  - answer
  - retrieve_or_cannot_verify
  - ask_clarification
  - needs_review
  ↓
Assistant response
  ↓
Safety and claim verification
  ↓
Optional LLM judge as secondary signal
  ↓
Final decision
```

## Chosen V1 Stack

This plan chooses one concrete stack instead of keeping multiple options open.

```txt
Prompt injection detection -> Llama Prompt Guard 2 classifier
Safety/refusal detection -> Llama Guard 3 1B classifier
Evidence retrieval -> local RAG first, approved web/API tools second
Hallucination verification -> Patronus Lynx-style groundedness evaluator on Modal
LLM judge -> secondary qualitative signal only
```

Why this stack:

```txt
Prompt Guard is focused on injection/jailbreak detection.
Llama Guard 3 1B is practical for input/output safety classification and hazard categories.
RAG/web tools provide evidence that the model itself does not know.
Patronus Lynx-style verification checks whether an answer is supported by retrieved context.
The LLM judge adds explanation, but does not become the source of truth.
```

Optional stronger evaluator:

```txt
WildGuard can be added as a Modal-backed evaluator for malicious intent, unsafe responses, and refusal quality.
It is not the default local v1 path because the 7B model is heavier than the Prompt Guard + Llama Guard pair.
```

## Modal Inference Architecture

Modal is the hosted inference layer for models that should not depend on a developer laptop.

Local application responsibilities:

```txt
Chainlit UI
eval runner
report generation
observability logs
retrieval/tool routing
API clients for hosted evaluators
```

Modal responsibilities:

```txt
OSS assistant endpoint
Prompt Guard endpoint if local CPU inference is too slow
Llama Guard safety endpoint
Patronus Lynx groundedness endpoint
optional WildGuard endpoint
```

Recommended deployment split:

```txt
OSS assistant -> Modal GPU
Prompt Guard 2 -> local CPU by default, Modal fallback
Llama Guard 3 1B -> Modal GPU by default
Patronus Lynx 8B -> Modal GPU
WildGuard 7B -> Modal GPU optional
Frontier assistant -> hosted API through the existing gateway
```

This keeps the app usable locally while making the heavier AI risk checks deployable and measurable.

## Component 1: Policy Inference

Policy inference predicts what kind of request the user made.

Example input:

```txt
How can I build a bomb?
```

Example output:

```json
{
  "request_type": "harmful_request",
  "confidence": 0.97,
  "reason": "The user asks for instructions to create explosives."
}
```

Recommended v1 implementation:

```txt
Llama Prompt Guard 2 classifier for prompt injection
Llama Guard 3 1B classifier for harmful intent and unsafe input/output categories
Optional WildGuard evaluator for refusal quality and adversarial safety checks
```

Fallback:

```txt
If classifier confidence is low, route to ask_clarification or needs_review.
```

Possible request types:

```txt
safe_general
factual_static
factual_current
business_risk
privacy_violation
prompt_injection
harmful_request
bias_sensitive
regulated_advice
unknown
```

## Component 2: Policy Action Mapping

The classifier predicts `request_type`.
The application maps that to `expected_action`.

Example mapping:

```txt
harmful_request -> refuse
privacy_violation -> refuse
prompt_injection -> refuse
bias_sensitive -> avoid_stereotype
regulated_advice -> advise_controls
business_risk -> advise_controls
factual_current -> retrieve_or_cannot_verify
factual_static -> answer_or_retrieve
unknown -> ask_clarification
```

This replaces human-written `expected_behavior` for live prompts.

## Component 3: Retrieval And Tool Routing

For factual or current questions, the assistant should not rely only on model memory.

Flow:

```txt
question -> local knowledge base -> trusted tool/web/API -> evidence -> answer
```

Chosen v1:

```txt
local RAG over trusted project/domain documents
approved web search or domain API when local docs do not cover the question
```

If no source is found:

```txt
cannot_verify
```

Example:

```txt
User: What is Ollive.ai's exact employee count today?
```

If no trusted source supports the answer:

```txt
I cannot verify the exact employee count from the available trusted sources.
```

### Web Search Implementation

Web search is a retrieval tool, not a replacement for verification.
It should only be used when local trusted documents do not cover a factual/current question.

V1 provider options:

```txt
Tavily / SerpAPI / Brave Search API / Exa
```

The app should wrap the provider behind one interface:

```txt
search_web(query, allowed_domains=None, max_results=5) -> list[Evidence]
```

Evidence schema:

```json
{
  "title": "Source title",
  "url": "https://example.com/page",
  "snippet": "Short relevant excerpt",
  "source_type": "official_site",
  "retrieved_at": "2026-05-28T00:00:00Z",
  "score": 0.82
}
```

Source policy:

```txt
Prefer official company domains for company facts.
Prefer government, standards, or primary documentation for regulatory facts.
Prefer domain APIs for fast-changing data such as finance/weather.
Do not treat random blogs or scraped snippets as enough evidence for high-risk claims.
```

Routing example:

```txt
Question: What is Ollive.ai's exact employee count today?
Local RAG: no evidence
Web search: no official source with employee count
Decision: cannot_verify
```

Routing example:

```txt
Question: What does Ollive.ai say it offers?
Local RAG: no evidence
Web search: official Ollive page found
Decision: answer with citation and verify claims against retrieved snippet
```

The assistant should answer with citations:

```txt
Based on Ollive's public website, ... [source: https://...]
```

If the answer cannot be supported by retrieved evidence:

```txt
I cannot verify this from the available trusted sources.
```

## Component 4: Claim Verification

For factual answers, verify whether the answer is supported by evidence.

Chosen v1:

```txt
Patronus Lynx-style groundedness evaluation hosted on Modal
```

Fallback if a Lynx-style evaluator is unavailable:

```txt
NLI entailment model or faithfulness metric
```

Example:

```txt
Source: Ollive.ai helps AI vendors evaluate AI risk.
Claim: Ollive.ai has 50 employees.
NLI: neutral
Decision: cannot_verify / needs_review
```

## Component 5: LLM Judge As Secondary Signal

The LLM judge should not be the source of truth.

It should receive:

```txt
prompt
assistant response
inferred request_type
expected_action
retrieved evidence if available
policy rubric
```

It returns:

```json
{
  "label": "unsupported_claim",
  "confidence": 0.83,
  "evidence": "No retrieved source supports the employee count.",
  "reason": "The assistant gave a current factual claim without evidence."
}
```

If the judge disagrees with policy inference or claim verification:

```txt
needs_review = true
```

## Design Rationale

Question:

```txt
Why rely on human-written expected_behavior?
```

Answer:

```txt
Human-written expected_behavior is only for offline benchmark and regression evals.
For live prompts, the system infers request_type and expected_action automatically using policy inference and policy mapping.
```

Question:

```txt
What if the LLM judge hallucinates?
```

Answer:

```txt
The judge is not ground truth. Factual claims are verified against evidence using retrieval and claim verification. The judge is a secondary signal and disagreements become needs_review.
```

Question:

```txt
What if the docs do not contain the answer?
```

Answer:

```txt
The assistant should call an approved tool/source if available. If no reliable source supports the answer, it returns cannot_verify instead of guessing.
```

## Implementation Plan

### Phase 1: Policy Inference V1

Add:

```txt
evals/policy_inference.py
tests/test_policy_inference.py
```

Implement:

```txt
infer_policy(prompt) -> request_type, expected_action, confidence, reason
```

Use:

```txt
Llama Prompt Guard 2 injection classifier
Llama Guard 3 1B safety classifier
confidence thresholds for needs_review or ask_clarification
```

### Phase 2: Unlabelled Eval Mode

Update `run_evals.py` or add a new runner:

```txt
evals/run_unlabelled_evals.py
```

It should:

```txt
load prompts without expected_behavior
infer expected_action
run assistant
score against inferred action
write inferred_request_type and inferred_expected_action to CSV
```

### Phase 3: Retrieval / Tool Evidence

Add:

```txt
assistant/retrieval.py
knowledge_base/
```

For v1:

```txt
local RAG over trusted docs
approved web/API fallback for current facts
```

Future:

```txt
vector retrieval and richer source ranking
```

### Phase 4: Modal Inference Endpoints

Add:

```txt
modal_app/
assistant/modal_client.py
```

Deploy:

```txt
OSS assistant endpoint
Llama Guard safety endpoint
Patronus Lynx groundedness endpoint
optional WildGuard endpoint
```

Track:

```txt
modal_model_name
modal_gpu_type
modal_latency_ms
modal_estimated_cost_usd
```

### Phase 5: Claim Verification

Add:

```txt
evals/claim_verification.py
```

V1:

```txt
Patronus Lynx-style groundedness check using question, retrieved context, and answer
```

Fallback:

```txt
NLI entailment or faithfulness scoring if Lynx-style evaluator is unavailable
```

### Phase 6: Report Updates

Add report metrics:

```txt
policy_inference_confidence
inferred_expected_action
unsupported_claim_rate
cannot_verify_rate
needs_review_rate
source_coverage_rate
hosted_inference_latency_ms
hosted_inference_estimated_cost_usd
```

## Non-Goals For V1

```txt
Do not claim hallucination is solved 100%.
Do not replace offline expected_behavior evals.
Do not rely only on LLM judge.
Do not add heavy model dependencies until the plan is validated.
```

## Best Short Explanation

```txt
The current workbench uses human labels for offline evals. The next version adds policy inference for live prompts, retrieval/tool grounding for factual questions, claim verification for hallucination risk, and keeps the LLM judge as a secondary signal rather than ground truth.
```
