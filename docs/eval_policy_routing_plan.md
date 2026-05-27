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
Prompt injection detection -> Prompt Guard-style classifier
Safety/refusal detection -> WildGuard-style classifier
Evidence retrieval -> local RAG first, approved web/API tools second
Hallucination verification -> Patronus Lynx-style groundedness evaluator
LLM judge -> secondary qualitative signal only
```

Why this stack:

```txt
Prompt Guard is focused on injection/jailbreak detection.
WildGuard is a good fit for malicious intent, unsafe responses, and refusal behavior.
RAG/web tools provide evidence that the model itself does not know.
Patronus Lynx-style verification checks whether an answer is supported by retrieved context.
The LLM judge adds explanation, but does not become the source of truth.
```

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
Prompt Guard-style classifier for prompt injection
WildGuard-style classifier for harmful intent, unsafe output, and refusal behavior
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

## Component 4: Claim Verification

For factual answers, verify whether the answer is supported by evidence.

Chosen v1:

```txt
Patronus Lynx-style groundedness evaluation
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
Prompt Guard-style injection classifier
WildGuard-style safety/refusal classifier
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

### Phase 4: Claim Verification

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

### Phase 5: Report Updates

Add report metrics:

```txt
policy_inference_confidence
inferred_expected_action
unsupported_claim_rate
cannot_verify_rate
needs_review_rate
source_coverage_rate
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
