# AI Engineering — Chip Huyen

| | |
|---|---|
| **Author** | Chip Huyen |
| **Published** | 2024 (O'Reilly) |
| **Status** | Read |
| **Date read** | _<!-- fill in -->_ |
| **Rating** | _<!-- /5 -->_ |
| **Tags** | `llm` `foundation-models` `ml-systems` `rag` `evaluation` |

---

## One-line summary

How to build *applications on top of* foundation models — the engineering discipline that emerged once you no longer need to train a model from scratch. The whole book is organized around adapting an existing model to your needs, cheapest technique first.

## Why I read it

_<!-- your reason — what were you hoping to get out of it? -->_

---

## The core mental model

> **AI Engineering = adapting foundation models, not training them.**

Traditional ML engineering: collect data → train a model → deploy. AI engineering inverts this: a capable model already exists; your job is **adaptation** and the **system around the model**. The barrier to entry collapsed, so the bottleneck moved from *modeling* to *evaluation, data, and product*.

The book pushes one recurring principle — **climb the adaptation ladder in cost order, stop at the first rung that works:**

```
Prompt engineering   →   RAG   →   Finetuning
  (cheapest, fastest)        (give the model     (change the model's
   change the input)          context/knowledge)   behavior/form — costly)
```

Rough rule the book repeats: **RAG when the model lacks *information*; finetuning when it lacks the right *behavior or format*.** Try prompting first, always.

---

## Chapter-by-chapter notes

### 1. Introduction to Building AI Applications with Foundation Models
- "Language model → LLM → foundation model" — foundation models are multimodal and general-purpose; this generality is what made application-building a distinct discipline.
- Common use case buckets: coding, image/video production, writing, education, conversational bots, information aggregation, data organization, workflow automation.
- **Plan before you build:** decide whether AI is even the right tool, set a clear success metric, and weigh the cost of *being wrong* (high-stakes vs. low-stakes apps demand very different rigor).
- The AI engineering stack has three layers: **application development** (prompting, context, evaluation, UX), **model development** (finetuning, dataset eng, inference optimization), and **infrastructure**.

### 2. Understanding Foundation Models
- **Training data** decides the ceiling — a model is only as good (and as biased / language-skewed) as its corpus.
- **Modeling:** transformer architecture, model size, and **scaling laws** (performance scales predictably with compute/data/params). Bigger isn't always the answer — data quality matters.
- **Post-training** is what makes a raw model usable:
  - *Supervised finetuning (SFT)* → teaches it to follow instructions.
  - *Preference finetuning (RLHF / DPO)* → aligns outputs with human preference.
- **Sampling** is the source of both magic and pain:
  - Temperature, top-k, top-p control randomness.
  - The model is **probabilistic** — same input can give different outputs. This *is* the creativity, and also the inconsistency.
  - **Hallucination**, two proposed causes: (1) the model can't separate given facts from its own generations; (2) mismatch between its internal knowledge and the training labels.
  - Structured outputs (JSON, etc.) constrain sampling to a schema.

### 3. Evaluation Methodology
- The book's thesis-within-a-thesis: **evaluation is the hardest, most under-invested part of AI engineering.** Open-ended outputs have no single right answer.
- Language-modeling metrics: **perplexity / cross-entropy** — useful signal, but don't measure task usefulness.
- Exact evaluation (functional correctness, similarity to reference) vs. **subjective evaluation**.
- **AI as a judge:** use a strong model to grade outputs — scalable but biased: *position bias*, *verbosity bias* (prefers longer answers), *self-bias* (prefers its own style).
- **Comparative evaluation:** rank models head-to-head (the basis of leaderboards like Chatbot Arena).

### 4. Evaluate AI Systems
- Evaluation criteria to define up front: **domain-specific capability, generation quality (factual consistency, safety), instruction-following, cost, and latency.**
- **Model selection:** build vs. buy; open-weight (self-host, control, privacy) vs. commercial API (easy, capable, but lock-in + data concerns). Decision hinges on your eval criteria, not hype.
- **Build an evaluation pipeline** — don't eyeball it. Define metrics, curate an eval set, automate, and keep it versioned.

### 5. Prompt Engineering
- The cheapest lever. Best practices: be explicit, give examples (few-shot), break tasks into steps (chain-of-thought), provide a persona/role, specify output format.
- Context construction matters as much as the instruction.
- **Defensive prompt engineering** — treat prompts as an attack surface:
  - *Prompt injection*, *jailbreaking*, *information / prompt extraction*.
  - Mitigations: input/output guardrails, system-prompt isolation, least-privilege tool access.

### 6. RAG and Agents
- **RAG (Retrieval-Augmented Generation):** retrieve relevant context, stuff it into the prompt. Solves the "model lacks information / context window too small" problem.
  - Retrieval algorithms: **term-based** (BM25/keyword — cheap, exact) vs. **embedding-based** (semantic, vector search). Often hybrid.
  - RAG quality is *retrieval* quality — garbage retrieved, garbage generated.
- **Agents:** a model that can **plan** and **use tools** in a loop to accomplish multi-step tasks.
  - Components: tools (read/write actions), planning, memory.
  - Failure modes compound — each step's error rate multiplies over a long trajectory. Agents are powerful but fragile; high-stakes actions need guardrails.
- **Memory** lets agents/chatbots carry context across steps and sessions.

### 7. Finetuning
- When prompting + RAG aren't enough — and only then (it's the most expensive, most operationally heavy rung).
- Use it to change **behavior/form/style**, not to inject facts (that's RAG's job).
- **Memory is the bottleneck** — full finetuning needs huge GPU memory.
- **PEFT (Parameter-Efficient FineTuning)**, esp. **LoRA**: freeze the base model, train small adapter matrices. Dramatically cheaper, composable.
- **Model merging** — combine multiple finetuned models.

### 8. Dataset Engineering
- "Data is the moat." Quality > quantity, but you need enough **coverage**.
- Three axes: **quality, coverage (diversity), quantity.**
- Data acquisition, **annotation** (often the real cost), **augmentation & synthesis** (use models to generate training data), and **processing** (dedup, clean, format).

### 9. Inference Optimization
- Make it fast and cheap in production.
- Know whether you're **compute-bound vs. memory-bound**, and **online (latency-sensitive) vs. batch (throughput-sensitive)**.
- Latency metrics that matter: **TTFT** (time to first token), **TPOT** (time per output token), **throughput**.
- **Model-level:** quantization (fewer bits), distillation (small model learns from big), better attention.
- **Service-level:** batching, **prompt caching**, parallelism.

### 10. AI Engineering Architecture & User Feedback
- Architecture grows in layers — start simple, add only when needed: model API → **context construction** → **guardrails** (input/output) → **model router & gateway** → **caching** → agent patterns → **observability/monitoring**.
- **User feedback is your real-world eval set.** Design the product to capture it (explicit thumbs up/down *and* implicit signals like regenerate, copy, edit, abandon).
- Feedback has biases and limitations — design for them; don't treat thumbs-up as ground truth.

---

## Key concepts / glossary

| Term | Quick definition |
|---|---|
| **Foundation model** | Large, general-purpose, often multimodal model usable across many tasks. |
| **Adaptation ladder** | Prompting → RAG → Finetuning, in increasing cost. |
| **SFT / RLHF / DPO** | Post-training steps: instruction-following, then preference alignment. |
| **Sampling (temp/top-k/top-p)** | Knobs controlling output randomness. |
| **Hallucination** | Confident, plausible, but false output. |
| **RAG** | Retrieve external context → inject into prompt. |
| **Agent** | Model that plans + uses tools in a loop. |
| **LoRA / PEFT** | Cheap finetuning via small trainable adapters. |
| **AI as a judge** | Using an LLM to grade LLM outputs (biased but scalable). |
| **TTFT / TPOT** | Latency metrics: first-token / per-token time. |
| **Guardrails** | Input/output filters for safety & defense. |

---

## My takeaways

_<!-- The 3–5 things that actually changed how you think. This is the part future-you will reread. -->_
- 
- 
- 

## How I'll apply this

_<!-- Concrete: a project, a decision, a habit. e.g. "Build an eval set before touching prompts on X" -->_
- 
- 

## Open questions / things to dig into

_<!-- What the book left you curious or unconvinced about -->_
- 
- 

## Quotes worth keeping

_<!-- Page + line if you want to find them again -->_
> 
