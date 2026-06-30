# AI Engineering by Chip Huyen 

| | |
|---|---|
| **Author** | Chip Huyen |
| **Published** | 2024 (O'Reilly) |
| **Status** | Reading |
| **Date read** | June 2026 (Better late than never:p) |
| **Rating** | _<!-- /5 -->_ |
| **Tags** | `llm` `foundation-models` `ml-systems` `rag` `evaluation` |

---

## Short summary

How to build *applications on top of* foundation models. Many fundamental concepts about AI and ML are concisely explained. The whole book is organized around adapting an existing model to our needs, or to put it blunty: how to build an AI wrapper saas app:D

---

## Chapter-by-chapter notes

I put Claude generated notes on chapters that I haven't read yet and will integrate my own notes and mark it "Done" once I have finished the reading myself:p

### 1. Introduction to Building AI Applications with Foundation Models (Done)
- **Evolution:** "ML → Language Model (LM) → LLM → Foundation Model"
- An **LM is one of many ML algorithms.** Others: object detection, topic modeling, recommender systems, weather forecasting, stock-price prediction, etc. The LM is special because it's trained with **self-supervision**, while the others use **supervision**.
- An **LM encodes statistical information** about one or more languages. Intuitively, this tells us *how likely a word is to appear in a given context.* For example, **Claude Shannon**: the GOAT, celebrated as the *"father of the digital age,"* who laid the theoretical foundations for modern computers, digital circuits, and information theory, used more sophisticated statistics to decipher enemies' messages during WWII. Many concepts from his paper *"Prediction and Entropy of Printed English"* are still used in LMs today.
- An LM can involve multiple languages. The basic unit of an LM is a **token** (a character or part of a word). On average, **100 tokens ≈ 75 words.** The set of all tokens is the **vocabulary.**
  - GPT-5.5 → ~200,000–256,000 tokens
  - Anthropic's Claude Opus 4.8 → ~200,000 tokens
  - GPT-4 and earlier → ~100,000 tokens
- **2 types of LM:**
  - ***Masked LM***: trained to predict missing tokens *anywhere* in a sequence, using context from before *and* after (like a fill-in-the-blank).
  - ***Autoregressive LM***: predicts the *next* token in a sequence. The model of choice for text generation.
- **Supervision:** label examples to show the model the desired behavior, then train on them. Once trained, the model can be applied to new data (e.g., a fraud-detection model). Its success in the 2010s kicked off the deep learning revolution. *Drawback:* data labeling is expensive, and not all labeling tasks are simple.
- **Self-supervision:** the model infers labels from the input data itself.
- An **LM scales up to become an LLM.** Size = number of **parameters.** A *parameter* is a variable within an ML model that is updated during training. In general (though not always), more parameters → more capacity to learn desired behaviors. E.g., GPT-2 had 1.5 billion parameters; GPT-5 is in the trillions now (*no source*). A larger model trained on a large dataset is trained more efficiently (same holds for small models).
- For a long time, **AI research was divided by data modality:** NLP dealt only with text, CV only with vision, audio-only models handled speech, etc. A model that works with more than one modality is a **multimodal model.** A *generative* multimodal model is an **LMM** (generating the next token from text *and* other modalities), hence the term **foundation models.** The shift is *task-specific → general-purpose.* We can also tweak a general-purpose model to maximize its performance on a specific task.
- **Three model adaptation techniques:**
  - **Prompt Engineering**: craft detailed instructions with examples of desirable outcomes.
  - **RAG (Retrieval-Augmented Generation)**: use a database to *supplement* the instructions.
  - **Finetuning**: further train the model on a dataset of high-quality outcomes.
- **3 types of competitive advantage (moats) in AI:** *technology, data, and distribution.*
  - With foundation models, the core tech will be similar for most.
  - Distribution likely belongs to big companies.
  - If a startup can get to market first and gather enough usage data to continually improve its product, **data becomes its moat.**
  - Successful companies whose original products *could* have been features of a larger product: Calendly, Mailchimp, Photoroom.
- **3 layers to any AI app stack:**
  1. **App development**: interfaces, providing the model with good prompts and necessary context, evaluation.
  2. **Model development**: tooling for developing models: frameworks for modeling, training, finetuning, inference optimization, and dataset engineering. *Most commonly associated with traditional ML engineering.* Modeling & training = coming up with a model architecture, training it, and finetuning it.
     - *Tools:* Google's TensorFlow, Hugging Face's Transformers, Meta's PyTorch.
     - *ML algorithm types:* clustering, logistic regression, decision trees, collaborative filtering.
     - *Neural network architectures:* feedforward, recurrent, convolutional, transformer.
     - *How models learn:* gradient descent, loss functions, regularization, etc.
  3. **Infra**: tooling for model serving, managing data and compute, and monitoring.
- **Model adaptation splits into 2 categories:** *prompt-based techniques* (prompt engineering) and *finetuning* (updating model weights). Finetuning is more complicated and needs more data, but can improve a model's quality, latency, and cost **SIGNIFICANTLY.** Adapting the model to a new task is usually the use case.
- **Training always changes model weights, but not all weight changes count as training.** E.g., *quantization* (reducing the precision of model weights) is not training. The different phases:
  1. **Pre-training**: train a model from scratch (weights randomly initialized). By far the most resource-intensive in compute, data, and time. *An art only a few practice.*
  2. **Finetuning**: continue training a previously trained model (weights inherited from the prior run). Requires fewer resources.
  3. **Post-training**: conceptually the same as finetuning. Convention: it's *post-training* when done by the model's own developers, *finetuning* when done by application developers.
- **Dataset engineering:** curating, generating, and annotating the data needed for training and adapting AI models. In AI engineering, data manipulation is more about *deduplication, tokenization, context retrieval, and quality control*, including removing sensitive information and toxic data.
- **Inference optimization:** making models faster and cheaper. A challenge with foundation models is that they're often *autoregressive*: if it takes 10 ms to generate one token, a 100-token output takes a full second, and longer outputs take even more. Since users are notoriously impatient (*rip attention span*), getting latency down to the ~100 ms expected of a typical internet app is a huge challenge. Inference optimization is an active subfield in both industry and academia.


### 2. Understanding Foundation Models
- **Training data** decides the ceiling, a model is only as good (and as biased / language-skewed) as its corpus.
- **Modeling:** transformer architecture, model size, and **scaling laws** (performance scales predictably with compute/data/params). Bigger isn't always the answer, data quality matters.
- **Post-training** is what makes a raw model usable:
  - *Supervised finetuning (SFT)* → teaches it to follow instructions.
  - *Preference finetuning (RLHF / DPO)* → aligns outputs with human preference.
- **Sampling** is the source of both magic and pain:
  - Temperature, top-k, top-p control randomness.
  - The model is **probabilistic**: same input can give different outputs. This *is* the creativity, and also the inconsistency.
  - **Hallucination**, two proposed causes: (1) the model can't separate given facts from its own generations; (2) mismatch between its internal knowledge and the training labels.
  - Structured outputs (JSON, etc.) constrain sampling to a schema.

### 3. Evaluation Methodology
- The book's thesis-within-a-thesis: **evaluation is the hardest, most under-invested part of AI engineering.** Open-ended outputs have no single right answer.
- Language-modeling metrics: **perplexity / cross-entropy**, useful signal, but don't measure task usefulness.
- Exact evaluation (functional correctness, similarity to reference) vs. **subjective evaluation**.
- **AI as a judge:** use a strong model to grade outputs, scalable but biased: *position bias*, *verbosity bias* (prefers longer answers), *self-bias* (prefers its own style).
- **Comparative evaluation:** rank models head-to-head (the basis of leaderboards like Chatbot Arena).

### 4. Evaluate AI Systems
- Evaluation criteria to define up front: **domain-specific capability, generation quality (factual consistency, safety), instruction-following, cost, and latency.**
- **Model selection:** build vs. buy; open-weight (self-host, control, privacy) vs. commercial API (easy, capable, but lock-in + data concerns). Decision hinges on your eval criteria, not hype.
- **Build an evaluation pipeline**: don't eyeball it. Define metrics, curate an eval set, automate, and keep it versioned.

### 5. Prompt Engineering
- The cheapest lever. Best practices: be explicit, give examples (few-shot), break tasks into steps (chain-of-thought), provide a persona/role, specify output format.
- Context construction matters as much as the instruction.
- **Defensive prompt engineering**: treat prompts as an attack surface:
  - *Prompt injection*, *jailbreaking*, *information / prompt extraction*.
  - Mitigations: input/output guardrails, system-prompt isolation, least-privilege tool access.

### 6. RAG and Agents
- **RAG (Retrieval-Augmented Generation):** retrieve relevant context, stuff it into the prompt. Solves the "model lacks information / context window too small" problem.
  - Retrieval algorithms: **term-based** (BM25/keyword, cheap, exact) vs. **embedding-based** (semantic, vector search). Often hybrid.
  - RAG quality is *retrieval* quality, garbage retrieved, garbage generated.
- **Agents:** a model that can **plan** and **use tools** in a loop to accomplish multi-step tasks.
  - Components: tools (read/write actions), planning, memory.
  - Failure modes compound, each step's error rate multiplies over a long trajectory. Agents are powerful but fragile; high-stakes actions need guardrails.
- **Memory** lets agents/chatbots carry context across steps and sessions.

### 7. Finetuning
- When prompting + RAG aren't enough, and only then (it's the most expensive, most operationally heavy rung).
- Use it to change **behavior/form/style**, not to inject facts (that's RAG's job).
- **Memory is the bottleneck**: full finetuning needs huge GPU memory.
- **PEFT (Parameter-Efficient FineTuning)**, esp. **LoRA**: freeze the base model, train small adapter matrices. Dramatically cheaper, composable.
- **Model merging**: combine multiple finetuned models.

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
- Architecture grows in layers, start simple, add only when needed: model API → **context construction** → **guardrails** (input/output) → **model router & gateway** → **caching** → agent patterns → **observability/monitoring**.
- **User feedback is your real-world eval set.** Design the product to capture it (explicit thumbs up/down *and* implicit signals like regenerate, copy, edit, abandon).
- Feedback has biases and limitations, design for them; don't treat thumbs-up as ground truth.

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
