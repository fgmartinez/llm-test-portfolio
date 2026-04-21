# Testing AI Agents and RAG Systems — Theory and Practice

## 1. Project objective

This project is a **QA laboratory for LLM-based systems**. The application itself is
intentionally simple: a scheduling assistant for a medical clinic. What matters is not
building the most sophisticated chatbot, but using a clear enough architecture to be
able to **measure, diagnose, and improve it**.

The core ideas are:

- show how a **RAG pipeline** works end-to-end;
- show how an **agent** uses tools to resolve tasks;
- separate the system into independently evaluable layers;
- demonstrate that in AI you cannot just ask "did it answer correctly?" — you need
  to measure multiple distinct quality dimensions.

This repository does not just answer questions and book appointments. It also teaches
**how to test an AI system professionally**.

---

## 2. Architecture overview

The project has two main levels:

1. **A RAG system** — retrieves information from the clinic knowledge base and
   generates an answer grounded in that context.

2. **A ReAct agent** — decides which tool to use based on the user's intent. One of
   those tools calls the RAG pipeline internally.

The relationship between them matters:

- The **RAG** resolves knowledge questions: hours, insurance, doctors, policies.
- The **agent** resolves tasks: check availability, book an appointment, answer
  general questions.
- The agent does not "know everything by itself": for informational queries it uses
  the `get_clinic_info` tool, which in turn invokes the RAG pipeline.

This layered structure is ideal for learning how to test AI:

1. evaluate **retrieval** first;
2. then **generation**;
3. then the agent's **tool selection**;
4. finally **safety and policy** compliance.

---

## 3. Project components

### 3.1 Central configuration

`src/config.py` centralises all system configuration:

- LLM provider (`ollama`, `anthropic`, `openai`);
- chat and embedding model names;
- RAG parameters: `chunk_size`, `chunk_overlap`, `retrieval_k`;
- file paths for the knowledge base, Chroma, and goldens;
- evaluation thresholds.

Centralising configuration avoids "magic numbers" scattered across the codebase
and makes it trivial to compare different models or settings.

### 3.2 LLM and embeddings factory

`src/llm/factory.py` encapsulates creation of:

- the **main system LLM**;
- the **embeddings** model;
- special **RAGAS wrappers**.

This layer separates "what I want to evaluate" from "which provider I am using."
The same test suite can run against different models without changing any
application code.

### 3.3 Knowledge base

The clinic's knowledge lives in `data/clinic_knowledge.md`. It contains structured
information about:

- hours (including Saturday urgent-only hours);
- accepted and rejected insurance plans;
- doctor specialties and availability;
- cancellation policy and fees;
- medication refill policy;
- emergency contacts.

This file is the **source of truth** for the RAG system. If information is not
present here, the system should refuse to invent it.

### 3.4 Ingestion

`src/rag/ingest.py` handles the offline phase of RAG:

1. load the markdown document;
2. split it into chunks with `MarkdownTextSplitter`;
3. embed each chunk;
4. persist the embeddings to Chroma.

This reflects a key RAG principle: **ingestion and inference are separate processes.**
A stable, repeatable index ensures metric values do not drift between test runs.

### 3.5 RAG pipeline

`src/rag/pipeline.py` implements the minimal 3-step pipeline:

1. **retrieve** — find the most relevant chunks;
2. **prompt assembly** — build context + question;
3. **generate** — the LLM answers using only the provided context.

The system prompt enforces important rules:

- answer only from the provided context;
- say you don't have the information if it is not in context;
- always advise the user to consult their doctor for medication questions.

This illustrates the difference between a raw LLM and a RAG system:

- a raw LLM answers from its trained parameters;
- a RAG answers from a combination of model + retrieved context.

### 3.6 Agent tools

`src/agent/tools.py` defines three tools:

- `get_clinic_info(topic)` — queries the RAG pipeline;
- `check_appointment_slots(doctor, date)` — consults a mock schedule;
- `book_appointment(doctor, date, time, patient)` — returns a mock confirmation.

The tool surface is intentionally small. That makes it easy to tell whether the
agent chose the right tool or made the wrong strategic decision.

### 3.7 ReAct agent

`src/agent/agent.py` implements a **ReAct**-style agent.

ReAct means:

- **reason** about what to do;
- **act** by calling a tool;
- observe the result;
- repeat until a final answer is ready.

The agent prompt defines rules like:

- use `get_clinic_info` for general questions;
- use `check_appointment_slots` before booking if availability is unknown;
- never invent confirmation numbers;
- remind users to consult their doctor for medication questions.

This turns the system into more than a chatbot: it becomes a **tool orchestrator**.

---

## 4. What is RAG and why does it matter?

**RAG** stands for *Retrieval-Augmented Generation*. It is an architecture where the
model answers not only from training memory, but from information dynamically
retrieved from an external source.

Its primary goals are:

- reduce hallucinations;
- use more controlled and verifiable information;
- update knowledge without retraining the model.

In this project, RAG answers questions like:

- "What are your Saturday hours?"
- "Do you accept Aetna?"
- "What is the cancellation policy?"

Without RAG, the model might produce a plausible but fabricated answer. With RAG,
the system has a concrete document base to ground its responses on.

---

## 5. RAG pipeline: step by step

### 5.1 Step 1 — chunking

The document is split into fragments called **chunks**.

Why not use the whole document:

- it would be less efficient;
- it would mix unrelated topics;
- it would degrade retrieval;
- it would add noise for the model.

Chunking directly affects retrieval quality:

- chunks too large: high coverage but lower precision;
- chunks too small: better local precision but risk of losing context;
- low overlap: may cut critical information;
- high overlap: improves continuity, but introduces redundancy.

### 5.2 Step 2 — embeddings

Each chunk is converted into a numeric vector representing its semantic meaning.

The user query is also embedded. Vectors are then compared to retrieve the closest
chunks — not by exact word match but by semantic similarity.

### 5.3 Step 3 — retrieval

The retriever selects the `k` most relevant chunks. `retrieval_k` in settings
controls how many fragments are fetched.

Design considerations:

- low `k`: less noise, but risk of missing necessary information;
- high `k`: higher recall, but more noise and lower precision.

### 5.4 Step 4 — prompt grounding

The retrieved chunks are inserted into the prompt as context. This is called
**grounding**: anchoring the answer to concrete evidence.

This step defines the boundary between:

- "answering with documentary backing";
- "answering from model intuition".

### 5.5 Step 5 — generation

The LLM produces the final answer.

Even when retrieval works correctly, generation can still fail:

- poor summarisation;
- answering a different question;
- inventing details not in context;
- omitting a required safety disclaimer.

This is why retrieval and generation must be evaluated separately.

---

## 6. What is an AI agent in this project?

An **agent** is a system that, faced with a task, can decide between different
actions rather than producing a single direct response.

In this project, the agent:

- interprets the user's request;
- decides if it is informational or action-oriented;
- selects a tool;
- can chain multiple tools;
- produces a final answer.

Examples:

- if the user asks about hours → `get_clinic_info`;
- if they ask about availability → `check_appointment_slots`;
- if they provide all booking details → `book_appointment`;
- if they ask to check availability and then book → two tools in sequence.

Testing agents introduces an extra challenge beyond plain RAG:

- it is not enough to measure whether the answer sounds good;
- you must measure whether the **resolution strategy** was correct.

---

## 7. Why AI QA is done layer by layer

One of the most common mistakes when starting to test LLMs is looking only at the
final answer. That is not sufficient.

If an answer is wrong, the problem could come from many layers:

- the retriever did not find the right evidence;
- the prompt did not guide the model well;
- the model misunderstood the question;
- the agent chose the wrong tool;
- the final format does not meet business policy;
- the answer is correct but unsafe.

That is why this project is organized into evaluation layers:

1. **Retrieval**
2. **Generation**
3. **Tool usage**
4. **Agent quality**
5. **Safety and policy**

This layering enables better diagnosis.

Example:

- if `Faithfulness` drops but retrieval is good, the problem is usually in the
  prompt or the generation step;
- if `Context Recall` drops, the problem is usually in embeddings, chunking, or `k`;
- if the final answer fails but RAG retrieval and generation are both good, the error
  is probably in the agent or tool selection.

---

## 8. The role of goldens

Goldens are manually curated reference datasets. This project has three:

- `eval/goldens/rag_goldens.json`
- `eval/goldens/agent_goldens.json`
- `eval/goldens/safety_goldens.json`

### 8.1 RAG goldens

Each entry includes:

- `user_input`
- `reference` (expected answer)
- `reference_contexts` (ideal retrieved snippets)

This enables comparison between what the system should have answered and what
context it should have retrieved.

### 8.2 Agent goldens

Each entry includes:

- `user_input`
- `expected_output`
- `expected_tools` (in order)
- `context`

This allows evaluation of not just the result, but the expected tool sequence.

### 8.3 Safety goldens

These include prompts covering:

- medical disclaimer requirements;
- demographic bias probes;
- booking confirmation format.

In theory, goldens are the AI-testing equivalent of traditional test cases, but
adapted for a probabilistic domain where there is often no single exact correct string.

---

## 9. Metrics and what each one measures

### 9.1 Retrieval metrics

#### Context Recall

> Did the retriever surface all the information needed to answer correctly?

High recall means no important evidence was left out.

Most impacted by: chunking, embeddings, `retrieval_k`, document quality.

If this drops: the answer may be incomplete even if the model is capable.

#### Context Precision

> Was everything retrieved genuinely relevant, or was there noise?

High precision means low distraction.

Most impacted by: embedding quality, chunk granularity, value of `k`, need for re-ranking.

If this drops: the model receives distracting context and may produce confused or
diluted answers.

#### Context Relevance

> Are the retrieved chunks semantically aligned with the query intent?

Similar to precision but focused on overall semantic alignment with the question.

### 9.2 Generation metrics

#### Faithfulness

> Is the answer supported by the retrieved context?

One of the most important metrics in RAG because it directly attacks hallucination.

If this drops: the model is adding claims not backed by the context. The answer may
sound convincing but be factually wrong.

#### Answer Relevancy

> Does the answer actually address what the user asked?

An answer can be faithful to context and still be unhelpful.

Classic example: the model copies the whole chunk, everything it says is in context,
but it does not directly answer the question.

### 9.3 Agent metrics

#### Tool Correctness

> Did the agent choose the right tools in the right order?

This metric evaluates operational strategy, not just the final text.

If this drops: the agent might produce a plausible-sounding answer via the wrong
path — hiding serious bugs, especially when real actions are involved.

#### Task Completion

> Did the agent actually complete the requested task?

Answering something relevant is not the same as completing an action.

Example: the user wants to book an appointment. The agent explains the booking
process. The answer may be relevant, but the task was not completed.

#### Hallucination

> Does the agent's output contradict the context or invent operational data?

In an agent, hallucination is critical: it can not only invent facts but also
fabricate supposedly completed actions.

Examples: inventing a confirmation number without calling `book_appointment`,
asserting unconfirmed availability, or altering doctor, date, or time relative
to the tool's actual return value.

### 9.4 Safety and policy metrics

#### BiasMetric

> Does the response contain biased language or validate an unjust premise?

In healthcare this is especially sensitive. The system should not reinforce
prejudice related to age, gender, ethnicity, or socioeconomic status.

Note: for this metric, **lower scores are better**.

#### GEval

GEval allows creating a custom rubric in natural language to evaluate requirements
not covered by any standard metric.

In this project, GEval covers:

- medication or treatment answers must include a "consult your doctor" disclaimer;
- booking confirmations must include doctor, date, time, and a `CONF-` number.

This is valuable because many real business rules are not about factual truth but
about **compliance criteria**.

---

## 10. Which metric impacts which part of the system

### 10.1 Knowledge base

Impacts: Context Recall, Faithfulness, Task Completion for informational queries.

If the knowledge base lacks information, the system should not invent it. If the
document is incomplete or ambiguous, the answers will be too.

### 10.2 Chunking

Impacts: Context Recall, Context Precision, Context Relevance, indirectly Faithfulness.

Chunking determines how knowledge is fragmented before indexing.

### 10.3 Embeddings

Impacts: Context Recall, Context Precision, Context Relevance.

Poor embeddings mean worse semantic neighbours and weaker retrieval.

### 10.4 Retrieval k

Impacts: Recall, Precision, Relevance, Faithfulness, Relevancy.

Higher `k` may improve coverage but degrades precision.

### 10.5 RAG prompt

Impacts: Faithfulness, Answer Relevancy, compliance with disclaimers.

This layer tells the model how to use the context.

### 10.6 Tool descriptions and agent prompt

Impacts: Tool Correctness, Task Completion, Hallucination.

If a tool is poorly described or the prompt does not clearly delimit its use,
the agent may call it in the wrong context.

### 10.7 Tool implementation

Impacts: Task Completion, Hallucination, GEval format checks.

Even if the agent reasons correctly, a tool that returns poor or inconsistent
output will degrade the final answer.

### 10.8 Goldens

Impacts: the quality of all evaluations.

A system may appear excellent with weak goldens or fail with poorly designed ones.
In AI testing, good evaluation depends heavily on good test case design.

---

## 11. What is concretely tested in this repository

### 11.1 Does the retriever find the right evidence?

Measured with RAGAS retrieval metrics over `rag_goldens.json`.

### 11.2 Does the model answer based on the context?

Measured with DeepEval using `FaithfulnessMetric` and `AnswerRelevancyMetric`.

### 11.3 Does the agent use the right tools?

Measured with agent goldens by comparing expected tools against actually called tools.

### 11.4 Does the agent complete tasks without hallucinating?

Evaluated with `TaskCompletionMetric` and `HallucinationMetric`.

### 11.5 Is the system safe and policy-compliant?

Tested with bias probes, medical disclaimer checks, and confirmation format validation.

---

## 12. Combining metrics intelligently

### 12.1 Faithfulness without recall is not enough

An answer may be fully faithful to the retrieved context, but if the retriever
fetched incomplete context, the answer will still be wrong.

- High `Faithfulness` does not guarantee overall correctness.
- `Context Recall` is also required.

### 12.2 High recall without precision is not enough either

Retrieving a lot of information may look like "everything is covered," but noise
can degrade generation.

- `Recall` without `Precision` can produce diffuse answers.

### 12.3 Tool correctness without task completion is not enough

The agent can call the right tool and still fail to complete the task.

- `ToolCorrectness` measures strategy.
- `TaskCompletion` measures operational outcome.

### 12.4 Safety is separate from utility

A response can be useful but unsafe.

Example: it answers a medication question, seems helpful, but omits "consult your
doctor." Safety should not be absorbed into general quality metrics.

### 12.5 Recommended layered view

A robust combination for professional AI QA work:

1. Retrieval: `Context Recall` + `Context Precision` + `Context Relevance`
2. Generation: `Faithfulness` + `Answer Relevancy`
3. Agentic behaviour: `Tool Correctness` + `Task Completion` + `Hallucination`
4. Safety/policy: `BiasMetric` + `GEval`

---

## 13. Diagnostic examples

### Case A — Incomplete answer, no fabrication

Pattern: low Recall, high Faithfulness.

Interpretation: the problem is in retrieval, not generation. The model is using
only what it was given; it simply was not given enough.

### Case B — Good context retrieved, but unsupported details in the answer

Pattern: high Recall, acceptable Precision, low Faithfulness.

Interpretation: the problem is in the prompt or the model's generation behaviour.

### Case C — Reasonable-sounding answer, but no booking made

Pattern: medium or low Tool Correctness, low Task Completion, acceptable Relevancy.

Interpretation: the agent understood the domain topic but did not execute the right
strategy.

### Case D — Correct answer, but policy non-compliant

Pattern: high Faithfulness, high Relevancy, low GEval.

Interpretation: the system knows how to answer, but does not meet domain-specific
requirements.

---

## 14. Why this project starts with RAG before the agent

The project guide recommends starting with RAG for solid theoretical reasons:

- RAG has fewer degrees of freedom;
- it is easier to isolate retrieval and generation;
- the agent depends partially on the RAG via `get_clinic_info`.

If you do not understand how the RAG fails first, it is much harder to interpret why
the agent fails later.

A sensible learning order:

1. understand the source document;
2. understand ingestion and chunking;
3. measure retrieval;
4. measure generation;
5. only then move to tools and the agent.

---

## 15. Pedagogical limitations and the value of a simple design

This repository intentionally simplifies many things:

- the schedule is mocked;
- bookings do not persist to a real database;
- there is no conversational memory;
- there is no query rewriting or complex re-ranking;
- there is no multi-agent orchestration.

Far from being a drawback, this simplicity has real didactic value:

- reduces conceptual noise;
- keeps every component visible and inspectable;
- makes it easier to attribute failures to a specific layer;
- turns the system into a clean, reproducible test bed.

This embodies an important idea in AI QA:

> To learn how to measure well, start with systems that are bounded and observable.

---

## 16. Suggested learning path

1. Read `data/clinic_knowledge.md` to understand the source of truth.
2. Read `src/rag/ingest.py` to understand how the vector index is built.
3. Read `src/rag/pipeline.py` to understand the retrieve → prompt → generate flow.
4. Review `eval/goldens/rag_goldens.json` to see how ground truth is defined.
5. Study the retrieval and generation metrics and their test implementations.
6. Move to `src/agent/tools.py` and `src/agent/agent.py`.
7. Review `agent_goldens.json` to understand how tool strategy is evaluated.
8. Finish with safety and GEval.

This path follows the system's increasing complexity.

---

## 17. Conclusion

This project demonstrates a fundamental idea in modern AI testing:

**An LLM-based system is not evaluated as a single black box, but as a set of layers
with different failure modes and different metrics.**

The most important takeaway from this repository is the mental framework:

- a **RAG** system needs retrieval and generation measured separately;
- an **agent** also needs tool behaviour, selection, and task completion measured;
- a real system additionally needs **safety, bias, and policy compliance** measured;
- **goldens** are the foundation that transforms probabilistic behaviour into
  systematic, repeatable evaluation.

---

## 18. Component-to-metric mapping

| System component         | What it does                                   | Most-impacted metrics                           |
|--------------------------|------------------------------------------------|-------------------------------------------------|
| Knowledge base           | Source of truth                                | Recall, Faithfulness, Relevancy                 |
| Chunking                 | Splits the document into retrievable units     | Recall, Precision, Relevance                    |
| Embeddings               | Represents chunks and queries semantically     | Recall, Precision, Relevance                    |
| Retriever                | Selects relevant context                       | Recall, Precision, Relevance, NoiseSensitivity  |
| RAG prompt               | Enforces grounding and response style          | Faithfulness, Relevancy, Safety                 |
| Generator LLM            | Produces the final answer                      | Faithfulness, Relevancy, Hallucination          |
| Tool descriptions        | Define correct tool usage                      | ToolCorrectness                                 |
| Agent prompt             | Orchestrates decision and tool sequence        | ToolCorrectness, TaskCompletion                 |
| Tool implementation      | Executes actions or queries                    | TaskCompletion, Hallucination, GEval            |
| Safety policies          | Prevent risky or biased responses              | BiasMetric, GEval                               |

---

## 19. Repository status

The RAG retrieval and generation test suites are fully implemented. The agent
tool-call and quality tests are also implemented. The `tests/safety/` files are
intentional scaffolds: their docstrings describe the evaluation strategy in detail,
making the repository valuable not only for running metrics but for learning how to
**design a testing strategy for AI systems**.
