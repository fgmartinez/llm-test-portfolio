# Clinic AI — QA Testing Portfolio

A portfolio project demonstrating **quality engineering for AI systems**.

The application is intentionally simple: a medical-clinic chatbot built with
a three-step RAG pipeline and a minimal ReAct agent. All the complexity lives
in the **testing framework** — RAG retrieval metrics, generation faithfulness,
tool-selection correctness, bias probes, and custom GEval criteria.

## Architecture

```
                ┌──────────────────────────┐
  User question │        ClinicAgent       │
  ───────────►  │   (LangChain ReAct, 3    │
                │   tools, 1 system msg)   │
                └──────────┬───────────────┘
                           │
         ┌─────────────────┼──────────────────────┐
         ▼                 ▼                      ▼
 get_clinic_info   check_appointment_slots   book_appointment
         │                 (mock)                (mock)
         ▼
 ┌──────────────────────┐
 │     RAGPipeline      │
 │  embed → retrieve →  │
 │      generate        │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │   Chroma persisted   │
 │   clinic_knowledge   │
 └──────────────────────┘
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Install Ollama (default provider)

1. Download and install Ollama: https://ollama.com/download
2. Pull the three local models used by this project:

```bash
ollama pull qwen2.5
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

3. Make sure the Ollama daemon is running (`ollama serve` or the desktop app).

### Switching providers

Edit `.env` and set `LLM_PROVIDER` to `anthropic` or `openai`, then provide
the corresponding API key. Embeddings fall back to Ollama for Anthropic; for
OpenAI, `OPENAI_EMBEDDING_MODEL` is used.

## Ingesting the knowledge base

The RAG pipeline reads from a persisted Chroma collection. Build it once:

```bash
python -m src.rag.ingest
```

This loads `data/clinic_knowledge.md`, splits it with `MarkdownTextSplitter`,
embeds every chunk, and persists them under `.chroma/`.

## Smoke test

```python
from src.rag.pipeline import RAGPipeline
from src.agent.agent import ClinicAgent

print(RAGPipeline().query("What are your Saturday hours?").answer)
print(ClinicAgent().run("Book Dr. Patel on 2026-05-06 at 11:00 for Jane Doe.").output)
```

## Running the test suite

```bash
# RAG retrieval quality (RAGAS — non-LLM, string-matching scorers)
pytest tests/rag/test_retrieval_ragas.py -v

# RAG generation quality (DeepEval — LLM-judged)
pytest tests/rag/test_generation_deepeval.py -v

# Agent tool-call correctness
pytest tests/agent/test_tool_calls.py -v

# Agent task completion & hallucination
pytest tests/agent/test_agent_quality.py -v

# Bias + custom GEval (medical disclaimer, confirmation format)
pytest tests/safety/ -v

# Full suite
pytest -v

# Show per-golden tool traces in stdout
pytest tests/agent/ -v -s
```

## Project layout

```
clinic-ai-testing/
├── src/
│   ├── config.py               # Pydantic settings (provider, thresholds, paths)
│   ├── llm/factory.py          # Multi-provider LLM + embeddings factory
│   ├── rag/ingest.py           # Chunk → embed → persist to Chroma
│   ├── rag/pipeline.py         # Minimal 3-step RAG chain
│   ├── agent/tools.py          # get_clinic_info / check_slots / book
│   └── agent/agent.py          # ReAct agent executor
├── tests/
│   ├── conftest.py             # Session-scoped fixtures
│   ├── _goldens.py             # Single source of truth for golden datasets
│   ├── rag/
│   │   ├── test_retrieval_ragas.py      # context_recall / precision / relevance
│   │   └── test_generation_deepeval.py  # faithfulness / answer_relevancy
│   ├── agent/
│   │   ├── _helpers.py             # Adapters, trace formatter, arg matcher
│   │   ├── test_tool_calls.py      # ToolCorrectnessMetric + arg validation
│   │   └── test_agent_quality.py   # TaskCompletion / AnswerRelevancy / Hallucination
│   └── safety/
│       ├── test_bias.py            # BiasMetric (scaffold — implement assertions)
│       └── test_custom_geval.py    # GEval: disclaimer + confirmation format (scaffold)
├── eval/goldens/
│   ├── rag_goldens.json        # 12 RAG Q&A reference pairs
│   ├── agent_goldens.json      # 8 agent tool-routing cases
│   └── safety_goldens.json     # 8 safety / bias / format cases
├── data/clinic_knowledge.md    # Source document for the RAG pipeline
├── GETTING_STARTED.md          # Step-by-step walkthrough of the framework
├── TESTING_AI_SYSTEMS.md       # Theory: RAG, agents, metrics, and diagnostics
├── .env.example
└── requirements.txt
```

## What is tested

| Layer      | File                                      | Metrics                                                      |
|------------|-------------------------------------------|--------------------------------------------------------------|
| Retrieval  | `tests/rag/test_retrieval_ragas.py`       | `context_recall`, `context_precision`, `context_relevance`   |
| Generation | `tests/rag/test_generation_deepeval.py`   | `FaithfulnessMetric`, `AnswerRelevancyMetric`                |
| Tools      | `tests/agent/test_tool_calls.py`          | `ToolCorrectnessMetric`, argument validation                 |
| Agent      | `tests/agent/test_agent_quality.py`       | `TaskCompletionMetric`, `AnswerRelevancyMetric`, `HallucinationMetric` |
| Safety     | `tests/safety/test_bias.py`               | `BiasMetric`                                                 |
| Policy     | `tests/safety/test_custom_geval.py`       | `GEval` (medical disclaimer, confirmation format)            |

The `tests/safety/` files are intentional scaffolds: their docstrings describe
the required assertions so readers can study the testing strategy before
implementing it.


## Test Report Screenshot
<img width="1906" height="714" alt="image" src="https://github.com/user-attachments/assets/dd2fb16d-5ddf-4693-9634-49c5505877cf" />
