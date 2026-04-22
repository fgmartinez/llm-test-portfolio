# LLM Test Portfolio

Portfolio repository demonstrating **quality engineering for LLM, RAG, and agentic AI systems**.

Built to show that testing AI is fundamentally different from testing conventional software — and requires a disciplined, layer-by-layer approach with the right metrics at each level.

---

## What this portfolio demonstrates

| Skill | Evidence |
|---|---|
| RAG pipeline evaluation | RAGAS metrics: context recall, precision, relevance |
| LLM-judged generation quality | DeepEval: faithfulness, answer relevancy |
| Agent tool-call correctness | `ToolCorrectnessMetric` + argument-level validation |
| Multi-step agent reasoning | Task completion, hallucination detection |
| Safety & policy enforcement | Bias probes, medical disclaimer GEval, confirmation format GEval |
| Multi-provider test harness | Ollama (local) · Anthropic Claude · OpenAI GPT — single env var switch |
| Parametrized golden datasets | 28+ test cases across RAG, agent, and safety layers |
| Professional test infrastructure | Session-scoped fixtures, HTML + JUnit reports, pytest markers |

---

## Current project

### Clinic AI Testing

The main project lives in [`clinic-ai-testing/`](clinic-ai-testing/).

A medical-clinic scheduling assistant built with LangChain (RAG + ReAct agent), evaluated across six test layers using DeepEval and RAGAS.

**Stack:** Python · LangChain · ChromaDB · DeepEval · RAGAS · Pydantic · Pytest

**Tested layers:**

| Layer | Metrics |
|---|---|
| Retrieval | `context_recall`, `context_precision`, `context_relevance` |
| Generation | `FaithfulnessMetric`, `AnswerRelevancyMetric` |
| Tool calls | `ToolCorrectnessMetric`, argument validation |
| Agent reasoning | `TaskCompletionMetric`, `HallucinationMetric`, `AnswerRelevancyMetric` |
| Safety | `BiasMetric` (gender, age, ethnicity probes) |
| Policy | `GEval` — medical disclaimer + booking confirmation format |

Start here:

- [Clinic AI README](clinic-ai-testing/README.md)
- [Getting Started](clinic-ai-testing/GETTING_STARTED.md)
- [Testing AI Systems — Theory & Practice](clinic-ai-testing/TESTING_AI_SYSTEMS.md)

---

## Quick Start

```bash
cd clinic-ai-testing
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Default provider is Ollama (fully local, no API key required):

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
python -m src.rag.ingest
pytest tests/agent/test_tool_calls.py -v
```

To run against Claude or GPT, set `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=openai` in `.env` with the corresponding API key. No code changes needed.

---

## Reports

```bash
pytest -v
```

Generates:

- `reports/pytest_report.html` — full HTML report with per-test traces
- `reports/junit.xml` — CI-compatible JUnit XML

These files are excluded from Git.

## Test Report

<img width="1906" height="714" alt="Pytest HTML report showing all test layers passing" src="https://github.com/user-attachments/assets/dd2fb16d-5ddf-4693-9634-49c5505877cf" />

---

## Repository Layout

```text
.
├── README.md
├── .gitignore
└── clinic-ai-testing/
    ├── README.md               # Architecture, setup, test matrix
    ├── GETTING_STARTED.md      # Step-by-step walkthrough
    ├── TESTING_AI_SYSTEMS.md   # Theory: RAG, agents, metrics, diagnostics
    ├── src/                    # Application code (RAG + agent)
    ├── tests/                  # Test suite (6 test files, 28+ parametrized cases)
    ├── eval/goldens/           # Golden datasets (JSON)
    ├── data/                   # Clinic knowledge base (Markdown)
    └── requirements.txt
```
