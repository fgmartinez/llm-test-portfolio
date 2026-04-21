# LLM Test Portfolio

Portfolio repository for quality engineering work on LLM, RAG, and agentic AI
systems.

## Current project

### Clinic AI Testing

The main project lives in [`clinic-ai-testing/`](clinic-ai-testing/).

It demonstrates how to test a medical-clinic assistant built with:

- LangChain
- RAG over a small clinic knowledge base
- A ReAct-style scheduling agent
- DeepEval metrics
- RAGAS retrieval metrics
- Pytest goldens and HTML/JUnit reporting

Start here:

- [Clinic AI README](clinic-ai-testing/README.md)
- [Getting Started](clinic-ai-testing/GETTING_STARTED.md)
- [Testing AI Systems Notes](clinic-ai-testing/TESTING_AI_SYSTEMS.md)

## Quick Start

```bash
cd clinic-ai-testing
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

If you use the default Ollama provider:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
python -m src.rag.ingest
pytest tests/agent/test_tool_calls.py -v
```

## Reports

Pytest is configured to generate local reports:

```bash
pytest -v
```

Generated artifacts:

- `reports/pytest_report.html`
- `reports/junit.xml`

These files are ignored by Git.

## Repository Layout

```text
.
|-- README.md
|-- .gitignore
`-- clinic-ai-testing/
    |-- README.md
    |-- GETTING_STARTED.md
    |-- TESTING_AI_SYSTEMS.md
    |-- data/
    |-- eval/goldens/
    |-- src/
    `-- tests/
```
