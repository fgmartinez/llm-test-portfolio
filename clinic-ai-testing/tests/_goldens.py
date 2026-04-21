"""
Single source of truth for golden datasets used by the test suite.

Why a module (not just a fixture)?
----------------------------------
`@pytest.mark.parametrize` runs at collection time, before any fixture can
execute. Previous versions of the suite worked around this by re-reading the
JSON file at module level in `test_tool_calls.py` while `conftest.py` also
exposed a `*_goldens` fixture. Two loaders, one dataset — easy to drift.

This module loads each JSON exactly once at import time and exposes both:
  * module-level lists (``AGENT_GOLDENS`` etc.) for ``parametrize`` decorators
  * a ``load_goldens`` helper that the conftest fixtures re-use
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import settings


def load_goldens(name: str) -> list[dict]:
    """Load ``eval/goldens/<name>_goldens.json`` as a list of dicts."""
    path: Path = settings.goldens_dir / f"{name}_goldens.json"
    return json.loads(path.read_text(encoding="utf-8"))


def input_id(golden: dict, limit: int = 60) -> str:
    """Readable pytest ID derived from ``user_input``. Enables ``pytest -k``."""
    slug = re.sub(r"[^a-z0-9]+", "-", golden["user_input"].lower()).strip("-")
    return slug[:limit] or "golden"


AGENT_GOLDENS: list[dict] = load_goldens("agent")
RAG_GOLDENS: list[dict] = load_goldens("rag")
SAFETY_GOLDENS: list[dict] = load_goldens("safety")
