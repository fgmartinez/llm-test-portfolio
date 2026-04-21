"""
LangChain tools exposed to the clinic agent.

Only three tools are defined — keeping the tool surface small is deliberate:
DeepEval's ToolCorrectnessMetric needs to be able to judge whether the agent
picked the RIGHT tool, and a tight tool set makes tool-selection errors easy to
spot in the test output.

Persistence is intentionally skipped: booking/cancellation return confirmation
strings. The portfolio is about TESTING, not about a real booking backend.
"""

from __future__ import annotations

import json
import random
import re
import string
from typing import Any

from langchain_core.tools import tool

from src.rag.pipeline import RAGPipeline

# Singleton pipeline reused across tool calls. Building it once avoids re-opening
# Chroma and re-loading the LLM every time the agent asks a clinic info question.
_rag_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


# Hardcoded mock schedule — enough variety for the agent tests to exercise
# different date/doctor combinations without needing a real database.
_MOCK_SLOTS: dict[str, list[str]] = {
    "dr_smith":   ["09:00", "10:30", "14:00"],
    "dr_patel":   ["11:00", "13:00", "15:30"],
    "dr_johnson": ["08:30", "12:00", "16:00"],
    "dr_garcia":  ["09:30", "11:30", "14:30"],
}


def _parse_tool_payload(payload: Any, required_fields: tuple[str, ...]) -> dict[str, str]:
    """
    Parse ReAct tool input into named arguments.

    LangChain's ReAct agent passes Action Input as a string, even when the
    model writes JSON. The action tools accept one payload and parse it here.
    """
    if isinstance(payload, dict):
        data = payload
    else:
        raw = str(payload).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {key: _extract_key_value(raw, key) for key in required_fields}

    missing = [field for field in required_fields if not str(data.get(field, "")).strip()]
    if missing:
        return {
            "error": (
                "Missing required field(s): "
                + ", ".join(missing)
                + ". Provide a JSON object with: "
                + ", ".join(required_fields)
            )
        }

    return {field: str(data[field]).strip() for field in required_fields}


def _extract_key_value(raw: str, key: str) -> str:
    match = re.search(rf'{key}\s*=\s*"?([^",}}]+)"?', raw, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


@tool
def get_clinic_info(topic: str) -> str:
    """Look up factual clinic information (hours, insurance, doctors, policies).

    Use for any general question about the clinic. Do NOT use for booking,
    cancelling, or checking slot availability.
    """
    result = _get_pipeline().query(topic)
    return result.answer


@tool
def check_appointment_slots(tool_input: str) -> str:
    """Return available appointment time slots for the given doctor and date.

    Args:
        tool_input: JSON object with doctor and date, e.g.
            {"doctor": "Dr. Smith", "date": "2026-05-02"}.
    """
    args = _parse_tool_payload(tool_input, ("doctor", "date"))
    if "error" in args:
        return args["error"]

    doctor = args["doctor"]
    date = args["date"]
    key = doctor.lower().replace(" ", "_").replace("dr._", "dr_")
    slots = _MOCK_SLOTS.get(key)
    if not slots:
        return f"No schedule found for {doctor}."
    return f"Available slots for {doctor} on {date}: {', '.join(slots)}"


@tool
def book_appointment(tool_input: str) -> str:
    """Book an appointment and return a confirmation string.

    Args:
        tool_input: JSON object with doctor, date, time, and patient, e.g.
            {"doctor": "Dr. Smith", "date": "2026-05-04", "time": "10:30", "patient": "Jane Doe"}.
    """
    # Deterministic-ish confirmation id — enough for format tests, but unique
    # enough to look like a real booking reference.
    args = _parse_tool_payload(tool_input, ("doctor", "date", "time", "patient"))
    if "error" in args:
        return args["error"]

    doctor = args["doctor"]
    date = args["date"]
    time = args["time"]
    patient = args["patient"]
    confirmation = "CONF-" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )
    return (
        f"Appointment confirmed for {patient} with {doctor} on {date} "
        f"at {time}. Confirmation number: {confirmation}"
    )


ALL_TOOLS = [get_clinic_info, check_appointment_slots, book_appointment]
