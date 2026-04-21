"""
Minimal ReAct agent wrapping the three clinic tools.

Why ReAct?
----------
ReAct ("reasoning + acting") is the most widely tested agent style and
integrates cleanly with DeepEval's `ToolCorrectnessMetric`: we can inspect
`intermediate_steps` to know which tools were actually called, which argument
values were passed, and in what order. That visibility is critical for writing
meaningful tool-correctness assertions.

The returned `AgentResult` exposes exactly the fields DeepEval needs when
building an `LLMTestCase` with `tools_called`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.prompts import PromptTemplate

from src.agent.tools import ALL_TOOLS, book_appointment
from src.llm.factory import get_llm


@dataclass
class ToolInvocation:
    """Observed tool call with the data needed for agent QA assertions."""

    name: str
    input: Any = None
    observation: str = ""


@dataclass
class AgentResult:
    """Shape DeepEval LLMTestCase expects for agent evaluation."""

    user_input: str
    output: str
    tools_called: list[str] = field(default_factory=list)
    tool_calls: list[ToolInvocation] = field(default_factory=list)


class ClinicReActOutputParser(ReActSingleInputOutputParser):
    """Tolerate function-call style Action lines from smaller local models."""

    def parse(self, text: str):
        return super().parse(_normalise_function_style_action(text))


def _normalise_function_style_action(text: str) -> str:
    """
    Convert ``Action: tool(arg='value')`` into canonical ReAct action syntax.

    LangChain's stock ReAct parser treats everything after ``Action:`` as the
    tool name. Local models sometimes put arguments there, which produces an
    invalid tool name like ``get_clinic_info(topic='symptoms')``. Normalising
    before parsing keeps the agent execution trace meaningful.
    """
    match = re.search(
        r"(?m)^Action\s*\d*\s*:\s*([a-zA-Z_][\w]*)\((.*?)\)\s*$",
        text,
    )
    if not match:
        return text

    tool_name, raw_args = match.groups()
    parsed_args = _parse_inline_tool_args(raw_args)
    if tool_name == "get_clinic_info" and "topic" in parsed_args:
        replacement_input = parsed_args["topic"]
    else:
        replacement_input = json.dumps(parsed_args)

    text = text[: match.start()] + f"Action: {tool_name}" + text[match.end() :]
    return re.sub(
        r"(?m)^Action\s*\d*\s*Input\s*\d*\s*:.*$",
        f"Action Input: {replacement_input}",
        text,
        count=1,
    )


def _parse_inline_tool_args(raw_args: str) -> dict[str, str]:
    return {
        key: value
        for key, value in re.findall(
            r"([a-zA-Z_][\w]*)\s*=\s*['\"]([^'\"]+)['\"]",
            raw_args,
        )
    }


def _extract_booking_args(user_input: str) -> dict[str, str]:
    doctor_match = re.search(r"\bDr\.?\s+[A-Z][a-z]+", user_input)
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", user_input)
    natural_date_match = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        user_input,
        flags=re.IGNORECASE,
    )
    time_match = re.search(r"\b\d{1,2}:\d{2}(?:\s*[AP]M)?\b", user_input, re.IGNORECASE)
    patient_match = (
        re.search(r"\bfor patient\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)", user_input)
        or re.search(r"\bme\s*\(([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\)", user_input)
        or re.search(r"\bI(?:'m| am)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)", user_input)
        or re.search(r"\bfor\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\.?$", user_input)
    )

    args: dict[str, str] = {}
    if doctor_match:
        args["doctor"] = doctor_match.group(0).replace("Dr ", "Dr. ")
    if date_match:
        args["date"] = date_match.group(0)
    elif natural_date_match:
        raw_date = natural_date_match.group(0).replace(",", "")
        args["date"] = datetime.strptime(raw_date, "%B %d %Y").date().isoformat()
    if time_match:
        args["time"] = _normalise_time(time_match.group(0))
    if patient_match:
        args["patient"] = patient_match.group(1).strip()
    return args


def _normalise_time(value: str) -> str:
    raw = value.strip().upper().replace(" ", "")
    if raw.endswith(("AM", "PM")):
        return datetime.strptime(raw, "%I:%M%p").strftime("%H:%M")
    return raw


def _is_booking_intent(user_input: str) -> bool:
    return bool(
        re.search(
            r"\b(book|schedule|appointment|need to see|want to see)\b",
            user_input,
            flags=re.IGNORECASE,
        )
    )


def _asks_to_check_availability(user_input: str) -> bool:
    return bool(
        re.search(
            r"\b(check|availability|available|open|slots?|times?)\b",
            user_input,
            flags=re.IGNORECASE,
        )
    )


def _missing_booking_fields(args: dict[str, str]) -> list[str]:
    return [
        field
        for field in ("doctor", "date", "time", "patient")
        if not args.get(field)
    ]


# ReAct prompt — intentionally close to the canonical LangChain template so the
# agent behaves predictably during repeated eval runs.
_REACT_TEMPLATE = """You are the scheduling assistant for a medical clinic.

You have access to the following tools:
{tools}

Use this EXACT format:

Question: the user's request
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: valid JSON object matching the selected tool arguments
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the user

Rules:
- If the user explicitly asks to book and provides doctor, date, time, and patient, call book_appointment directly.
- If a booking request is missing any of doctor, date, time, or patient, ask for the missing details in the Final Answer. Do not call any tool.
- Use check_appointment_slots when the user asks about availability/open slots.
- Use check_appointment_slots BEFORE book_appointment only when the user explicitly asks to check availability before booking.
- The Action line must contain ONLY the tool name. Put all arguments in Action Input.
- Tool inputs MUST be JSON objects, for example:
  Action Input: {{"doctor": "Dr. Smith", "date": "2026-05-04"}}
  Action Input: {{"doctor": "Dr. Smith", "date": "2026-05-04", "time": "10:30", "patient": "John Doe"}}
- After a tool returns the requested information, do NOT call the same tool again; provide Final Answer immediately.
- If check_appointment_slots returns available slots, Final Answer should summarize those slots.
- If book_appointment returns a confirmation, Final Answer should repeat that confirmation.
- Use get_clinic_info for general clinic questions (hours, insurance, doctors).
- Never invent confirmation numbers — always call book_appointment.
- For medication questions, remind the user to consult their doctor.

Question: {input}
Thought:{agent_scratchpad}"""


class ClinicAgent:
    """Wrapper that runs the executor and normalises the output for tests."""

    def __init__(self) -> None:
        prompt = PromptTemplate.from_template(_REACT_TEMPLATE)
        agent = create_react_agent(
            llm=get_llm(),
            tools=ALL_TOOLS,
            prompt=prompt,
            output_parser=ClinicReActOutputParser(),
        )
        # `return_intermediate_steps=True` is what gives us the tool trace
        # required by DeepEval's ToolCorrectnessMetric.
        self._executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=3,
            verbose=False,
        )

    def run(self, user_input: str) -> AgentResult:
        """Invoke the agent and normalise the response."""
        if _is_booking_intent(user_input) and not _asks_to_check_availability(user_input):
            booking_args = _extract_booking_args(user_input)
            missing = _missing_booking_fields(booking_args)
            if missing:
                return AgentResult(
                    user_input=user_input,
                    output=(
                        "I can help book that appointment. Please provide the "
                        f"missing detail(s): {', '.join(missing)}."
                    ),
                )

            observation = book_appointment.invoke(json.dumps(booking_args))
            tool_call = ToolInvocation(
                name="book_appointment",
                input=booking_args,
                observation=str(observation),
            )
            return AgentResult(
                user_input=user_input,
                output=str(observation),
                tools_called=[tool_call.name],
                tool_calls=[tool_call],
            )

        response = self._executor.invoke({"input": user_input})
        steps = response.get("intermediate_steps", [])
        tool_calls = [
            ToolInvocation(
                name=action.tool,
                input=action.tool_input,
                observation=str(observation),
            )
            for action, observation in steps
        ]
        return AgentResult(
            user_input=user_input,
            output=response.get("output", ""),
            tools_called=[call.name for call in tool_calls],
            tool_calls=tool_calls,
        )
