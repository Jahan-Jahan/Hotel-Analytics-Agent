"""The agent loop: LLM -> tool calls -> tool results -> LLM, until a final answer."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI, OpenAIError

from hotel_agent.config import Settings
from hotel_agent.prompts import SYSTEM_PROMPT
from hotel_agent.tools import ToolBox

MAX_TOOL_CONTENT_CHARS = 12_000
MAX_HISTORY_TURNS = 6


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_s: float


@dataclass
class AgentResult:
    answer: str
    completed: bool
    steps: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None
    retry_nudges: int = 0           # times the loop re-prompted the model after a fixable tool error

    @property
    def charts(self) -> list[str]:
        return [c.result["path"] for c in self.tool_calls
                if c.name == "make_chart" and c.result.get("ok")]


class Agent:
    def __init__(self, client: Any, toolbox: ToolBox, model: str, max_steps: int = 8):
        self.client = client
        self.toolbox = toolbox
        self.model = model
        self.max_steps = max_steps
        self.turns: list[list[dict[str, Any]]] = []   # one list of messages per user turn

    def reset(self) -> None:
        self.turns.clear()

    def _messages(self) -> list[dict[str, Any]]:
        history = [m for turn in self.turns[-MAX_HISTORY_TURNS:] for m in turn]
        return [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    def ask(self, question: str) -> AgentResult:
        started = time.perf_counter()
        turn: list[dict[str, Any]] = [{"role": "user", "content": question}]
        self.turns.append(turn)
        result = AgentResult(answer="", completed=False)
        retry_hint: str | None = None   # set when the last tool calls all failed with a fixable error

        for step in range(1, self.max_steps + 1):
            result.steps = step
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=self._messages(), tools=self.toolbox.schemas,
                    tool_choice="auto", temperature=0,
                )
            except OpenAIError as exc:
                result.error = f"LLM request failed: {type(exc).__name__}: {str(exc)[:200]}"
                result.answer = "Sorry, the language model is unavailable right now."
                break

            usage = getattr(response, "usage", None)
            if usage is not None:
                result.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                result.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

            message = response.choices[0].message
            entry: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            turn.append(entry)

            if not message.tool_calls:
                if retry_hint and result.retry_nudges == 0:
                    # The model gave up right after a fixable argument error (e.g. a field it forgot).
                    # Re-prompt ONCE per turn: small models often stop instead of retrying.
                    result.retry_nudges += 1
                    turn.append({"role": "user", "content": (
                        f"Automatic check: your last tool call failed ({retry_hint}). Re-read my original "
                        f"message. If it already contains the missing values, call the tool again now with "
                        f"them; do not ask me to confirm. Ask me only for values I really did not give.")})
                    retry_hint = None
                    continue
                result.answer = (message.content or "").strip()
                result.completed = True
                break

            retry_hint = None
            failed_retryable: list[str] = []
            for tc in message.tool_calls:
                t0 = time.perf_counter()
                output = self.toolbox.call(tc.function.name, tc.function.arguments)
                try:
                    parsed_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    parsed_args = {"_raw": tc.function.arguments}
                result.tool_calls.append(ToolCallRecord(
                    tc.function.name, parsed_args, output, time.perf_counter() - t0))
                content = json.dumps(output, default=str)
                if len(content) > MAX_TOOL_CONTENT_CHARS:
                    content = content[:MAX_TOOL_CONTENT_CHARS] + '..."[truncated]"'
                turn.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                if not output.get("ok") and output.get("retryable"):
                    failed_retryable.append(str(output.get("error", ""))[:200])
            if failed_retryable and len(failed_retryable) == len(message.tool_calls):
                retry_hint = failed_retryable[0]
        else:
            result.answer = (f"I could not finish within {self.max_steps} steps. "
                             f"Try a more specific question.")
            result.error = "max_steps_reached"

        result.latency_s = time.perf_counter() - started
        return result


def build_agent(settings: Settings | None = None, toolbox: ToolBox | None = None) -> Agent:
    """Create an agent. Pass a shared `toolbox` to reuse DB connections across sessions (the UI does)."""
    settings = settings or Settings.from_env()
    settings.require_llm()
    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                    timeout=60, max_retries=2)
    return Agent(client, toolbox or ToolBox(settings), settings.llm_model, settings.max_steps)
