"""LLM interpreter for any OpenAI-compatible chat API (Groq by default; also OpenRouter, Cerebras,
GitHub Models, etc. by changing LLM_BASE_URL).

Contract: the model sees the utterance plus "is a patient open
(and their sex)", picks exactly one tool, and everything it returns is validated downstream.
"""
from __future__ import annotations

import json
from typing import Any

from .llm import SYSTEM_PROMPT, LLMInterpreterError
from .schema import LLM_TOOLS, InterpretContext, ToolCall, to_internal_call

# Convert the shared tool definitions to the OpenAI "function" format once.
OPENAI_TOOLS: list[dict[str, Any]] = [
    {"type": "function",
     "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in LLM_TOOLS
]


def reasoning_params(model: str, effort: str | None) -> dict[str, Any]:
    """Keep reasoning models fast. Groq: gpt-oss supports low/medium/high; Qwen supports none."""
    m = model.lower()
    if "gpt-oss" in m:
        return {"reasoning_effort": effort or "low", "include_reasoning": False}
    if m.startswith("qwen/"):
        return {"reasoning_effort": effort or "none", "reasoning_format": "hidden"}
    return {"reasoning_effort": effort} if effort else {}


class OpenAICompatInterpreter:
    def __init__(self, api_key: str | None, model: str, base_url: str | None, timeout_s: float = 8.0,
                 client: Any = None, provider: str = "groq", reasoning_effort: str | None = None,
                 price_in_per_mtok: float = 0.0, price_out_per_mtok: float = 0.0):
        if not model:
            raise LLMInterpreterError("LLM_MODEL is not set")
        if client is None:
            if not api_key:
                raise LLMInterpreterError(f"API key for '{provider}' is not set")
            try:
                from openai import OpenAI  # lazy: optional dependency
            except ImportError as e:
                raise LLMInterpreterError("The 'openai' package is not installed. "
                                          "Run: pip install -r requirements-providers.txt") from e

            # No SDK retries: on a rate limit or error we fall back to the rule-based interpreter
            # immediately instead of making a clinician wait for a retry.
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self.client = client
        self.model = model
        self.name = provider
        self.extra = reasoning_params(model, reasoning_effort)
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok

    def interpret(self, text: str, ctx: InterpretContext, mode: str = "auto") -> ToolCall:
        context_line = (f"Context: a patient is open in this conversation (sex: {ctx.active_patient_sex})."
                        if ctx.has_active_patient else "Context: no patient is open yet.")
        tools = OPENAI_TOOLS
        tool_choice: Any = "required"
        if mode == "dictation":
            tool_choice = {"type": "function", "function": {"name": "draft_findings"}}
        elif mode == "command":
            tools = [t for t in OPENAI_TOOLS if t["function"]["name"] != "draft_findings"]

        resp = self.client.chat.completions.create(
            model=self.model,
            # The system prompt and tools never change, so Groq's automatic prompt caching applies;
            # cached tokens don't count toward the free-tier tokens-per-minute limit. Per-turn context
            # therefore goes in the user message, not the system prompt.
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"{context_line}\nUtterance: {text}"}],
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.1,
            max_tokens=1024,
            extra_body=self.extra or None,
        )
        usage = getattr(resp, "usage", None)
        usage_dict = None
        if usage is not None:
            tin = getattr(usage, "prompt_tokens", 0) or 0
            tout = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details is not None else 0
            usage_dict = {"model": self.model, "input_tokens": tin, "cached_tokens": cached or 0, "output_tokens": tout,
                          "est_cost_usd": round((tin * self.price_in + tout * self.price_out) / 1_000_000, 6)}
        choices = getattr(resp, "choices", None) or []
        calls = getattr(choices[0].message, "tool_calls", None) if choices else None
        if not calls:
            raise LLMInterpreterError("Model returned no tool call")
        fn = calls[0].function
        try:
            args = json.loads(fn.arguments or "{}")
        except json.JSONDecodeError as e:
            raise LLMInterpreterError("Tool arguments were not valid JSON") from e
        if not isinstance(args, dict):
            raise LLMInterpreterError("Tool arguments were not a JSON object")
        try:
            name, args = to_internal_call(fn.name, args, text)
        except ValueError as e:
            raise LLMInterpreterError(str(e)) from e
        return ToolCall(name=name, args=args, source="llm", usage=usage_dict)
