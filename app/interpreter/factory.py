"""Builds the configured LLM interpreter (or None for the rule-based default)."""
from __future__ import annotations

from typing import Any

from ..config import Settings
from .llm import LLMInterpreterError
from .openai_compat import OpenAICompatInterpreter


def build_interpreter(settings: Settings) -> Any:
    kind = settings.interpreter
    if kind in ("rule_based", "", "rules"):
        return None
    if kind in ("groq", "openai_compat"):
        if kind == "openai_compat" and not settings.llm_base_url:
            raise LLMInterpreterError("LLM_BASE_URL is required for INTERPRETER=openai_compat")
        return OpenAICompatInterpreter(settings.llm_api_key, settings.llm_model, settings.llm_base_url,
                                       settings.llm_timeout_s, provider=kind,
                                       reasoning_effort=settings.llm_reasoning_effort,
                                       price_in_per_mtok=settings.llm_price_in_per_mtok,
                                       price_out_per_mtok=settings.llm_price_out_per_mtok)
    raise LLMInterpreterError(f"Unknown INTERPRETER '{kind}' (use rule_based, groq or openai_compat)")
