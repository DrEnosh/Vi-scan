"""Runtime settings, read from environment variables (optionally via a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
STATIC_DIR = APP_DIR.parent / "static"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Per-provider defaults. Prices are USD per million tokens and only feed the per-call cost estimate.
LLM_DEFAULTS: dict[str, dict[str, object]] = {
    "groq": {"model": "openai/gpt-oss-120b", "base_url": GROQ_BASE_URL, "price_in": 0.0, "price_out": 0.0},
    "openai_compat": {"model": "", "base_url": None, "price_in": 0.0, "price_out": 0.0},
}
STT_DEFAULTS = {"groq": "whisper-large-v3-turbo", "openai": "whisper-1", "mock": ""}


@dataclass(frozen=True)
class Settings:
    interpreter: str = "rule_based"          # rule_based | groq | openai_compat
    llm_api_key: str | None = None           # key for the selected interpreter provider
    llm_model: str = ""
    llm_base_url: str | None = None
    llm_timeout_s: float = 8.0
    llm_reasoning_effort: str | None = None  # Groq reasoning models: none/low/medium/high
    llm_price_in_per_mtok: float = 0.0
    llm_price_out_per_mtok: float = 0.0
    stt_provider: str = "mock"               # mock | groq | openai
    stt_api_key: str | None = None
    stt_model: str = ""
    max_audio_mb: float = 10.0
    max_text_chars: int = 2000
    session_ttl_s: int = 1800
    default_clinician_id: str | None = "dr_demo"


def _secret(name: str) -> str | None:
    """Read a key from the environment, tolerating stray spaces or quotes from copy-paste."""
    v = (os.getenv(name) or "").strip().strip('"').strip("'").strip()
    return v or None


def _env(name: str) -> str | None:
    v = (os.getenv(name) or "").strip()
    return v or None


def load_settings() -> Settings:
    interp = (_env("INTERPRETER") or "rule_based").lower()
    d = LLM_DEFAULTS.get(interp, {})
    llm_key = {"groq": _secret("GROQ_API_KEY"), "openai_compat": _secret("LLM_API_KEY")}.get(interp)

    stt = (_env("STT_PROVIDER") or "mock").lower()
    stt_key = {"groq": _secret("GROQ_API_KEY"), "openai": _secret("OPENAI_API_KEY")}.get(stt)

    default_clinician = (os.getenv("DEFAULT_CLINICIAN_ID", "dr_demo") or "").strip() or None
    return Settings(
        interpreter=interp,
        llm_api_key=llm_key,
        llm_model=_env("LLM_MODEL") or str(d.get("model") or ""),
        llm_base_url=_env("LLM_BASE_URL") or (str(d["base_url"]) if d.get("base_url") else None),
        llm_timeout_s=float(_env("LLM_TIMEOUT_S") or 8),
        llm_reasoning_effort=_env("LLM_REASONING_EFFORT"),
        llm_price_in_per_mtok=float(_env("LLM_PRICE_IN_PER_MTOK") or d.get("price_in") or 0),
        llm_price_out_per_mtok=float(_env("LLM_PRICE_OUT_PER_MTOK") or d.get("price_out") or 0),
        stt_provider=stt,
        stt_api_key=stt_key,
        stt_model=_env("STT_MODEL") or STT_DEFAULTS.get(stt, ""),
        max_audio_mb=float(_env("MAX_AUDIO_MB") or 10),
        max_text_chars=int(_env("MAX_TEXT_CHARS") or 2000),
        session_ttl_s=int(_env("SESSION_TTL_S") or 1800),
        default_clinician_id=default_clinician,
    )
