"""Live tests against the configured LLM provider (Groq or any OpenAI-compatible API). Skipped unless RUN_LIVE=1
and a key is configured.

    RUN_LIVE=1 pytest -m live          (PowerShell: $env:RUN_LIVE=1; pytest -m live)

They check behaviour, not exact wording: the right tool, no fallback, and that every draft
finding that survives is grounded in the transcript.
"""
import os
import time

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app

settings = load_settings()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE") != "1" or settings.interpreter == "rule_based" or not settings.llm_api_key,
                       reason="set RUN_LIVE=1, INTERPRETER and its API key to run live API tests"),
]


@pytest.fixture(scope="module")
def live():
    from app.interpreter.factory import build_interpreter

    client = TestClient(create_app(settings, interpreter=build_interpreter(settings)))

    def say(text, sid=None, mode="auto"):
        time.sleep(float(os.getenv("LIVE_PACE_S", "8")))   # stay under free-tier tokens-per-minute limits
        r = client.post("/api/assistant/message", json={"text": text, "session_id": sid, "mode": mode})
        assert r.status_code == 200, r.text
        body = r.json()
        assert not any(s["step"] == "interpret_fallback" for s in body["trace"]), body["trace"]
        return body
    return say


def _tool(r):
    return next(s["detail"]["tool"] for s in r["trace"] if s["step"] == "interpret" and s["detail"]["engine"] != "rule_based")


def test_live_retrieval_and_followup(live):
    r1 = live("Open Priya Sharma's latest OPG report")
    assert _tool(r1) == "open_report"
    assert r1["data"]["report"]["report_id"] == "R-1003"
    r2 = live("Summarize her report", r1["session_id"])
    assert _tool(r2) == "summarize_report" and r2["data"]["report_id"] == "R-1003"


def test_live_casual_phrasing(live):
    r = live("Can you get me the panoramic x-ray we took for Rahul Menon?")
    assert r["type"] == "report" and r["data"]["patient"]["patient_id"] == "P003"


def test_live_ambiguous_name_still_asks(live):
    assert live("Open Priya's latest OPG report")["type"] == "clarification"


def test_live_dictation_is_grounded(live):
    text = "Lower right third molar is horizontally impacted. No periapical radiolucency in relation to 36."
    d = live(text)["data"]
    assert d["extraction_engine"].startswith("llm+grounding")
    got = {(f["category"], f["site"]["fdi"] if f["site"] else None, f["present"]) for f in d["findings"]}
    assert ("impaction", "48", True) in got
    assert ("periapical", "36", False) in got
    assert all(f["category"] in ("impaction", "periapical") for f in d["findings"])
    assert "caries" not in d["draft_text"].lower()
