"""Groq / OpenAI-compatible interpreter and STT, tested with scripted clients (no network)."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings, load_settings
from app.interpreter import InterpretContext, OpenAICompatInterpreter
from app.interpreter.factory import build_interpreter
from app.main import create_app
from app.stt import OpenAICompatSTT


class FakeChat:
    def __init__(self, name: str | None = None, args: Any = None, raw_args: str | None = None):
        self.calls: list[dict[str, Any]] = []
        self.name, self.args, self.raw_args = name, args, raw_args
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        tool_calls = None
        if self.name:
            arguments = self.raw_args if self.raw_args is not None else json.dumps(self.args)
            tool_calls = [SimpleNamespace(function=SimpleNamespace(name=self.name, arguments=arguments))]
        msg = SimpleNamespace(tool_calls=tool_calls, content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=900, completion_tokens=60))


def groq(fake: FakeChat, model: str = "openai/gpt-oss-120b") -> OpenAICompatInterpreter:
    return OpenAICompatInterpreter(None, model, "https://api.groq.com/openai/v1", client=fake, provider="groq")


def test_request_shape_and_tool_parsing():
    fake = FakeChat("report_request", {"action": "open", "patient_name": "Priya Sharma", "patient_reference": "named",
                                       "modality": "OPG", "which": "latest"})
    call = groq(fake).interpret("Open Priya Sharma's latest OPG report", InterpretContext(), "auto")
    assert call.name == "open_report" and call.args["patient_name"] == "Priya Sharma" and call.source == "llm"
    assert "action" not in call.args
    sent = fake.calls[0]
    assert sent["tool_choice"] == "required"
    assert {t["function"]["name"] for t in sent["tools"]} == {"report_request", "draft_findings", "unsupported_request"}
    assert sent["extra_body"] == {"reasoning_effort": "low", "include_reasoning": False}   # gpt-oss kept fast
    assert sent["messages"][0]["role"] == "system" and "Priya" not in sent["messages"][0]["content"]
    assert call.usage["input_tokens"] == 900 and call.usage["model"] == "openai/gpt-oss-120b"


def test_modes_force_or_exclude_dictation():
    fake = FakeChat("draft_findings", {"findings": []})
    groq(fake).interpret("x", InterpretContext(), "dictation")
    assert fake.calls[0]["tool_choice"] == {"type": "function", "function": {"name": "draft_findings"}}
    fake2 = FakeChat("open_report", {"patient_reference": "none", "which": "latest"})
    groq(fake2, "qwen/qwen3.8-27b").interpret("x", InterpretContext(), "command")
    assert "draft_findings" not in {t["function"]["name"] for t in fake2.calls[0]["tools"]}
    assert fake2.calls[0]["extra_body"]["reasoning_effort"] == "none"


def test_groq_end_to_end_through_app():
    fake = FakeChat("open_report", {"patient_name": "Rahul Menon", "patient_reference": "named",
                                    "modality": "OPG", "which": "latest"})
    client = TestClient(create_app(Settings(), interpreter=groq(fake)))
    text = "Can you get me the panoramic x-ray we took for Rahul Menon?"
    r = client.post("/api/assistant/message", json={"text": text}).json()
    assert r["type"] == "report" and r["data"]["report"]["report_id"] == "R-3001"
    step = next(s for s in r["trace"] if s["step"] == "interpret")
    assert step["detail"]["engine"] == "groq" and step["detail"]["usage"]["output_tokens"] == 60


def test_bad_model_output_falls_back_to_rules():
    for fake in (FakeChat(None), FakeChat("open_report", raw_args="{not json")):
        client = TestClient(create_app(Settings(), interpreter=groq(fake)))
        r = client.post("/api/assistant/message", json={"text": "Open Priya Sharma's latest OPG report"}).json()
        assert r["data"]["report"]["report_id"] == "R-1003"
        assert any(s["step"] == "interpret_fallback" for s in r["trace"])


def test_settings_for_groq(monkeypatch):
    monkeypatch.setenv("INTERPRETER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", '  "gsk_test_123"  ')
    monkeypatch.setenv("STT_PROVIDER", "groq")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("STT_MODEL", raising=False)
    s = load_settings()
    assert s.llm_api_key == "gsk_test_123" and s.stt_api_key == "gsk_test_123"   # one key, both uses
    assert s.llm_model == "openai/gpt-oss-120b" and s.llm_base_url == "https://api.groq.com/openai/v1"
    assert s.stt_model == "whisper-large-v3-turbo"
    made: dict[str, Any] = {}
    fake_sdk = SimpleNamespace(OpenAI=lambda **kw: made.update(kw) or SimpleNamespace())
    monkeypatch.setitem(sys.modules, "openai", fake_sdk)          # no network, no optional package needed
    interp = build_interpreter(s)
    assert interp.name == "groq" and interp.model == "openai/gpt-oss-120b"
    assert made["base_url"] == "https://api.groq.com/openai/v1" and made["max_retries"] == 0   # fail fast, fall back


def test_missing_provider_package_keeps_the_app_running(monkeypatch):
    """INTERPRETER=groq / STT_PROVIDER=groq without `pip install -r requirements-providers.txt`:
    the server still starts in offline mode and says why, instead of crashing."""
    monkeypatch.setitem(sys.modules, "openai", None)              # makes `import openai` fail
    for k, v in {"INTERPRETER": "groq", "STT_PROVIDER": "groq", "GROQ_API_KEY": "gsk_test_123"}.items():
        monkeypatch.setenv(k, v)
    client = TestClient(create_app(load_settings()))
    cfg = client.get("/api/config").json()
    assert cfg["interpreter"] == "rule_based" and "requirements-providers.txt" in cfg["llm_startup_error"]
    assert cfg["stt_provider"] == "mock" and "requirements-providers.txt" in cfg["stt_startup_error"]
    r = client.post("/api/assistant/message", json={"text": "Open Priya Sharma's latest OPG report"})
    assert r.status_code == 200 and r.json()["data"]["report"]["report_id"] == "R-1003"


def test_groq_stt_request():
    sent: dict[str, Any] = {}

    def create(**kw: Any) -> Any:
        sent.update(kw)
        return SimpleNamespace(text=" Lower right third molar is horizontally impacted. ")

    fake = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    stt = OpenAICompatSTT(None, "whisper-large-v3-turbo", name="groq", client=fake)
    client = TestClient(create_app(Settings(), stt=stt))
    r = client.post("/api/assistant/voice", files={"audio": ("mic.webm", b"opus-bytes", "audio/webm;codecs=opus")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] == "Lower right third molar is horizontally impacted."
    assert body["type"] == "draft_report" and body["trace"][0]["detail"]["provider"] == "groq"
    assert sent["model"] == "whisper-large-v3-turbo" and sent["language"] == "en" and "OPG" in sent["prompt"]
    assert sent["file"][2] == "audio/webm"


def test_null_arguments_are_accepted():
    """Groq validates generated tool calls against the schema; models often send null for optional args."""
    from app.interpreter.schema import LLM_TOOLS

    props = {t["name"]: t["input_schema"]["properties"] for t in LLM_TOOLS}
    assert "null" in props["report_request"]["patient_id"]["type"] and "null" in props["report_request"]["date"]["type"]
    assert None in props["report_request"]["modality"]["enum"]
    assert props["report_request"]["action"]["type"] == "string"          # required stays strict
    item = props["draft_findings"]["findings"]["items"]["properties"]
    assert "null" in item["tooth_fdi"]["type"] and item["evidence"]["type"] == "string"   # required stays strict

    fake = FakeChat("open_report", {"patient_name": "Priya Sharma", "patient_id": None, "patient_reference": "named",
                                    "modality": "OPG", "which": "latest", "date": None})
    client = TestClient(create_app(Settings(), interpreter=groq(fake)))
    r = client.post("/api/assistant/message", json={"text": "Open Priya Sharma's latest OPG report"}).json()
    assert r["data"]["report"]["report_id"] == "R-1003"
    assert not any(s["step"] == "interpret_fallback" for s in r["trace"])

    fake2 = FakeChat("draft_findings", {"findings": [
        {"category": "impaction", "qualifiers": None, "tooth_fdi": 48, "site_label": None, "present": True,
         "certainty": "stated", "evidence": "Lower right third molar is horizontally impacted"}]})
    client2 = TestClient(create_app(Settings(), interpreter=groq(fake2)))
    text = "Lower right third molar is horizontally impacted"
    d = client2.post("/api/assistant/message", json={"text": text}).json()["data"]
    assert [(f["category"], f["site"]["fdi"]) for f in d["findings"]] == [("impaction", "48")]
    assert d["rejected_findings"] == []


def test_system_prompt_is_static_for_caching():
    """Groq caches repeated prompt prefixes (cached tokens don't count toward the free-tier TPM limit)."""
    fake = FakeChat("summarize_report", {"patient_reference": "pronoun_female", "which": "latest"})
    interp = groq(fake)
    interp.interpret("Summarize her report", InterpretContext(), "auto")
    interp.interpret("Summarize her report", InterpretContext(has_active_patient=True, active_patient_sex="F"), "auto")
    first, second = fake.calls
    assert first["messages"][0] == second["messages"][0]                  # identical, cacheable prefix
    assert first["tools"] == second["tools"]
    assert "patient is open" in second["messages"][1]["content"]           # context travels with the utterance
    assert second["messages"][1]["content"].endswith("Utterance: Summarize her report")


def test_dictation_merges_findings_the_llm_missed():
    text = "Lower right third molar is horizontally impacted with roots close to the mandibular canal."
    fake = FakeChat("draft_findings", {"findings": [
        {"category": "impaction", "qualifiers": ["horizontal"], "tooth_fdi": "48", "present": True,
         "certainty": "stated", "evidence": "Lower right third molar is horizontally impacted"}]})
    client = TestClient(create_app(Settings(), interpreter=groq(fake)))
    r = client.post("/api/assistant/message", json={"text": text}).json()
    d = r["data"]
    assert d["extraction_engine"] == "llm+grounding+rules"
    assert [(f["category"], f["site"]["fdi"], f["source"]) for f in d["findings"]] == [
        ("impaction", "48", "llm"), ("canal_proximity", "48", "rule_parser")]
    step = next(s for s in r["trace"] if s["step"] == "tool:draft_findings")
    assert step["detail"]["added_from_rules"] == 1


def test_whisper_tooth_numbers_with_hyphen():
    from app.dental import find_sites

    assert [s.fdi for s in find_sites("Possible caries on tooth 4-6")] == ["46"]
    assert [s.fdi for s in find_sites("No radiolucency in relation to tooth 3 6")] == ["36"]
    assert [s.fdi for s in find_sites("review in 4-6 weeks")] == []


def _whisper(text: str, segments: list[dict[str, Any]]) -> OpenAICompatSTT:
    create = lambda **kw: SimpleNamespace(text=text, segments=segments)  # noqa: E731
    fake = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    return OpenAICompatSTT(None, "whisper-large-v3-turbo", name="groq", client=fake)


def test_whisper_silence_hallucination_is_rejected():
    import pytest

    from app.stt import STTError

    # what the live run actually produced for a silent clip
    with pytest.raises(STTError, match="No speech"):
        silent = _whisper("Thank you.", [{"text": "Thank you.", "no_speech_prob": 0.12, "avg_logprob": -0.4}])
        silent.transcribe(b"x", "a.wav", "audio/wav")
    # a segment Whisper itself marks as probably-not-speech is dropped, real speech is kept
    mixed = _whisper("Summarize her report. Thanks for watching!", [
        {"text": "Summarize her report.", "no_speech_prob": 0.01, "avg_logprob": -0.2},
        {"text": "Thanks for watching!", "no_speech_prob": 0.83, "avg_logprob": -0.9}])
    text, meta = mixed.transcribe_with_meta(b"x", "a.wav", "audio/wav")
    assert text == "Summarize her report." and meta["dropped_no_speech"] == 1


def test_whisper_request_uses_verbose_json():
    sent: dict[str, Any] = {}

    def create(**kw: Any) -> Any:
        sent.update(kw)
        said = "Open Priya Sharma's latest OPG report"
        return SimpleNamespace(text=said, segments=[{"text": said, "no_speech_prob": 0.02, "avg_logprob": -0.15}])

    fake = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    stt = OpenAICompatSTT(None, "whisper-large-v3-turbo", name="groq", client=fake)
    client = TestClient(create_app(Settings(), stt=stt))
    body = client.post("/api/assistant/voice", files={"audio": ("mic.webm", b"opus", "audio/webm")}).json()
    assert sent["response_format"] == "verbose_json"
    assert body["type"] == "report" and body["trace"][0]["detail"]["max_no_speech_prob"] == 0.02


def test_report_request_actions_map_to_internal_tools():
    from app.interpreter.schema import to_internal_call

    assert to_internal_call("report_request", {"action": "summarize", "patient_reference": "pronoun_female",
                                               "question": None}, "x")[0] == "summarize_report"
    name, args = to_internal_call("report_request", {"action": "question", "patient_reference": "pronoun_female"},
                                  "Does she have any caries?")
    assert name == "answer_report_question" and args["question"] == "Does she have any caries?"
    fake = FakeChat("report_request", {"action": "teleport", "patient_reference": "none"})
    client = TestClient(create_app(Settings(), interpreter=groq(fake)))
    r = client.post("/api/assistant/message", json={"text": "Open Priya Sharma's latest OPG report"}).json()
    assert any(s["step"] == "interpret_fallback" for s in r["trace"]) and r["type"] == "report"


def test_misattributed_qualifier_is_dropped():
    text = "Lower right third molar is horizontally impacted with roots close to the mandibular canal."
    fake = FakeChat("draft_findings", {"findings": [
        {"category": "impaction", "qualifiers": ["horizontal", "roots close to the mandibular canal"],
         "tooth_fdi": "48", "present": True, "certainty": "stated",
         "evidence": "Lower right third molar is horizontally impacted"}]})
    client = TestClient(create_app(Settings(), interpreter=groq(fake)))
    d = client.post("/api/assistant/message", json={"text": text}).json()["data"]
    imp = d["findings"][0]
    assert imp["qualifiers"] == ["horizontal"]
    assert any("not dictated for this finding" in r for r in imp["review_reasons"])
    assert ("canal_proximity", "48") in [(f["category"], f["site"]["fdi"]) for f in d["findings"]]


def test_spoken_name_wins_over_llm_pronoun():
    """Live run: the model read "Priya's" as a pronoun. A dropped name must not fall back to session context."""
    fake = FakeChat("report_request", {"action": "open", "patient_reference": "pronoun_female", "modality": "OPG",
                                       "which": "latest"})
    client = TestClient(create_app(Settings(), interpreter=groq(fake)))
    r = client.post("/api/assistant/message", json={"text": "Open Priya's latest OPG report"}).json()
    assert r["type"] == "clarification" and [c["patient_id"] for c in r["data"]["candidates"]] == ["P001", "P002"]
    assert any(s["step"] == "name_cross_check" for s in r["trace"])


def test_dropped_name_cannot_open_the_wrong_patient():
    say = lambda c, text, sid=None: c.post("/api/assistant/message", json={"text": text, "session_id": sid}).json()  # noqa: E731
    opener = FakeChat("report_request", {"action": "open", "patient_name": "Rahul Menon", "patient_reference": "named",
                                         "modality": "OPG", "which": "latest"})
    interp = groq(opener)
    client = TestClient(create_app(Settings(), interpreter=interp))
    sid = say(client, "Open Rahul Menon's latest OPG")["session_id"]                       # a male patient is open
    interp.client = FakeChat("report_request", {"action": "open", "patient_reference": "pronoun_male",
                                                "modality": "OPG", "which": "latest"})      # model drops "Arjun"
    r = say(client, "Open Arjun Nair's OPG", sid)
    assert r["data"]["patient"]["patient_id"] == "P004" and r["data"]["report"]["report_id"] == "R-4001"


def test_qualifier_wording_is_canonicalised():
    fake = FakeChat("draft_findings", {"findings": [
        {"category": "impaction", "qualifiers": ["horizontally"], "tooth_fdi": "48", "present": True,
         "certainty": "stated", "evidence": "Lower right third molar is horizontally impacted"}]})
    client = TestClient(create_app(Settings(), interpreter=groq(fake)))
    text = "Lower right third molar is horizontally impacted"
    d = client.post("/api/assistant/message", json={"text": text}).json()["data"]
    assert d["findings"][0]["qualifiers"] == ["horizontal"]
