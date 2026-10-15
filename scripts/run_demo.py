"""Runs a scripted demo against the real app (in-process) and writes docs/DEMO.md.

    python scripts/run_demo.py

Every response in DEMO.md is produced by this run; nothing is hand-written. The last section
uses a scripted stand-in for the LLM so the grounding validator can be shown rejecting
hallucinated findings without an API key.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.interpreter import OpenAICompatInterpreter  # noqa: E402
from app.main import create_app  # noqa: E402

OUT = ROOT / "docs" / "DEMO.md"
lines: list[str] = []


def trim(resp: dict[str, Any]) -> dict[str, Any]:
    """Keep what a reviewer needs; the full payload is available from the API."""
    if "type" not in resp:  # draft endpoints return the draft itself
        return trim_draft(resp)
    out: dict[str, Any] = {"session_id": resp["session_id"], "type": resp["type"], "message": resp["message"]}
    if resp.get("transcript"):
        out["transcript"] = resp["transcript"]
    if resp.get("warnings"):
        out["warnings"] = resp["warnings"]
    if resp.get("evidence"):
        draft = resp["type"] == "draft_report"
        out["evidence"] = f"{len(resp['evidence'])} transcript quote(s)" if draft else resp["evidence"]
    d = resp.get("data") or {}
    if resp["type"] == "report":
        r = d["report"]
        out["data"] = {"patient": d["patient"],
                       "report": {k: r[k] for k in ("report_id", "modality", "study_date", "region")},
                       "findings": [f"{f['id']} {f['site']}: {f['text']}" for f in r["findings"]]}
    elif resp["type"] == "clarification":
        out["data"] = d
    elif resp["type"] == "draft_report":
        out["data"] = trim_draft(d)
    elif d:
        out["data"] = {k: v for k, v in d.items() if k not in ("points", "impression", "recommendation")}
    shown = ("interpret", "tool:", "context", "clarification", "validate", "stt")
    out["trace"] = [f"{s['step']}" + (f" ({s['ms']} ms)" if s.get("ms") is not None else "") +
                    (f" -> {json.dumps(s['detail'])}" if s["step"].startswith(shown) else "")
                    for s in resp.get("trace", [])]
    out["total_ms"] = resp.get("total_ms")
    return out


def trim_draft(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "draft_id": d["draft_id"], "status": d["status"], "version": d["version"],
        "extraction_engine": d["extraction_engine"],
        "patient": d["patient"]["patient_id"] if d.get("patient") else None,
        "findings": [{"id": f["finding_id"], "category": f["category"],
                      "site": f["site"]["label"] if f["site"] else None, "fdi": f["site"]["fdi"] if f["site"] else None,
                      "qualifiers": f["qualifiers"], "present": f["present"], "certainty": f["certainty"],
                      "evidence": f["evidence"], "needs_review": f["needs_review"],
                      "review_reasons": f["review_reasons"], "source": f["source"]} for f in d["findings"]],
        "unparsed_segments": d["unparsed_segments"],
        "rejected_findings": d["rejected_findings"],
        "draft_text": d["draft_text"].split("\n"),
    }


def section(title: str, note: str = "") -> None:
    lines.append(f"\n## {title}\n")
    if note:
        lines.append(note + "\n")


def record(client: TestClient, method: str, path: str, *, body: dict[str, Any] | None = None,
           clinician: str = "dr_demo", files: dict[str, Any] | None = None,
           form: dict[str, str] | None = None) -> dict[str, Any]:
    headers = {"X-Clinician-Id": clinician}
    if files:
        r = client.request(method, path, files=files, data=form or {}, headers=headers)
        fname = next(iter(files.values()))[0]
        curl = (f"curl -s -X {method} localhost:8000{path} -H 'X-Clinician-Id: {clinician}' "
                f"-F 'audio=@samples/{fname};type=audio/wav'"
                + "".join(f" -F '{k}={v}'" for k, v in (form or {}).items()))
    else:
        r = client.request(method, path, json=body, headers=headers)
        curl = f"curl -s -X {method} localhost:8000{path} -H 'X-Clinician-Id: {clinician}'"
        if body is not None:
            curl += f" -H 'Content-Type: application/json' -d {shlex.quote(json.dumps(body))}"
    data = r.json()
    lines.append(f"```bash\n{curl}\n```")
    lines.append(f"HTTP {r.status_code}\n")
    shown = trim(data) if r.status_code == 200 else data
    lines.append("```json\n" + json.dumps(shown, indent=2, ensure_ascii=False) + "\n```\n")
    return data


def say(client: TestClient, text: str, sid: str | None = None, clinician: str = "dr_demo",
        mode: str = "auto") -> dict[str, Any]:
    body: dict[str, Any] = {"text": text}
    if sid:
        body["session_id"] = sid
    if mode != "auto":
        body["mode"] = mode
    return record(client, "POST", "/api/assistant/message", body=body, clinician=clinician)


def conversation(client: TestClient) -> None:
    """Sections 1-5: retrieval, clarification, follow-ups and grounded questions."""
    section("1. Retrieval with an ambiguous name",
            "Two patients are called Priya in this clinic, so the assistant asks instead of guessing. "
            "Nothing is opened yet.")
    r = say(client, "Open Priya's latest OPG report")
    sid = r["session_id"]
    section("2. Clarification answered, original request completes",
            "The answer \"Priya Sharma\" resolves the pending question and the original intent (latest OPG) runs. "
            "Note the look-alike name alert.")
    say(client, "Priya Sharma", sid)

    section("3. Follow-up without repeating the patient",
            "\"her\" resolves from session context (see `context_check` in the trace). The summary is extractive: "
            "every line is copied from a report field and keeps its finding ID.")
    r = say(client, "Summarize her report", sid)
    lines.append("Full summary text returned in `message`:\n")
    lines.append("```text\n" + r["message"] + "\n```\n")

    section("4. Grounded questions, including ones the report can't answer",
            "R-1003 has no caries finding, so the answer says so and points to the reports that do, "
            "rather than saying yes or no. A question about something the report doesn't cover (a tumour) is "
            "neither confirmed nor denied, and a treatment question gets the report's own recommendation, quoted.")
    say(client, "Does she have any caries?", sid)
    say(client, "What about 48?", sid)
    say(client, "Does she have a tumour?", sid)
    say(client, "Does she need an extraction?", sid)

    section("5. Pronoun that doesn't fit the open patient",
            "\"his\" while a female patient is open triggers a question.")
    say(client, "Summarize his report", sid)
    say(client, "cancel", sid)


def dictation(client: TestClient) -> None:
    """Sections 6-8: dictation to a draft, then edit and approval."""
    section("6. Dictation to structured draft (assignment example)",
            "One sentence in, one finding out, with FDI tooth number, the exact evidence quote, and no impression.")
    r = say(client, "Lower right third molar is horizontally impacted")

    section("7. Dictation with negation, hedging, an unspecified side and off-vocabulary speech",
            "\"No ...\" becomes `present: false`, \"Possible\" becomes `uncertain` and is flagged, "
            "\"lower third molar\" is not guessed to be 38 or 48, and the sentence the parser cannot map is kept "
            "verbatim.")
    r = say(client, "Lower third molar is impacted. Possible caries on 46. "
                    "No periapical radiolucency in relation to 36. Patient is anxious about the procedure.",
            mode="dictation")
    did = r["data"]["draft_id"]

    section("8. Clinician edits the JSON, links the patient, approves",
            "Approval is blocked while no patient is linked or a tooth is unresolved.")
    record(client, "POST", f"/api/drafts/{did}/approve")
    record(client, "PUT", f"/api/drafts/{did}", body={"patient_id": "P001", "findings": [
        {"finding_id": "D1", "category": "impaction", "qualifiers": ["horizontal"], "present": True,
         "certainty": "stated", "tooth_fdi": "48"},
        {"finding_id": "D2", "category": "caries", "present": True, "certainty": "uncertain", "tooth_fdi": "46"},
        {"finding_id": "D3", "category": "periapical", "present": False, "certainty": "stated", "tooth_fdi": "36"}]})
    record(client, "POST", f"/api/drafts/{did}/approve")


def voice_and_scope(client: TestClient) -> None:
    """Sections 9-10: the voice path, a missing report and clinic scope."""
    section("9. Voice path: audio upload -> STT -> same pipeline",
            "With `STT_PROVIDER=mock` the sample file name maps to a fixed transcript; with `STT_PROVIDER=groq` the "
            "same endpoint sends the audio to Whisper (see `docs/LIVE_RUN.md`). "
            "The `stt` step appears first in the trace.")
    record(client, "POST", "/api/assistant/voice", files={"audio": ("dictation_multi.wav", b"RIFF-demo", "audio/wav")},
           form={"mode": "auto"})

    section("10. Missing report and authorization scope")
    say(client, "Open Arjun Nair's CBCT")
    lines.append("Priya Shah (P008) belongs to another clinic. For `dr_demo` she does not exist (closest in-scope name "
                 "is offered for confirmation, P008 never appears); for `dr_mysuru` she opens normally.\n")
    say(client, "Open Priya Shah's OPG")
    say(client, "Open Priya Shah's OPG", clinician="dr_mysuru")


def llm_grounding() -> None:
    """Section 11: a scripted model proposes bad findings and the validator rejects them."""
    section("11. LLM path: hallucinated findings are rejected by the grounding validator",
            "This call uses a scripted stand-in for the LLM tool call (no API key needed) that returns one correct "
            "finding and three bad ones: a quote that was never said, a real quote attached to a claim it doesn't "
            "support, and the wrong tooth. Only the grounded finding survives.")
    proposed = {"findings": [
        {"category": "impaction", "qualifiers": ["horizontal"], "tooth_fdi": "48", "present": True,
         "certainty": "stated", "evidence": "Lower right third molar is horizontally impacted"},
        {"category": "caries", "tooth_fdi": "46", "present": True, "certainty": "stated",
         "evidence": "46 shows deep caries"},
        {"category": "canal_proximity", "tooth_fdi": "48", "present": True, "certainty": "stated",
         "evidence": "Lower right third molar is horizontally impacted"},
        {"category": "impaction", "tooth_fdi": "38", "present": True, "certainty": "stated",
         "evidence": "Lower right third molar is horizontally impacted"}]}
    lines.append("Simulated model output (`draft_findings` tool call):\n")
    lines.append("```json\n" + json.dumps(proposed, indent=2) + "\n```\n")
    tool_call = SimpleNamespace(function=SimpleNamespace(name="draft_findings", arguments=json.dumps(proposed)))
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call], content=None))],
        usage=SimpleNamespace(prompt_tokens=861, completion_tokens=210, prompt_tokens_details=None)))))
    scripted = OpenAICompatInterpreter(None, "openai/gpt-oss-120b", None, client=fake, provider="scripted-llm")
    llm_client = TestClient(create_app(Settings(), interpreter=scripted))
    say(llm_client, "Lower right third molar is horizontally impacted")


def main() -> None:
    client = TestClient(create_app(Settings()))
    lines.append("# Demo: documented API calls with real outputs\n")
    lines.append("Generated by `python scripts/run_demo.py` against the app with the default settings "
                 "(`INTERPRETER=rule_based`, `STT_PROVIDER=mock`). Responses are trimmed for reading: the full "
                 "payload also carries every report field and per-step trace details. All data is synthetic.\n")
    lines.append("Session IDs are random, so they differ between runs.\n")

    conversation(client)
    dictation(client)
    voice_and_scope(client)
    llm_grounding()

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} blocks)")


if __name__ == "__main__":
    main()
