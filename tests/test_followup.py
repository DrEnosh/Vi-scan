"""Conversational follow-up: context carries over only when the reference is unambiguous."""


def test_summarize_her_report_uses_open_patient(say):
    r1 = say("Open Priya Sharma's latest OPG report")
    r2 = say("Summarize her report", r1["session_id"])
    assert r2["type"] == "summary"
    assert r2["data"]["report_id"] == "R-1003" and r2["data"]["patient"]["patient_id"] == "P001"
    ctx = [s for s in r2["trace"] if s["step"] == "context_check"]
    assert ctx and ctx[0]["detail"]["result"] == "resolved_from_session"
    assert r2["evidence"][0]["report_id"] == "R-1003"
    assert all(p["source"].startswith("R-1003#F") for p in r2["data"]["points"])


def test_summary_is_extractive(client, say):
    r1 = say("Open Priya Sharma's latest OPG report")
    r2 = say("Summarize her report", r1["session_id"])
    report = client.get("/api/reports/R-1003").json()
    texts = {f["id"]: f["text"] for f in report["findings"]}
    for p in r2["data"]["points"]:
        fid = p["source"].split("#")[1]
        assert p["text"].endswith(texts[fid])          # every line is copied from the report
    assert r2["data"]["impression"]["text"] == report["impression"]


def test_pronoun_that_does_not_fit_is_questioned(say):
    r1 = say("Open Priya Sharma's latest OPG report")
    r2 = say("Summarize his report", r1["session_id"])
    assert r2["type"] == "clarification" and "female" in r2["message"]


def test_followup_without_any_context_asks(say):
    r = say("Summarize her report")
    assert r["type"] == "clarification"


def test_followup_question_about_a_tooth(say):
    r1 = say("Open Priya Sharma's latest OPG report")
    r2 = say("What about 48?", r1["session_id"])
    assert r2["type"] == "answer" and r2["data"]["matched_findings"] == ["F3"]
    r3 = say("What was recommended?", r1["session_id"])
    assert r3["type"] == "answer" and "Surgical consultation" in r3["message"]


def test_context_is_not_shared_across_clinicians(say):
    r1 = say("Open Priya Sharma's latest OPG report")
    r2 = say("Summarize her report", r1["session_id"], clinician="dr_mysuru")
    assert r2["session_id"] != r1["session_id"]
    assert r2["type"] == "clarification"
