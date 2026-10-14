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


def test_question_words_are_not_read_as_patient_names(say):
    """Offline interpreter: topic words in a follow-up ("tumour", "healthy", "diagnosis") must not be
    taken as a patient name. Each question stays on the open patient and is answered from the report."""
    expected = {
        "Does she have a tumour?": "unsupported",                 # not documented: can't confirm or rule out
        "Does she have oral leukoplakia?": "unsupported",         # unknown term after "she": pronoun wins
        "Is she healthy?": "unsupported",
        "Has she got bone loss?": "answer",
        "What is the diagnosis?": "answer",                       # the report's impression, quoted
        "Is the impaction mesioangular?": "answer",
        "How is her 36 doing?": "answer",
        "Is there evidence of infection?": "answer",
        "Does she need an extraction?": "answer",                 # the report's recommendation, quoted
        "Should we extract 48?": "answer",                        # a question, not dictation
        "Is there anything concerning in her report?": "summary",
        "Is 48 close to the nerve?": "answer",                    # anatomy words are not names either
        "Is there a crown on 46?": "answer",
        "Can you check her 46?": "answer",
        "What was found on the OPG?": "summary",
        "Read me the recommendation": "answer",
    }
    sid = say("Open Priya Sharma's latest OPG report")["session_id"]
    for question, rtype in expected.items():
        r = say(question, sid)
        assert "No patient named" not in r["message"], question
        assert r["type"] == rtype, (question, r["type"], r["message"])
        assert r["data"]["patient"]["patient_id"] == "P001", question
    r = say("Does she need an extraction?", sid)
    assert "Surgical consultation for 48" in r["message"] and r["data"]["fields"] == ["recommendation"]
    r = say("Does Priya Shetty have caries?", sid)                # a real name after the verb still wins
    assert r["data"]["patient"]["patient_id"] == "P002"


def test_unknown_word_in_a_followup_offers_the_open_patient(say):
    """An unrecognised word is never silently swapped for the open patient: the assistant asks first."""
    sid = say("Open Priya Sharma's latest OPG report")["session_id"]
    r = say("Does Kavya have caries?", sid)
    assert r["type"] == "clarification" and "Kavya" in r["message"] and "Priya Sharma (P001)" in r["message"]
    assert [c["patient_id"] for c in r["data"]["candidates"]] == ["P001"]
    r = say("yes", sid)
    assert r["type"] == "answer" and r["data"]["patient"]["patient_id"] == "P001"
    assert "does not mention caries" in r["message"]
    r = say("Open Kavya's OPG", sid)                               # a direct open is not redirected
    assert r["type"] == "not_found" and "No patient named" in r["message"]
