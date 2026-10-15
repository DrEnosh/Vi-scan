"""Required test 3: unsupported clinical claims never reach the draft or the answer.

Covers both engines: the deterministic parser, and an LLM that hallucinates (simulated with a
scripted LLM client) whose output must be filtered by the grounding validator.
"""

FORBIDDEN_ADDITIONS = ["caries", "extraction", "recommend", "surgery", "cyst", "infection", "pericoronitis", "nerve"]


def test_dictation_adds_nothing_beyond_what_was_said(say):
    text = "Lower right third molar is horizontally impacted"
    r = say(text)
    assert r["type"] == "draft_report"
    d = r["data"]
    assert len(d["findings"]) == 1
    f = d["findings"][0]
    assert (f["category"], f["site"]["fdi"], f["qualifiers"], f["present"]) == ("impaction", "48", ["horizontal"], True)
    assert f["evidence"] == text
    assert d["requires_clinician_review"] is True and d["status"] == "draft_pending_review"
    body = d["draft_text"].lower()
    for word in FORBIDDEN_ADDITIONS:
        assert word not in body, f"draft invented '{word}'"
    assert "impression: not generated" in body


def test_negation_and_hedging_are_preserved(say):
    d = say("Possible caries on 46. No periapical radiolucency in relation to 36.")["data"]
    by_cat = {f["category"]: f for f in d["findings"]}
    assert by_cat["caries"]["certainty"] == "uncertain" and by_cat["caries"]["needs_review"] is True
    assert by_cat["periapical"]["present"] is False and by_cat["periapical"]["site"]["fdi"] == "36"


def test_unrecognised_speech_is_kept_verbatim_not_interpreted(say):
    d = say("Lower right third molar is horizontally impacted. Patient is anxious about the procedure",
            mode="dictation")["data"]
    assert [f["category"] for f in d["findings"]] == ["impaction"]
    assert d["unparsed_segments"] == ["Patient is anxious about the procedure"]


def test_unspecified_side_is_not_guessed(say):
    f = say("Lower third molar is impacted")["data"]["findings"][0]
    assert f["site"]["type"] == "unresolved" and f["site"]["fdi"] is None
    assert f["site"]["candidates"] == ["38", "48"] and f["needs_review"] is True


def test_question_about_undocumented_finding_is_not_answered_yes(say):
    r1 = say("Open Priya Sharma's latest OPG report")             # R-1003 has no caries finding
    r2 = say("Does she have any caries?", r1["session_id"])
    assert r2["type"] == "answer"
    assert r2["data"]["matched_findings"] == []
    assert "does not mention caries" in r2["message"]
    assert "R-1002" in r2["message"]                               # points to where it IS documented


def test_llm_hallucinated_findings_are_rejected(llm_app, llm_say):
    text = "Lower right third molar is horizontally impacted"
    client, _ = llm_app(tool_name="draft_findings", tool_input={"findings": [
        {"category": "impaction", "qualifiers": ["horizontal"], "tooth_fdi": "48", "present": True,
         "certainty": "stated", "evidence": "Lower right third molar is horizontally impacted"},
        {"category": "caries", "tooth_fdi": "46", "present": True, "certainty": "stated",
         "evidence": "46 shows deep caries"},                                  # quote not in transcript
        {"category": "canal_proximity", "tooth_fdi": "48", "present": True, "certainty": "stated",
         "evidence": "Lower right third molar is horizontally impacted"},      # real quote, wrong claim
        {"category": "impaction", "tooth_fdi": "38", "present": True, "certainty": "stated",
         "evidence": "Lower right third molar is horizontally impacted"},      # wrong tooth
    ]})
    r = llm_say(client)(text)
    d = r["data"]
    assert d["extraction_engine"] == "llm+grounding"
    assert [(f["category"], f["site"]["fdi"]) for f in d["findings"]] == [("impaction", "48")]
    reasons = [x["reason"] for x in d["rejected_findings"]]
    assert len(reasons) == 3
    assert any("does not appear in the transcript" in x for x in reasons)
    assert any("does not mention root proximity" in x for x in reasons)
    assert any("Site mismatch" in x for x in reasons)
    assert "caries" not in d["draft_text"].lower()


def test_llm_wrong_negation_and_invented_qualifier_are_corrected(llm_app, llm_say):
    text = "Possible caries on 46. No periapical radiolucency in relation to 36."
    client, _ = llm_app(tool_name="draft_findings", tool_input={"findings": [
        {"category": "caries", "qualifiers": ["deep"], "tooth_fdi": "46", "present": True, "certainty": "stated",
         "evidence": "caries on 46"},
        {"category": "periapical", "tooth_fdi": "36", "present": True, "certainty": "stated",
         "evidence": "periapical radiolucency in relation to 36"},
    ]})
    d = llm_say(client)(text)["data"]
    caries, pa = d["findings"]
    assert caries["qualifiers"] == [] and caries["certainty"] == "uncertain"   # "deep" was never said
    assert pa["present"] is False                                               # the dictation said "No"
    assert any("disagreed" in x for x in pa["review_reasons"])


def test_llm_outage_falls_back_to_rules(llm_app, llm_say):
    client, _ = llm_app(error=TimeoutError("simulated timeout"))
    r = llm_say(client)("Open Priya Sharma's latest OPG report")
    assert r["type"] == "report" and r["data"]["report"]["report_id"] == "R-1003"
    assert any(s["step"] == "interpret_fallback" for s in r["trace"])


def test_llm_invalid_tool_arguments_are_not_executed(llm_app, llm_say):
    client, _ = llm_app(tool_name="report_request", tool_input={"action": "open", "patient_name": "Priya Sharma",
                                                                  "patient_reference": "named",
                                                               "which": "latest", "modality": "XRAY"})
    r = llm_say(client)("Open Priya Sharma's latest OPG report")
    validate = [s for s in r["trace"] if s["step"] == "validate"]
    assert validate[0]["detail"]["ok"] is False                  # bad enum caught before any tool ran
    assert r["data"]["report"]["report_id"] == "R-1003"          # rule-based re-interpretation


def test_spoken_tooth_numbers_from_stt(say):
    """Whisper often writes FDI numbers as words ('four eight', 'thirty-six'); ages must not become teeth."""
    d = say("Tooth four eight is horizontally impacted. No periapical radiolucency in relation to thirty-six. "
            "Patient is twenty years old.", mode="dictation")["data"]
    got = {(f["category"], f["site"]["fdi"], f["present"]) for f in d["findings"]}
    assert got == {("impaction", "48", True), ("periapical", "36", False)}


def test_duplicate_llm_findings_are_merged(llm_app, llm_say):
    """Two proposals for the same finding become one line, the cautious reading wins, and a vaguer
    category for the same tooth ("radiolucency" next to "periapical radiolucency") is dropped."""
    text = "No caries on 46. Possible periapical radiolucency in relation to 36."
    client, _ = llm_app(tool_name="draft_findings", tool_input={"findings": [
        {"category": "caries", "tooth_fdi": "46", "present": False, "certainty": "stated", "evidence": "No caries on 46"},
        {"category": "caries", "tooth_fdi": "46", "present": True, "certainty": "stated", "evidence": "caries on 46"},
        {"category": "periapical", "tooth_fdi": "36", "present": True, "certainty": "stated",
         "evidence": "periapical radiolucency in relation to 36"},
        {"category": "periapical", "tooth_fdi": "36", "present": True, "certainty": "uncertain",
         "evidence": "Possible periapical radiolucency in relation to 36"},
        {"category": "radiolucency", "tooth_fdi": "36", "present": True, "certainty": "uncertain",     # less specific
         "evidence": "Possible periapical radiolucency in relation to 36"}]})
    d = llm_say(client)(text)["data"]
    got = [(f["category"], f["site"]["fdi"], f["present"], f["certainty"]) for f in d["findings"]]
    assert got == [("caries", "46", False, "stated"), ("periapical", "36", True, "uncertain")]
    assert d["draft_text"].count("Caries") + d["draft_text"].count("caries") == 1
