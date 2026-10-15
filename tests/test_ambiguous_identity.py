"""Required test 2: ambiguous patient identity is never resolved by guessing."""
import json


def test_first_name_only_asks_which_patient(say):
    r = say("Open Priya's latest OPG report")
    assert r["type"] == "clarification"
    assert [c["patient_id"] for c in r["data"]["candidates"]] == ["P001", "P002"]
    assert "report" not in (r["data"] or {})                       # nothing opened
    assert r["evidence"] == []


def test_clarification_by_full_name_runs_original_request(say):
    r1 = say("Open Priya's latest OPG report")
    r2 = say("Priya Shetty", r1["session_id"])
    assert r2["type"] == "report"
    assert r2["data"]["patient"]["patient_id"] == "P002"
    assert r2["data"]["report"]["report_id"] == "R-2001"          # Priya Shetty's latest OPG


def test_clarification_by_ordinal_and_by_id(say):
    r1 = say("Open Priya's latest OPG report")
    assert say("the first one", r1["session_id"])["data"]["patient"]["patient_id"] == "P001"
    r3 = say("Open Priya's latest OPG report")
    assert say("P002", r3["session_id"])["data"]["patient"]["patient_id"] == "P002"


def test_shared_surname_is_ambiguous(say):
    r = say("Open Rao's latest report")
    assert r["type"] == "clarification"
    assert {c["patient_id"] for c in r["data"]["candidates"]} == {"P006", "P007"}


def test_fuzzy_name_needs_confirmation(say):
    r1 = say("Open Pria Sharma's OPG")                              # STT-style misspelling
    assert r1["type"] == "clarification"
    assert r1["data"]["kind"] == "confirm_patient"
    r2 = say("yes", r1["session_id"])
    assert r2["type"] == "report" and r2["data"]["report"]["report_id"] == "R-1003"


def test_cancel_leaves_no_context(say):
    r1 = say("Open Priya's latest OPG report")
    assert say("cancel", r1["session_id"])["type"] == "cancelled"
    r3 = say("Summarize her report", r1["session_id"])
    assert r3["type"] == "clarification"                            # no patient was ever opened


def test_new_request_replaces_pending_question(say):
    r1 = say("Open Priya's latest OPG report")
    r2 = say("Open Rahul Menon's latest OPG", r1["session_id"])
    assert r2["type"] == "report" and r2["data"]["patient"]["patient_id"] == "P003"


def test_exact_match_still_warns_about_sound_alike(say):
    r = say("Open Ananya Rao's latest OPG")
    assert r["type"] == "report" and r["data"]["patient"]["patient_id"] == "P006"
    assert any("Anaya Rao (P007)" in w for w in r["warnings"])


def test_llm_cannot_pick_a_patient_who_was_not_named(llm_app, llm_say):
    """The patient comes from what was said. A name or ID the model adds on its own is discarded."""
    base = {"action": "open", "patient_reference": "named", "modality": "OPG", "which": "latest"}
    cases = [
        ("Open Priya Sharma's latest OPG report", {"patient_name": "Priya Shetty"}, "R-1003"),   # swapped look-alike
        ("Open Priya Sharma's latest OPG report", {"patient_id": "P002"}, "R-1003"),             # invented ID
        ("Open Rahul Menon's OPG", {"patient_name": "Arjun Nair"}, "R-3001"),                    # different patient
    ]
    for text, wrong, report_id in cases:
        client, _ = llm_app(tool_name="report_request", tool_input={**base, **wrong})
        r = llm_say(client)(text)
        assert r["data"]["report"]["report_id"] == report_id, (text, wrong)
        assert any(s["step"] == "identity_check" for s in r["trace"])
    # the clinician said "Pria": the model's corrected spelling must not skip the confirmation
    client, _ = llm_app(tool_name="report_request", tool_input={**base, "patient_name": "Priya Sharma"})
    r = llm_say(client)("Open Pria Sharma's latest OPG report")
    assert r["type"] == "clarification" and "Did you mean Priya Sharma (P001)" in r["message"]
    # a name the model invents when only a pronoun was said is dropped, so the assistant asks
    client, _ = llm_app(tool_name="report_request", tool_input={"action": "summarize", "patient_reference": "named",
                                                                "patient_name": "Priya Shetty"})
    r = llm_say(client)("Summarize her report")
    assert r["type"] == "clarification" and "P002" not in json.dumps(r["data"])
