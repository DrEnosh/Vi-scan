"""Offline name finding: a word is part of a patient name only with evidence for it."""
import pytest

from app.interpreter.rule_based import find_name

CLINIC = ["Priya Sharma", "Priya Shetty", "Rahul Menon", "Arjun Nair", "Fatima Khan", "Ananya Rao", "Anaya Rao",
          "Kiran Das"]


@pytest.mark.parametrize("text, name", [
    ("Open Priya Sharma's latest OPG", "priya sharma"),
    ("open priya sharmas latest opg", "priya sharma"),           # apostrophe lost in typing or transcription
    ("Open the latest report for Rahul Menon", "rahul menon"),
    ("Does Arjun have any caries?", "arjun"),
    ("Pull up Fatima Khan please", "fatima khan"),
    ("Any caries for Kiran Das?", "kiran das"),
    ("Does Priya Sharma currently have bone loss?", "priya sharma"),
    ("Open Pria Sharma's OPG", "pria sharma"),                   # kept as said; the patient match asks to confirm
])
def test_names_on_the_patient_list_are_found(text, name):
    assert find_name(text, CLINIC) == name


@pytest.mark.parametrize("text", [
    "Is malignancy ruled out?", "What happened to 36?", "Any signs of infection?", "Does she need surgery?",
    "Should we refer her for surgery?", "Is the lesion growing?", "Who reported this scan?",
    "What's the weather like?", "Open the patient's latest scan", "Show last month's OPG", "What did Dr. Rao say?",
])
def test_unknown_words_without_evidence_are_not_names(text):
    assert find_name(text, CLINIC) is None


@pytest.mark.parametrize("text, name", [
    ("Does Kavya have caries?", "kavya"),                        # subject of "does ... have"
    ("does kavya reddy have bone loss", "kavya reddy"),
    ("Open Meera's report", "meera"),                            # possessive
    ("open patient deepak's scan", "deepak"),                    # after "patient"
    ("Show me the report for Meera", "meera"),                   # capitalised mid-sentence
    ("Open Rohan latest OPG", "rohan"),                          # object of "open"
])
def test_unknown_names_in_name_positions_are_still_names(text, name):
    """A name that is not on this clinic's list must still be heard, so the assistant can say it found
    no such patient instead of quietly answering about the patient who is open."""
    assert find_name(text, CLINIC) == name


def test_answer_to_which_patient_is_read_as_a_name():
    assert find_name("Shetty", CLINIC, bare=True) == "shetty"
    assert find_name("I mean Priya Shetty", CLINIC, bare=True) == "priya shetty"
    assert find_name("kavya", CLINIC, bare=True) == "kavya"


def test_follow_up_with_new_clinical_word_stays_on_the_open_patient(say):
    sid = say("Open Priya Sharma's latest OPG report")["session_id"]
    for q in ("Is malignancy ruled out?", "What happened to 36?", "Any signs of osteomyelitis?"):
        r = say(q, sid)
        assert "patient named" not in r["message"], q
        assert r["data"]["patient"]["patient_id"] == "P001", q
    r = say("Does Kavya have caries?", sid)                      # an unknown name is not answered for Priya
    assert r["type"] == "clarification" and "Kavya" in r["message"]
