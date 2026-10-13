"""Required test 1: correct retrieval (plus missing-report and authorization edge cases)."""
import json


def test_open_latest_opg_by_full_name(say):
    r = say("Open Priya Sharma's latest OPG report")
    assert r["type"] == "report"
    assert r["data"]["patient"]["patient_id"] == "P001"
    assert r["data"]["report"]["report_id"] == "R-1003"          # newest OPG, not the newer CBCT
    assert r["evidence"][0]["report_id"] == "R-1003"
    steps = [s["step"] for s in r["trace"]]
    assert "tool:find_patient" in steps and "tool:get_report" in steps
    assert any("Priya Shetty" in w for w in r["warnings"])        # look-alike name alert


def test_previous_earliest_and_latest_any_modality(say):
    assert say("Show me Priya Sharma's previous OPG")["data"]["report"]["report_id"] == "R-1001"
    assert say("Open Priya Sharma's earliest report")["data"]["report"]["report_id"] == "R-1001"
    assert say("Open Priya Sharma's latest report")["data"]["report"]["report_id"] == "R-1004"  # CBCT is newest overall


def test_report_by_spoken_date(say):
    r = say("Open the CBCT for Priya Sharma from September 2026")
    assert r["data"]["report"]["report_id"] == "R-1004"


def test_lookup_by_patient_id(say):
    r = say("Open the latest OPG for patient P003")
    assert r["type"] == "report" and r["data"]["report"]["report_id"] == "R-3001"


def test_missing_modality_is_reported_with_alternatives(say):
    r = say("Open Arjun Nair's CBCT")
    assert r["type"] == "not_found"
    assert "No CBCT report" in r["message"]
    assert {x["report_id"] for x in r["data"]["available_reports"]} == {"R-4001", "R-4002"}


def test_patient_with_no_reports(say):
    r = say("Open Kiran Das's report")
    assert r["type"] == "not_found"
    assert "no reports on file" in r["message"]


def test_unknown_patient_is_not_guessed(say):
    r = say("Open Zoya's OPG")
    assert r["type"] == "not_found"
    assert r["data"] is None


def test_other_clinic_patient_is_invisible(say):
    r = say("Open Priya Shah's OPG")                    # P008 belongs to clinic MYS-02
    assert r["type"] != "report"
    assert "P008" not in json.dumps(r)                  # existence is not leaked
    own = say("Open Priya Shah's OPG", clinician="dr_mysuru")
    assert own["type"] == "report" and own["data"]["report"]["report_id"] == "R-8001"


def test_report_endpoint_enforces_clinic_scope(client):
    assert client.get("/api/reports/R-8001", headers={"X-Clinician-Id": "dr_demo"}).status_code == 404
    assert client.get("/api/reports/R-8001", headers={"X-Clinician-Id": "dr_mysuru"}).status_code == 200
