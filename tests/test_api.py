"""HTTP layer: voice path, input validation, auth and the draft edit/approve flow."""


def test_health_and_ui(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "Viscan Voice Assistant" in client.get("/").text


def test_voice_upload_goes_through_stt_then_assistant(client):
    files = {"audio": ("dictation_48_impacted.wav", b"RIFF....WAVEfmt fake-bytes", "audio/wav")}
    r = client.post("/api/assistant/voice", files=files, data={"mode": "auto"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] == "Lower right third molar is horizontally impacted"
    assert body["trace"][0]["step"] == "stt"
    assert body["type"] == "draft_report"


def test_voice_failures_are_clean_errors(client):
    unknown = client.post("/api/assistant/voice", files={"audio": ("random.wav", b"abc", "audio/wav")})
    assert unknown.status_code == 422 and unknown.json()["detail"]["error"] == "stt_failed"
    wrong = client.post("/api/assistant/voice", files={"audio": ("a.txt", b"abc", "text/plain")})
    assert wrong.status_code == 415
    empty = client.post("/api/assistant/voice", files={"audio": ("dictation_48_impacted.wav", b"", "audio/wav")})
    assert empty.status_code == 400


def test_input_validation(client):
    assert client.post("/api/assistant/message", json={"text": "   "}).status_code == 422
    assert client.post("/api/assistant/message", json={"text": "x" * 2001}).status_code == 422
    assert client.post("/api/assistant/message", json={"text": "hi", "session_id": "bad id!"}).status_code == 422
    assert client.post("/api/assistant/message", json={"text": "hi", "mode": "shout"}).status_code == 422


def test_unknown_clinician_is_rejected(client):
    r = client.post("/api/assistant/message", json={"text": "Open Priya Sharma's OPG"}, headers={"X-Clinician-Id": "intruder"})
    assert r.status_code == 401


def test_draft_edit_link_and_approve(client, say):
    d = say("Lower third molar is impacted. Possible caries on 46.")["data"]
    did = d["draft_id"]
    assert client.post(f"/api/drafts/{did}/approve").status_code == 409          # no patient linked yet
    edit = {"patient_id": "P001", "findings": [
        {"finding_id": "D1", "category": "impaction", "qualifiers": ["horizontal"], "present": True,
         "certainty": "stated", "tooth_fdi": "48"},
        {"finding_id": "D2", "category": "caries", "present": True, "certainty": "uncertain", "tooth_fdi": "46"},
    ]}
    r = client.put(f"/api/drafts/{did}", json=edit)
    assert r.status_code == 200, r.text
    edited = r.json()
    assert edited["status"] == "edited_pending_review" and edited["version"] == 2
    assert edited["findings"][0]["source"] == "clinician_edit" and edited["findings"][0]["site"]["fdi"] == "48"
    assert edited["findings"][0]["evidence"] == "Lower third molar is impacted"   # original evidence kept
    approved = client.post(f"/api/drafts/{did}/approve").json()
    assert approved["status"] == "clinician_approved" and approved["approved_by"] == "dr_demo"
    assert client.put(f"/api/drafts/{did}", json=edit).status_code == 409         # approved drafts are frozen


def test_draft_edit_validation_and_ownership(client, say):
    did = say("Lower right third molar is horizontally impacted")["data"]["draft_id"]
    bad = client.put(f"/api/drafts/{did}", json={"findings": [{"category": "impaction", "tooth_fdi": "99"}]})
    assert bad.status_code == 422
    other = client.get(f"/api/drafts/{did}", headers={"X-Clinician-Id": "dr_mysuru"})
    assert other.status_code == 404
