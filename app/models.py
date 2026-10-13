"""Pydantic schemas shared across the app: domain records, draft reports and API payloads."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Modality = Literal["OPG", "IOPA", "CBCT", "LATERAL_CEPH", "BITEWING"]

MODALITY_LABELS: dict[str, str] = {
    "OPG": "OPG",
    "IOPA": "IOPA",
    "CBCT": "CBCT",
    "LATERAL_CEPH": "Lateral ceph",
    "BITEWING": "Bitewing",
}


# --------------------------------------------------------------------------- domain records
class Clinician(BaseModel):
    clinician_id: str
    name: str
    clinic_ids: list[str]


class Patient(BaseModel):
    patient_id: str
    name: str
    sex: Literal["F", "M", "O"]
    dob: date
    clinic_id: str

    def age_on(self, today: date | None = None) -> int:
        today = today or date.today()
        return today.year - self.dob.year - ((today.month, today.day) < (self.dob.month, self.dob.day))

    def card(self) -> dict[str, Any]:
        """Minimal identity block shown to the clinician (two identifiers: ID + age/sex)."""
        return {"patient_id": self.patient_id, "name": self.name, "age": self.age_on(), "sex": self.sex}


class ReportFinding(BaseModel):
    id: str
    site: str
    category: str
    present: bool = True
    text: str


class Report(BaseModel):
    report_id: str
    patient_id: str
    modality: Modality
    study_date: date
    region: str
    reported_by: str
    status: str
    findings: list[ReportFinding]
    impression: str
    recommendation: str

    def headline(self) -> str:
        return f"{MODALITY_LABELS[self.modality]} dated {self.study_date.isoformat()} ({self.report_id})"


# --------------------------------------------------------------------------- dictation drafts
class Site(BaseModel):
    type: Literal["tooth", "region", "generalized", "unresolved"]
    fdi: str | None = None
    label: str
    candidates: list[str] = Field(default_factory=list)


class DraftFinding(BaseModel):
    finding_id: str
    category: str
    finding: str
    qualifiers: list[str] = Field(default_factory=list)
    present: bool = True
    certainty: Literal["stated", "uncertain"] = "stated"
    certainty_cue: str | None = None
    site: Site | None = None
    evidence: str
    site_evidence: str | None = None
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    source: Literal["rule_parser", "llm", "clinician_edit"] = "rule_parser"
    statement: str = ""


class RejectedFinding(BaseModel):
    proposed: dict[str, Any]
    reason: str


class DraftReport(BaseModel):
    draft_id: str
    status: Literal["draft_pending_review", "edited_pending_review", "clinician_approved"]
    created_by: str
    created_at: datetime
    updated_at: datetime
    patient: dict[str, Any] | None = None
    transcript: str
    findings: list[DraftFinding]
    unparsed_segments: list[str] = Field(default_factory=list)
    rejected_findings: list[RejectedFinding] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    draft_text: str
    requires_clinician_review: bool = True
    extraction_engine: str
    version: int = 1
    approved_by: str | None = None
    approved_at: datetime | None = None


# --------------------------------------------------------------------------- assistant responses
ResponseType = Literal[
    "report", "clarification", "not_found", "summary", "answer", "draft_report", "unsupported", "cancelled"
]


class Evidence(BaseModel):
    source: Literal["report", "transcript"]
    report_id: str | None = None
    finding_ids: list[str] = Field(default_factory=list)
    quote: str | None = None


class TraceStep(BaseModel):
    step: str
    detail: dict[str, Any] = Field(default_factory=dict)
    ms: float | None = None


class AssistantResponse(BaseModel):
    session_id: str
    type: ResponseType
    message: str
    data: dict[str, Any] | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    transcript: str | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    total_ms: float | None = None


# --------------------------------------------------------------------------- API requests
class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    mode: Literal["auto", "command", "dictation"] = "auto"
    channel: Literal["text", "voice_browser"] = "text"

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        cleaned = "".join(ch for ch in v if ch.isprintable() or ch in "\n\t").strip()
        if not cleaned:
            raise ValueError("text is empty after removing control characters")
        return cleaned


class DraftFindingEdit(BaseModel):
    finding_id: str | None = None
    category: str
    finding: str | None = None
    qualifiers: list[str] = Field(default_factory=list)
    present: bool = True
    certainty: Literal["stated", "uncertain"] = "stated"
    tooth_fdi: str | None = None
    site_label: str | None = None
    evidence: str | None = None


class DraftUpdateRequest(BaseModel):
    findings: list[DraftFindingEdit]
    patient_id: str | None = None
