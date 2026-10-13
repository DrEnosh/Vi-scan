"""Tool-call contract between the interpreter and the deterministic tools.

The interpreter (LLM or rule-based) may only emit one of these tool calls. Arguments are
validated with Pydantic before any tool runs, so a malformed or invented call never touches
patient data.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ToolName = Literal["open_report", "summarize_report", "answer_report_question", "draft_findings", "unsupported_request"]
PatientReference = Literal["named", "pronoun_female", "pronoun_male", "pronoun_neutral", "none"]
ModalityArg = Literal["OPG", "IOPA", "CBCT", "LATERAL_CEPH", "BITEWING", "ANY"]


class ToolArgs(BaseModel):
    """Models often send `null` for arguments they don't have; treat that the same as leaving them out."""

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


class PatientRefArgs(ToolArgs):
    patient_name: str | None = Field(default=None, max_length=80)
    patient_id: str | None = Field(default=None, pattern=r"^[Pp]\d{3}$")
    patient_reference: PatientReference = "none"


class ReportQueryArgs(PatientRefArgs):
    modality: ModalityArg = "ANY"
    which: Literal["latest", "previous", "earliest", "by_date"] = "latest"
    date: str | None = Field(default=None, max_length=10)

    @property
    def report_specified(self) -> bool:
        return self.modality != "ANY" or self.which != "latest" or bool(self.date)


class QuestionArgs(PatientRefArgs):
    question: str = Field(..., min_length=1, max_length=500)


class LLMFinding(ToolArgs):
    category: str
    qualifiers: list[str] = Field(default_factory=list)
    tooth_fdi: str | None = None
    site_label: str | None = None
    present: bool = True
    certainty: Literal["stated", "uncertain"] = "stated"
    evidence: str = Field(..., min_length=1)

    @field_validator("tooth_fdi", mode="before")
    @classmethod
    def _fdi_as_text(cls, v: Any) -> Any:
        return str(v) if isinstance(v, int) else v


class DraftArgs(ToolArgs):
    findings: list[dict[str, Any]] | None = None   # None -> deterministic parser runs on the transcript


class UnsupportedArgs(ToolArgs):
    reason: str = ""


ARG_MODELS: dict[str, type[BaseModel]] = {
    "open_report": ReportQueryArgs,
    "summarize_report": ReportQueryArgs,
    "answer_report_question": QuestionArgs,
    "draft_findings": DraftArgs,
    "unsupported_request": UnsupportedArgs,
}


class ToolCall(BaseModel):
    name: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    source: Literal["rule_based", "llm"]
    usage: dict[str, Any] | None = None


class InterpretContext(BaseModel):
    """What the interpreter is allowed to know. No names, no records: just whether a patient is open."""
    has_active_patient: bool = False
    active_patient_sex: str | None = None


# ----------------------------------------------------------------------------- LLM tool definitions
# The model sees three compact tools. The three report actions share one tool so the patient and
# report fields are sent once, which keeps each request near 1K tokens (free tiers limit tokens
# per minute). `to_internal_call` maps them back onto the internal tool names used everywhere else.
FINDING_CATEGORIES = [
    "impaction", "caries", "periapical", "radiolucency", "radiopacity", "rct", "missing", "bone_loss", "restoration",
    "crown", "implant", "root_stump", "root_resorption", "fracture", "calculus", "pdl_widening", "sinus_thickening",
    "sinus_opacification", "canal_proximity", "retained_deciduous", "supernumerary", "cyst", "unerupted", "other",
]

_RAW_TOOLS: list[dict[str, Any]] = [
    {"name": "report_request",
     "description": "Open, summarize, or answer a question about a patient's imaging report.",
     "input_schema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["open", "summarize", "question"]},
         "patient_name": {"type": "string", "description": "Name as spoken, without 's. Omit for pronouns."},
         "patient_id": {"type": "string", "description": "Only if spoken, e.g. P001."},
         "patient_reference": {"type": "string",
                               "enum": ["named", "pronoun_female", "pronoun_male", "pronoun_neutral", "none"],
                               "description": "named, pronoun_* for her/his/their, or none."},
         "modality": {"type": "string", "enum": ["OPG", "IOPA", "CBCT", "LATERAL_CEPH", "BITEWING", "ANY"],
                      "description": "OPG = panoramic. ANY if not stated."},
         "which": {"type": "string", "enum": ["latest", "previous", "earliest", "by_date"]},
         "date": {"type": "string", "description": "YYYY-MM-DD, YYYY-MM, YYYY, or MM."},
         "question": {"type": "string", "description": "For action=question: the question, verbatim."},
     }, "required": ["action", "patient_reference"]}},
    {"name": "draft_findings",
     "description": ("Dictated radiographic observations. Extract only what was said: no added diagnosis, cause, "
                     "severity or advice. FDI tooth numbers. Quote the exact words in evidence."),
     "input_schema": {"type": "object", "properties": {"findings": {"type": "array", "items": {
         "type": "object",
         "properties": {
             "category": {"type": "string", "enum": FINDING_CATEGORIES},
             "qualifiers": {"type": "array", "items": {"type": "string"}, "description": "Spoken descriptors only."},
             "tooth_fdi": {"type": ["string", "integer"], "description": "FDI number, e.g. 48."},
             "site_label": {"type": "string", "description": "Non-tooth site, e.g. left maxillary sinus."},
             "present": {"type": "boolean", "description": "false if dictated as absent."},
             "certainty": {"type": "string", "enum": ["stated", "uncertain"],
                           "description": "uncertain if hedged (possible, suspected, cannot rule out)."},
             "evidence": {"type": "string", "description": "Exact words from the utterance."},
         },
         "required": ["category", "present", "certainty", "evidence"]}}},
         "required": ["findings"]}},
    {"name": "unsupported_request",
     "description": "Anything else (scheduling, billing, small talk, unclear audio).",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
]

_ACTION_TO_TOOL = {"open": "open_report", "summarize": "summarize_report", "question": "answer_report_question"}


def to_internal_call(name: str, args: dict[str, Any], text: str) -> tuple[str, dict[str, Any]]:
    """Map an LLM-facing tool call onto the internal tool names. Internal names pass through unchanged."""
    if name != "report_request":
        return name, args
    args = dict(args)
    action = args.pop("action", None)
    if action not in _ACTION_TO_TOOL:
        raise ValueError(f"report_request with unknown action {action!r}")
    if action == "question" and not args.get("question"):
        args["question"] = text[:500]
    if action != "question":
        args.pop("question", None)
    return _ACTION_TO_TOOL[action], args


def _nullable(prop: dict[str, Any]) -> dict[str, Any]:
    p = dict(prop)
    t = p.get("type")
    if isinstance(t, str) and t != "null":
        p["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        p["type"] = [*t, "null"]
    if "enum" in p and None not in p["enum"]:
        p["enum"] = [*p["enum"], None]
    return p


def _relax(schema: dict[str, Any]) -> dict[str, Any]:
    """Optional properties accept null; nested object/array schemas get the same treatment."""
    out = dict(schema)
    if "properties" in out:
        required = set(out.get("required", []))
        props = {}
        for name, prop in out["properties"].items():
            prop = _relax(prop)
            props[name] = prop if name in required else _nullable(prop)
        out["properties"] = props
    if isinstance(out.get("items"), dict):
        out["items"] = _relax(out["items"])
    return out


LLM_TOOLS: list[dict[str, Any]] = [{**t, "input_schema": _relax(t["input_schema"])} for t in _RAW_TOOLS]
