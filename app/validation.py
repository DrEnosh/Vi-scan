"""Business-rule validation that sits between the interpreter and the tools.

1. `validate_tool_call` checks the tool name and argument schema (Pydantic) before anything runs.
2. `ground_llm_findings` re-checks every LLM-proposed dictation finding against the transcript
   with the deterministic vocabulary. A finding survives only if its quoted evidence is really
   in the transcript AND that evidence actually mentions the finding. Site, negation and
   qualifiers are re-derived from the words, not taken on trust.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError

from .dental import (
    FINDING_BY_KEY,
    FINDINGS,
    FindingDef,
    assertion_status,
    find_sites,
    is_valid_fdi,
    qualifiers_in,
    site_display,
)
from .interpreter.schema import ARG_MODELS, LLMFinding, ToolCall
from .models import DraftFinding, RejectedFinding, Site
from .tools.dictation import finding_statement, review_reasons, split_pieces
from .tools.retrieval import parse_date_filter


class ToolCallInvalid(ValueError):
    pass


def validate_tool_call(call: ToolCall) -> BaseModel:
    model = ARG_MODELS.get(call.name)
    if model is None:
        raise ToolCallInvalid(f"Unknown tool '{call.name}'")
    try:
        args = model.model_validate(call.args)
    except ValidationError as e:
        raise ToolCallInvalid(f"Invalid arguments for {call.name}: {e.errors()[0].get('msg')}") from e
    date = getattr(args, "date", None)
    if date:
        try:
            parse_date_filter(date)
        except ValueError as e:
            raise ToolCallInvalid(str(e)) from e
    if getattr(args, "patient_name", None):
        cleaned = re.sub(r"[’']s?\b", "", args.patient_name).strip()  # type: ignore[attr-defined]
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{0,79}", cleaned):
            raise ToolCallInvalid("Patient name contains unexpected characters")
        args.patient_name = cleaned  # type: ignore[attr-defined]
    return args


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


class _Rejected(Exception):
    """A proposed finding that fails a check. The message is shown to the clinician."""


def ground_llm_findings(transcript: str, proposed: list[dict[str, Any]]
                        ) -> tuple[list[DraftFinding], list[RejectedFinding]]:
    accepted: list[DraftFinding] = []
    rejected: list[RejectedFinding] = []
    norm_transcript, pieces = _norm(transcript), split_pieces(transcript)
    for raw in proposed:
        try:
            accepted.append(_ground_one(raw, norm_transcript, pieces))
        except _Rejected as e:
            rejected.append(RejectedFinding(proposed=raw, reason=str(e)))
    for i, f in enumerate(accepted, 1):
        f.finding_id = f"D{i}"
    return accepted, rejected


def _ground_one(raw: dict[str, Any], norm_transcript: str, pieces: list[Any]) -> DraftFinding:
    try:
        item = LLMFinding.model_validate(raw)
    except ValidationError:
        raise _Rejected("Malformed finding from the model") from None
    ev = item.evidence.strip()
    nev = _norm(ev)
    if not nev or nev not in norm_transcript:
        raise _Rejected("Quoted evidence does not appear in the transcript")
    # widen to the clause it came from, so a quote that dropped the word "no" is still caught
    context = next((p.text for p in pieces if nev in _norm(p.text)), ev)

    defn, reasons = _vocabulary_entry(item, ev)
    site, site_reasons = _ground_site(item, context)
    present, certainty, cue, assertion_reasons = _ground_assertion(item, defn, context)
    qualifiers, qualifier_reasons = _ground_qualifiers(item, defn, context)

    f = DraftFinding(
        finding_id="", category=item.category,
        finding=defn.label if defn else (item.site_label or "Other finding"),
        qualifiers=qualifiers, present=present, certainty=certainty, certainty_cue=cue,
        site=site, evidence=ev, source="llm",
    )
    f.review_reasons = review_reasons(certainty=certainty, cue=cue, site=site, cross_sentence=False,
                                      interpretive=bool(defn and defn.interpretive), category=item.category)
    f.review_reasons += reasons + site_reasons + assertion_reasons + qualifier_reasons
    f.needs_review = bool(f.review_reasons)
    f.statement = finding_statement(f)
    return f


def _vocabulary_entry(item: LLMFinding, evidence: str) -> tuple[FindingDef | None, list[str]]:
    """The finding must be in the controlled vocabulary, and the quoted words must actually mention it."""
    if item.category == "other":
        return None, ["Category outside the controlled vocabulary"]
    defn = FINDING_BY_KEY.get(item.category)
    if defn is None:
        raise _Rejected(f"Unknown finding category '{item.category}'")
    if not defn.pattern.search(evidence):
        raise _Rejected(f"The quoted evidence does not mention {defn.label.lower()}")
    return defn, []


def _tooth_site(fdi: str) -> Site:
    return Site(type="tooth", fdi=fdi, label=site_display(fdi))


def _ground_site(item: LLMFinding, context: str) -> tuple[Site | None, list[str]]:
    """The site is re-derived from the words; a tooth number from the model must match any tooth that was said."""
    sites = find_sites(context)
    teeth = {s.fdi for s in sites if s.type == "tooth"}
    unresolved = next((s for s in sites if s.type == "unresolved"), None)
    region = next((s for s in sites if s.type in ("region", "generalized")), None)
    if item.tooth_fdi:
        if not is_valid_fdi(item.tooth_fdi):
            raise _Rejected(f"'{item.tooth_fdi}' is not a valid FDI tooth number")
        if teeth and item.tooth_fdi not in teeth:
            raise _Rejected(f"Site mismatch: the dictation names tooth {', '.join(sorted(teeth))}, "
                            f"the model said {item.tooth_fdi}")
        if teeth:
            return _tooth_site(item.tooth_fdi), []
        if unresolved:
            return (Site(type="unresolved", label=unresolved.label, candidates=unresolved.candidates),
                    [f"Model picked tooth {item.tooth_fdi} but the side/arch was not dictated"])
        return _tooth_site(item.tooth_fdi), ["Tooth number not stated in this sentence; confirm"]
    if teeth:
        return _tooth_site(sorted(teeth)[0]), []
    if unresolved:
        return Site(type="unresolved", label=unresolved.label, candidates=unresolved.candidates), []
    if region:
        return Site(type="generalized" if region.type == "generalized" else "region", label=region.label), []
    return None, []


def _ground_assertion(item: LLMFinding, defn: FindingDef | None, context: str
                      ) -> tuple[bool, str, str | None, list[str]]:
    """Negation and hedging come from the dictated words, not from the model's flags."""
    present, certainty, cue, reasons = item.present, item.certainty, None, []
    m = defn.pattern.search(context) if defn else None
    if m:
        det_present, det_cert, cue = assertion_status(context, m.start(), m.end(), context.rstrip().endswith("?"))
        if det_present != item.present:
            reasons.append("Model's present/absent flag disagreed with the dictated wording; wording kept")
            present = det_present
        if det_cert == "uncertain":
            certainty = "uncertain"
    if certainty == "uncertain" and cue is None:
        cue = "model-flagged"
    return present, certainty, cue, reasons


def _ground_qualifiers(item: LLMFinding, defn: FindingDef | None, context: str) -> tuple[list[str], list[str]]:
    """Keep only descriptors that were dictated for this finding, in the vocabulary's wording."""
    if defn is None:
        return [q for q in item.qualifiers if _norm(q) and _norm(q) in _norm(context)], []
    allowed = qualifiers_in(defn, context)
    kept = [q for q in item.qualifiers
            if (q in allowed or (_norm(q) and _norm(q) in _norm(context)))
            and not any(d.key != defn.key and d.pattern.search(q) for d in FINDINGS)]
    dropped = [q for q in item.qualifiers if q not in kept]
    reasons = [f"Dropped qualifier(s) not dictated for this finding: {', '.join(dropped)}"] if dropped else []
    canonical: list[str] = []
    for q in kept:   # "horizontally" -> "horizontal"
        c = next((name for name, rx in defn.qualifiers if rx.search(q)), q)
        if c not in canonical:
            canonical.append(c)
    return canonical or allowed, reasons
