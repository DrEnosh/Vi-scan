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
    FINDING_BY_KEY, FINDINGS, assertion_status, find_sites, is_valid_fdi, qualifiers_in, site_display,
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


def ground_llm_findings(transcript: str, proposed: list[dict[str, Any]]) -> tuple[list[DraftFinding], list[RejectedFinding]]:
    accepted: list[DraftFinding] = []
    rejected: list[RejectedFinding] = []
    norm_transcript = _norm(transcript)
    pieces = split_pieces(transcript)

    for raw in proposed:
        try:
            item = LLMFinding.model_validate(raw)
        except ValidationError:
            rejected.append(RejectedFinding(proposed=raw, reason="Malformed finding from the model"))
            continue
        ev = item.evidence.strip()
        nev = _norm(ev)
        if not nev or nev not in norm_transcript:
            rejected.append(RejectedFinding(proposed=raw, reason="Quoted evidence does not appear in the transcript"))
            continue

        # widen to the clause it came from, so a quote that dropped the word "no" is still caught
        context = next((p.text for p in pieces if nev in _norm(p.text)), ev)
        reasons: list[str] = []

        if item.category == "other":
            defn = None
            reasons.append("Category outside the controlled vocabulary")
        else:
            defn = FINDING_BY_KEY.get(item.category)
            if defn is None:
                rejected.append(RejectedFinding(proposed=raw, reason=f"Unknown finding category '{item.category}'"))
                continue
            if not defn.pattern.search(ev):
                rejected.append(RejectedFinding(
                    proposed=raw, reason=f"The quoted evidence does not mention {defn.label.lower()}"))
                continue

        # site: re-derive from the words
        ctx_sites = find_sites(context)
        teeth = {s.fdi for s in ctx_sites if s.type == "tooth"}
        unresolved = [s for s in ctx_sites if s.type == "unresolved"]
        regions = [s for s in ctx_sites if s.type in ("region", "generalized")]
        site: Site | None = None
        if item.tooth_fdi:
            if not is_valid_fdi(item.tooth_fdi):
                rejected.append(RejectedFinding(proposed=raw, reason=f"'{item.tooth_fdi}' is not a valid FDI tooth number"))
                continue
            if teeth and item.tooth_fdi not in teeth:
                rejected.append(RejectedFinding(
                    proposed=raw, reason=f"Site mismatch: the dictation names tooth {', '.join(sorted(teeth))}, "
                                         f"the model said {item.tooth_fdi}"))
                continue
            if teeth:
                site = Site(type="tooth", fdi=item.tooth_fdi, label=site_display(item.tooth_fdi))
            elif unresolved:
                u = unresolved[0]
                site = Site(type="unresolved", label=u.label, candidates=u.candidates)
                reasons.append(f"Model picked tooth {item.tooth_fdi} but the side/arch was not dictated")
            else:
                site = Site(type="tooth", fdi=item.tooth_fdi, label=site_display(item.tooth_fdi))
                reasons.append("Tooth number not stated in this sentence; confirm")
        elif teeth:
            fdi = sorted(teeth)[0]
            site = Site(type="tooth", fdi=fdi, label=site_display(fdi))
        elif unresolved:
            u = unresolved[0]
            site = Site(type="unresolved", label=u.label, candidates=u.candidates)
        elif regions:
            r = regions[0]
            site = Site(type="generalized" if r.type == "generalized" else "region", label=r.label)

        # negation / hedging from the words themselves
        present, certainty, cue = item.present, item.certainty, None
        if defn is not None:
            m = defn.pattern.search(context)
            if m:
                det_present, det_cert, cue = assertion_status(context, m.start(), m.end(), context.rstrip().endswith("?"))
                if det_present != item.present:
                    reasons.append("Model's present/absent flag disagreed with the dictated wording; wording kept")
                    present = det_present
                if det_cert == "uncertain":
                    certainty = "uncertain"
            allowed = qualifiers_in(defn, context)
            kept = [q for q in item.qualifiers
                    if (q in allowed or (_norm(q) and _norm(q) in _norm(context)))
                    and not any(d.key != defn.key and d.pattern.search(q) for d in FINDINGS)]
            dropped = [q for q in item.qualifiers if q not in kept]
            if dropped:
                reasons.append(f"Dropped qualifier(s) not dictated for this finding: {', '.join(dropped)}")
            canonical: list[str] = []
            for q in kept:   # "horizontally" -> "horizontal"
                c = next((name for name, rx in defn.qualifiers if rx.search(q)), q)
                if c not in canonical:
                    canonical.append(c)
            qualifiers = canonical or allowed
        else:
            qualifiers = [q for q in item.qualifiers if _norm(q) and _norm(q) in _norm(context)]
        if certainty == "uncertain" and cue is None:
            cue = "model-flagged"

        f = DraftFinding(
            finding_id="", category=item.category,
            finding=defn.label if defn else (item.site_label or "Other finding"),
            qualifiers=qualifiers, present=present, certainty=certainty, certainty_cue=cue,
            site=site, evidence=ev, source="llm",
        )
        f.review_reasons = review_reasons(certainty=certainty, cue=cue, site=site, cross_sentence=False,
                                          interpretive=bool(defn and defn.interpretive),
                                          category=item.category) + reasons
        f.needs_review = bool(f.review_reasons)
        f.statement = finding_statement(f)
        accepted.append(f)

    for i, f in enumerate(accepted, 1):
        f.finding_id = f"D{i}"
    return accepted, rejected
