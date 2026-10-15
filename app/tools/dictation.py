"""Clinical dictation -> structured findings + readable draft (deterministic engine).

Rules this module never breaks:
  * every finding carries the exact transcript text it came from (`evidence`)
  * negation ("no periapical radiolucency") and hedging ("possible caries") are preserved
  * anything it cannot map to the vocabulary is kept verbatim in `unparsed_segments`
  * it never writes an impression, diagnosis or treatment advice
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..dental import (
    SiteMatch, assertion_status, find_findings, find_sites, qualifiers_in,
)
from ..models import DraftFinding, Site

_ABBREV_RE = re.compile(r"\b(?:approx|w\.r\.t|i\.r\.t|e\.g|i\.e|dr|vs|no|wrt|irt)\.", re.I)
_SENT_RE = re.compile(r"[^.;!?\n]+\??")
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<!\d),\s*(?!\d{2}\b)|(?<!\d)\s+and\s+(?!\d{2}\b)|\s+but\s+|\s*\bhowever\b,?\s*|\s+whereas\s+|\s+also\s+",
    re.I)
_FILLER = {
    "ok", "okay", "um", "uh", "so", "next", "then", "and", "also", "the", "a", "an", "findings", "finding",
    "note", "dictation", "end", "full", "stop", "comma", "period", "new", "line", "paragraph", "is", "are",
    "there", "of", "in", "to", "please", "thanks", "thank", "you", "right", "yes", "fine",
}


@dataclass
class Piece:
    sent_idx: int
    start: int
    end: int
    text: str
    question: bool


@dataclass
class ParseResult:
    findings: list[DraftFinding] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)


def split_pieces(transcript: str) -> list[Piece]:
    """Split into sentences, then into clauses that each carry their own finding or site."""
    masked = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "․"), transcript)  # same length
    pieces: list[Piece] = []
    for si, sm in enumerate(_SENT_RE.finditer(masked)):
        s_start = sm.start()
        sent = masked[sm.start():sm.end()]
        question = sent.rstrip().endswith("?")
        bounds: list[tuple[int, int]] = []
        last = 0
        for b in _CLAUSE_SPLIT_RE.finditer(sent):
            bounds.append((last, b.start()))
            last = b.end()
        bounds.append((last, len(sent)))

        merged: list[tuple[int, int]] = []
        for a, b in bounds:
            seg = transcript[s_start + a:s_start + b]
            if not seg.strip():
                continue
            # a fragment with no finding and no site ("possibly deep") stays with the clause before it
            if merged and not (find_findings(seg) or find_sites(seg)):
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        for a, b in merged:
            raw = transcript[s_start + a:s_start + b]
            lead = len(raw) - len(raw.lstrip())
            st, en = s_start + a + lead, s_start + a + len(raw.rstrip())
            if en > st:
                pieces.append(Piece(si, st, en, transcript[st:en], question))
    return pieces


def _meaningful(text: str) -> bool:
    words = [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w not in _FILLER]
    return len(words) >= 2


def _to_site(s: SiteMatch | None) -> Site | None:
    if s is None:
        return None
    return Site(type=s.type, fdi=s.fdi, label=s.label, candidates=s.candidates)  # type: ignore[arg-type]


def _choose_sites(sites: list[SiteMatch], category: str) -> list[SiteMatch]:
    teeth = [s for s in sites if s.type in ("tooth", "unresolved")]
    regions = [s for s in sites if s.type == "region"]
    general = [s for s in sites if s.type == "generalized"]
    if category.startswith("sinus") and regions:
        return regions
    return teeth or regions or general


def finding_statement(f: DraftFinding) -> str:
    site = f.site.label if f.site else "Site not specified"
    desc = f.finding if f.present else f"No {f.finding[0].lower()}{f.finding[1:]}"
    if f.qualifiers:
        desc += f" ({', '.join(f.qualifiers)})"
    s = f"{site}: {desc}."
    if f.certainty == "uncertain":
        s += f' [uncertain as dictated: "{f.certainty_cue}"]'
    return s


def review_reasons(*, certainty: str, cue: str | None, site: Site | None, cross_sentence: bool,
                   interpretive: bool, category: str) -> list[str]:
    reasons: list[str] = []
    if certainty == "uncertain":
        reasons.append(f'Dictated with uncertainty ("{cue}")')
    if site is None:
        reasons.append("No anatomical site stated")
    elif site.type == "unresolved":
        reasons.append("Tooth not fully specified; clinician must pick the tooth")
    if cross_sentence:
        reasons.append("Site carried over from the previous sentence")
    if interpretive:
        reasons.append("Interpretive term; confirm before finalising")
    if category in ("radiolucency", "radiopacity"):
        reasons.append("Non-specific description; clinician to characterise")
    return reasons


def parse_dictation(transcript: str) -> ParseResult:
    result = ParseResult()
    last_sites: tuple[list[SiteMatch], Piece] | None = None
    pending_site_only: Piece | None = None
    counter = 0

    for piece in split_pieces(transcript):
        sites = find_sites(piece.text)
        fms = find_findings(piece.text)

        if not fms:
            if sites:
                if pending_site_only is not None:
                    result.unparsed.append(pending_site_only.text)
                pending_site_only = piece
                last_sites = (sites, piece)
            elif _meaningful(piece.text):
                result.unparsed.append(piece.text)
            continue

        inherited = False
        site_piece = piece
        if sites:
            use = sites
        elif last_sites is not None:
            use, site_piece = last_sites
            inherited = True
        else:
            use = []
        if pending_site_only is not None:
            if not (inherited and site_piece is pending_site_only):
                result.unparsed.append(pending_site_only.text)
            pending_site_only = None
        cross_sentence = inherited and site_piece.sent_idx != piece.sent_idx

        for fm in fms:
            present, certainty, cue = assertion_status(piece.text, fm.start, fm.end, piece.question)
            quals = qualifiers_in(fm.defn, piece.text)
            chosen = _choose_sites(use, fm.defn.key) or [None]
            for sm in chosen:
                site = _to_site(sm)
                counter += 1
                f = DraftFinding(
                    finding_id=f"D{counter}",
                    category=fm.defn.key,
                    finding=fm.defn.label,
                    qualifiers=quals,
                    present=present,
                    certainty=certainty,  # type: ignore[arg-type]
                    certainty_cue=cue,
                    site=site,
                    evidence=piece.text,
                    site_evidence=site_piece.text if inherited else None,
                    source="rule_parser",
                )
                f.review_reasons = review_reasons(
                    certainty=certainty, cue=cue, site=site, cross_sentence=cross_sentence,
                    interpretive=fm.defn.interpretive, category=fm.defn.key)
                f.needs_review = bool(f.review_reasons)
                f.statement = finding_statement(f)
                result.findings.append(f)
        if sites:
            last_sites = (sites, piece)

    if pending_site_only is not None:
        result.unparsed.append(pending_site_only.text)
    result.findings = dedupe_findings(result.findings)
    return result


def _site_key(f: DraftFinding) -> str:
    if f.site is None:
        return "-"
    return f.site.fdi or f.site.label.lower()


def dedupe_findings(findings: list[DraftFinding]) -> list[DraftFinding]:
    out: list[DraftFinding] = []
    index: dict[tuple[str, str, bool], DraftFinding] = {}
    for f in findings:
        key = (f.category, _site_key(f), f.present)
        if key in index:
            kept = index[key]
            for q in f.qualifiers:
                if q not in kept.qualifiers:
                    kept.qualifiers.append(q)
            if f.certainty == "uncertain" and kept.certainty != "uncertain":  # keep the more cautious reading
                kept.certainty, kept.certainty_cue = "uncertain", f.certainty_cue
            if f.needs_review:
                kept.needs_review = True
                kept.review_reasons += [r for r in f.review_reasons if r not in kept.review_reasons]
            kept.statement = finding_statement(kept)
            continue
        index[key] = f
        out.append(f)
    for i, f in enumerate(out, 1):
        f.finding_id = f"D{i}"
    return out


def build_draft_text(findings: list[DraftFinding], unparsed: list[str], patient: dict[str, Any] | None,
                     status: str = "draft_pending_review") -> str:
    approved = status == "clinician_approved"
    lines = ["REPORT - APPROVED BY CLINICIAN" if approved else "DRAFT REPORT - NOT FINALISED"]
    lines.append("Status: clinician approved" if approved else "Status: pending clinician review")
    if patient:
        lines.append(f"Patient: {patient['name']} ({patient['patient_id']}), {patient['age']} y, {patient['sex']}")
    else:
        lines.append("Patient: not linked (link a patient before sign-off)")
    lines += ["", "Findings (as dictated):"]
    if findings:
        lines += [f"{i}. {f.statement or finding_statement(f)}" for i, f in enumerate(findings, 1)]
    else:
        lines.append("No structured findings could be extracted from this dictation.")
    if unparsed:
        lines += ["", "Dictated text not converted (kept verbatim for review):"]
        lines += [f'- "{u}"' for u in unparsed]
    lines += ["", "Impression: not generated. This draft only restates dictated observations; "
                  "diagnosis and treatment advice are left to the clinician."]
    return "\n".join(lines)


def merge_missing(primary: list[DraftFinding], extra: list[DraftFinding]) -> tuple[list[DraftFinding], int]:
    """Add findings from `extra` (deterministic parser) that `primary` (grounded LLM output) missed.

    Both lists are grounded in the transcript, so the union raises recall without adding anything
    that wasn't said. Returns the merged list (renumbered) and how many findings were added.
    """
    seen = {(f.category, _site_key(f), f.present) for f in primary}
    merged = list(primary)
    added = 0
    for f in extra:
        key = (f.category, _site_key(f), f.present)
        if key not in seen:
            seen.add(key)
            merged.append(f.model_copy(deep=True))
            added += 1
    for i, f in enumerate(merged, 1):
        f.finding_id = f"D{i}"
    return merged, added
