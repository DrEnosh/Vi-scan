"""Deterministic tools: patient lookup, report selection, extractive summary, grounded Q&A.

These are the business rules. The interpreter (LLM or rule-based) only proposes arguments;
the decisions about identity, ambiguity and what counts as evidence are made here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from typing import Any, Literal

from ..dental import CATEGORY_LABELS, categories_in_question, find_sites, is_valid_fdi
from ..models import MODALITY_LABELS, Patient, Report, ReportFinding

# ----------------------------------------------------------------------------- patient matching
FUZZY_THRESHOLD = 0.78


def normalize_name(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"'s\b|'", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(s.split())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class PatientMatch:
    status: Literal["resolved", "ambiguous", "needs_confirmation", "not_found"]
    query: str
    patient: Patient | None = None
    candidates: list[Patient] = field(default_factory=list)
    method: str = ""


def match_patients(query: str, patients: list[Patient]) -> PatientMatch:
    q = normalize_name(query)
    qt = q.split()
    if not qt:
        return PatientMatch("not_found", query)
    exact: list[Patient] = []
    partial: list[Patient] = []
    fuzzy: list[tuple[float, Patient]] = []
    for p in patients:
        n = normalize_name(p.name)
        nt = n.split()
        if qt == nt:
            exact.append(p)
        elif all(t in nt for t in qt):
            partial.append(p)
        else:
            token_score = sum(max(_ratio(a, b) for b in nt) for a in qt) / len(qt)
            score = max(token_score, _ratio(q, n))
            if score >= FUZZY_THRESHOLD:
                fuzzy.append((score, p))
    for group, method in ((exact, "exact"), (partial, "partial")):
        if len(group) == 1:
            return PatientMatch("resolved", query, patient=group[0], candidates=group, method=method)
        if len(group) > 1:
            return PatientMatch("ambiguous", query, candidates=sorted(group, key=lambda p: p.name), method=method)
    if fuzzy:
        fuzzy.sort(key=lambda x: -x[0])
        return PatientMatch("needs_confirmation", query, candidates=[p for _, p in fuzzy[:3]], method="fuzzy")
    return PatientMatch("not_found", query, method="none")


def similar_names(patient: Patient, patients: list[Patient]) -> list[Patient]:
    """Sound-alike / look-alike names worth a safety alert even when the match was exact."""
    n = normalize_name(patient.name).split()
    out = []
    for p in patients:
        if p.patient_id == patient.patient_id:
            continue
        m = normalize_name(p.name).split()
        if (_ratio(" ".join(n), " ".join(m)) >= 0.8 or n[0] == m[0]
                or (n[-1] == m[-1] and _ratio(n[0], m[0]) >= 0.75)):
            out.append(p)
    return out


# ----------------------------------------------------------------------------- report selection
@dataclass
class DateFilter:
    year: int | None = None
    month: int | None = None
    day: int | None = None

    def matches(self, d: date) -> bool:
        return ((self.year is None or d.year == self.year) and (self.month is None or d.month == self.month)
                and (self.day is None or d.day == self.day))

    def describe(self) -> str:
        parts = []
        if self.day:
            parts.append(str(self.day))
        if self.month:
            parts.append(date(2000, self.month, 1).strftime("%B"))
        if self.year:
            parts.append(str(self.year))
        return " ".join(parts)


def parse_date_filter(value: str | None) -> DateFilter | None:
    """Accepts YYYY-MM-DD, YYYY-MM, YYYY or MM (month only, any year)."""
    if not value:
        return None
    v = value.strip()
    if m := re.fullmatch(r"((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})", v):
        df = DateFilter(int(m[1]), int(m[2]), int(m[3]))
    elif m := re.fullmatch(r"((?:19|20)\d{2})-(\d{1,2})", v):
        df = DateFilter(int(m[1]), int(m[2]))
    elif m := re.fullmatch(r"((?:19|20)\d{2})", v):
        df = DateFilter(int(m[1]))
    elif m := re.fullmatch(r"(?:--)?(\d{1,2})", v):
        df = DateFilter(month=int(m[1]))
    else:
        raise ValueError(f"Unrecognised date '{value}'")
    if df.month is not None and not 1 <= df.month <= 12:
        raise ValueError(f"Invalid month in '{value}'")
    if df.day is not None and not 1 <= df.day <= 31:
        raise ValueError(f"Invalid day in '{value}'")
    return df


@dataclass
class ReportSelection:
    report: Report | None
    reason: str = ""
    available: list[Report] = field(default_factory=list)


def select_report(reports: list[Report], modality: str | None, which: str,
                  date_filter: DateFilter | None) -> ReportSelection:
    """`reports` must be newest first."""
    mod = None if modality in (None, "ANY") else modality
    label = MODALITY_LABELS.get(mod, "") if mod else ""
    if not reports:
        return ReportSelection(None, "There are no reports on file for this patient.")
    pool = [r for r in reports if mod is None or r.modality == mod]
    if not pool:
        return ReportSelection(None, f"No {label} report is on file for this patient.", reports)
    if date_filter is not None:
        pool = [r for r in pool if date_filter.matches(r.study_date)]
        if not pool:
            what = f"{label} report" if label else "report"
            return ReportSelection(None, f"No {what} dated {date_filter.describe()} is on file.", reports)
    if which == "previous":
        if len(pool) < 2:
            what = f"{label} report" if label else "report"
            return ReportSelection(None, f"Only one {what} is on file, so there is no previous one.", reports)
        return ReportSelection(pool[1])
    if which == "earliest":
        return ReportSelection(pool[-1])
    return ReportSelection(pool[0])


def report_listing(reports: list[Report]) -> list[dict[str, str]]:
    return [{"report_id": r.report_id, "modality": MODALITY_LABELS[r.modality],
             "study_date": r.study_date.isoformat()} for r in reports]


# ----------------------------------------------------------------------------- summary & Q&A
def site_text(site: str) -> str:
    if is_valid_fdi(site):
        return f"Tooth {site}"
    return site[0].upper() + site[1:]


def finding_line(f: ReportFinding) -> str:
    return f"{site_text(f.site)}: {f.text}"


def summarize_report(report: Report, patient: Patient) -> dict[str, Any]:
    """Extractive summary: every line is copied from a field of the signed report."""
    positives = [f for f in report.findings if f.present]
    negatives = [f for f in report.findings if not f.present]
    points = [{"text": finding_line(f), "source": f"{report.report_id}#{f.id}"} for f in positives]
    points += [{"text": finding_line(f), "source": f"{report.report_id}#{f.id}"} for f in negatives]
    lines = [f"{patient.name} ({patient.patient_id}) - {report.headline()}", "Key findings:"]
    lines += [f"- {finding_line(f)} [{f.id}]" for f in positives]
    if negatives:
        lines.append("Documented as not seen:")
        lines += [f"- {finding_line(f)} [{f.id}]" for f in negatives]
    lines.append(f"Impression (as reported): {report.impression}")
    lines.append(f"Recommendation (as reported): {report.recommendation}")
    return {
        "message": "\n".join(lines),
        "points": points,
        "impression": {"text": report.impression, "source": f"{report.report_id}#impression"},
        "recommendation": {"text": report.recommendation, "source": f"{report.report_id}#recommendation"},
        "method": "extractive (no generated wording; each line maps to a report field)",
        "finding_ids": [f.id for f in report.findings],
    }


_IMPRESSION_RE = re.compile(r"\bimpression\b|\bconclusion\b|\bdiagnos\w*|\boverall\b", re.I)
_RECOMMEND_RE = re.compile(r"\brecommend\w*|\badvi[cs]\w*|\bnext\s+steps?\b|\bplan\b|\btreatment\b|\bsuggest\w*|"
                           r"\bshould\b|\bneeds?\b|\bextract(?:ion|ions)?\b|\bsurg(?:ery|ical)\b|\bremov(?:e|al)\b|"
                           r"\bfollow[- ]?ups?\b", re.I)


def answer_question(question: str, report: Report, other_reports: list[Report]) -> dict[str, Any]:
    cats = categories_in_question(question)
    teeth: set[str] = set()
    for s in find_sites(question):
        if s.type == "tooth" and s.fdi:
            teeth.add(s.fdi)
        elif s.type == "unresolved":
            teeth.update(s.candidates)
    asks_impression = bool(_IMPRESSION_RE.search(question))
    asks_recommendation = bool(_RECOMMEND_RE.search(question))
    head = f"{report.report_id} ({MODALITY_LABELS[report.modality]}, {report.study_date.isoformat()})"

    if not (cats or teeth or asks_impression or asks_recommendation):
        return {"type": "unsupported", "finding_ids": [], "fields": [],
                "message": (f"I can only answer from what {head} documents, so I can't confirm or rule that out. "
                            "Ask about a tooth (\"what about 48?\"), a finding (\"any caries?\"), "
                            "the impression or the recommendation.")}

    lines: list[str] = []
    fields: list[str] = []
    if asks_impression:
        lines.append(f"Impression in {head}: {report.impression}")
        fields.append("impression")
    if asks_recommendation:
        lines.append(f"Recommendation in {head}: {report.recommendation}")
        fields.append("recommendation")

    matched: list[ReportFinding] = []
    if cats or teeth:
        matched = [f for f in report.findings
                   if (not cats or f.category in cats) and (not teeth or f.site in teeth)]
        target = _describe_target(cats, teeth)
        if matched:
            lines.append(f"{head} documents, for {target}:")
            lines += [f"- {finding_line(f)} [{f.id}]" for f in matched]
        else:
            lines.append(f"{head} does not mention {target}. I can't confirm or rule it out from this report.")
            others = [r.report_id for r in other_reports if r.report_id != report.report_id and any(
                (not cats or f.category in cats) and (not teeth or f.site in teeth) for f in r.findings)]
            if others:
                lines.append(f"Other reports on file that mention it: {', '.join(others)}. "
                             "Ask me to open one to review it.")
    return {"type": "answer", "message": "\n".join(lines), "finding_ids": [f.id for f in matched], "fields": fields}


def _describe_target(cats: set[str], teeth: set[str]) -> str:
    parts = []
    if cats:
        parts.append(" / ".join(sorted(CATEGORY_LABELS.get(c, c).lower() for c in cats)))
    if teeth:
        parts.append("tooth " + " / ".join(sorted(teeth)))
    return " on ".join(parts) if len(parts) == 2 else parts[0]
