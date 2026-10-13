"""Deterministic interpreter: text -> one ToolCall, no network, no API key.

It produces exactly the same tool calls the LLM interpreter would, so everything downstream
(validation, tools, tests) is shared. It is also the fallback when the LLM is unavailable.
"""
from __future__ import annotations

import re

from ..dental import categories_in_question, find_findings, find_sites
from .schema import InterpretContext, ToolCall

DICTATE_PREFIX_RE = re.compile(r"^\s*(?:dictate|dictation|findings?|note)\s*[:\-]", re.I)
COMMAND_START_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay)\s+\w+[,\s]+)?(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|"
    r"i\s+(?:want|need|would\s+like|wanna)\s+(?:to\s+)?|let(?:'s|\s+me|\s+us)\s+)?(?:take\s+a\s+)?"
    r"(?P<verb>open|show|display|pull|get|fetch|find|bring|load|retrieve|view|see|summari[sz]e|give|read|"
    r"look|what|what's|whats|does|did|do|is|are|was|were|any|has|have|had|tell|list|check|which|how|where|when)\b",
    re.I)
SUMMARY_RE = re.compile(r"\bsummar(?:y|ies|ise|ize|ised|ized|ising|izing)\b|\bbrief\s+me\b|\boverview\b|\bgist\b|\brecap\b", re.I)
QUESTION_VERBS = {"what", "what's", "whats", "does", "did", "do", "is", "are", "was", "were", "any", "has", "have",
                  "had", "tell", "which", "how", "where", "when", "check", "list"}
OPEN_VERBS = {"open", "show", "display", "pull", "get", "fetch", "find", "bring", "load", "retrieve", "view", "see",
              "give", "read", "look"}
FIELD_RE = re.compile(r"\bimpression\b|\brecommend\w*|\badvi[cs]\w*|\bnext\s+steps?\b|\bconclusion\b", re.I)
REPORT_WORD_RE = re.compile(r"\breports?\b|\bscans?\b|\bx-?rays?\b|\bradiographs?\b|\bstud(?:y|ies)\b|\bimages?\b|\bfilms?\b", re.I)

MODALITIES: list[tuple[str, re.Pattern[str]]] = [
    ("CBCT", re.compile(r"\bc[\s.]?b[\s.]?c[\s.]?t\b|\bcone[- ]beam\b|\b3d\s+scan\b", re.I)),
    ("LATERAL_CEPH", re.compile(r"\b(?:lateral\s+|lat\s+)?ceph(?:alogram|alometric|alograph)?s?\b", re.I)),
    ("BITEWING", re.compile(r"\bbite[- ]?wings?\b|\bbw\b", re.I)),
    ("IOPA", re.compile(r"\biopa\b|\bintra-?oral\s+periapical\b|\b(?:periapical|pa)\s+(?:x-?ray|radiograph|film|view)s?\b", re.I)),
    ("OPG", re.compile(r"\bo[\s.]?p[\s.]?gs?\b|\bopt\b|\bpanoramic\b|\bpano\b|\borthopantomogra\w*", re.I)),
]
PREVIOUS_RE = re.compile(r"\b(?:previous|prior|earlier|older|second\s+(?:latest|last|most\s+recent)|one\s+before|"
                         r"before\s+that|last\s+but\s+one)\b", re.I)
EARLIEST_RE = re.compile(r"\b(?:earliest|oldest|first|initial|baseline)\b", re.I)

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
           "nov": 11, "dec": 12}
MONTH_RE = re.compile(
    r"\b(?:(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?)?"
    r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b(?:\s+(?P<day2>\d{1,2})(?:st|nd|rd|th)?\b)?(?:,?\s+(?P<year>(?:19|20)\d{2}))?",
    re.I)
ISO_RE = re.compile(r"\b(?P<year>(?:19|20)\d{2})-(?P<month>\d{1,2})(?:-(?P<day>\d{1,2}))?\b")
YEAR_RE = re.compile(r"\b(?:in|from|of|during)\s+(?P<year>(?:19|20)\d{2})\b", re.I)
PATIENT_ID_RE = re.compile(r"\b(?:patient\s+(?:id\s+)?)?p[-\s]?(?P<num>\d{3})\b", re.I)

PRONOUN_F = re.compile(r"\b(?:her|hers|she)\b", re.I)
PRONOUN_M = re.compile(r"\b(?:his|him|he)\b", re.I)
PRONOUN_N = re.compile(r"\b(?:their|them|they|this\s+patient|the\s+patient|same\s+patient|current\s+patient)\b", re.I)

STOPWORDS = set("""
open show display pull up get fetch find bring load retrieve view see summarize summarise summary summaries summarized
summarised summarizing summarising give read me the a an for of from to on in at with about please can could would you
i want need like let lets us report reports scan scans xray xrays x ray rays radiograph radiographs study studies image
images film films latest last recent newest current most previous prior earlier older second one before that earliest
oldest first initial baseline dated date taken done her hers his him she he their them they this patient patients same is
are was were does do did any has have had what whats tell which how where when say says said mention mentions mentioned
shows showing there it its quick brief overview gist recap again now also and or just okay ok hey dental dentist doctor dr
up thanks thank status file files chart record records result results findings finding note notes id number view
something anything everything all some side tooth teeth s recommended recommendation recommendations recommend
impression advice advised advise plan next step steps conclusion treatment suggested suggest suggestion seen noted
written documented reported mentioned there's anything else more details detail mr mrs ms miss master madam sir
look looking took taken done we us our at need want
""".split())


def detect_modality(text: str) -> str:
    for name, rx in MODALITIES:
        if rx.search(text):
            return name
    return "ANY"


def detect_date(text: str) -> tuple[str | None, list[tuple[int, int]]]:
    """Returns (date string for the tool call, spans to mask)."""
    if m := ISO_RE.search(text):
        y, mo, d = m["year"], int(m["month"]), m["day"]
        return (f"{y}-{mo:02d}-{int(d):02d}" if d else f"{y}-{mo:02d}"), [m.span()]
    for m in MONTH_RE.finditer(text):
        mon = m["mon"].lower()[:3]
        if mon == "may" and not re.search(r"\b(?:in|from|of|on|dated|since)\s+$", text[:m.start()], re.I):
            continue  # "may" is usually the verb
        month = _MONTHS[mon]
        day = m["day"] or m["day2"]
        year = m["year"]
        if year and day:
            return f"{year}-{month:02d}-{int(day):02d}", [m.span()]
        if year:
            return f"{year}-{month:02d}", [m.span()]
        return f"{month:02d}", [m.span()]
    if m := YEAR_RE.search(text):
        return m["year"], [m.span()]
    return None, []


def detect_which(text: str, has_date: bool) -> str:
    if PREVIOUS_RE.search(text):
        return "previous"
    if EARLIEST_RE.search(text):
        return "earliest"
    return "by_date" if has_date else "latest"


def extract_name(text: str, extra_masks: list[tuple[int, int]] | None = None) -> str | None:
    """Strip command words, modalities, dates, findings and sites; what remains is the spoken name."""
    masked = list(text)
    spans = list(extra_masks or [])
    for _, rx in MODALITIES:
        spans += [m.span() for m in rx.finditer(text)]
    spans += [m.span() for m in PATIENT_ID_RE.finditer(text)]
    spans += [(f.start, f.end) for f in find_findings(text)]
    spans += [(s.start, s.end) for s in find_sites(text)]
    for a, b in spans:
        for i in range(a, b):
            masked[i] = " "
    cleaned = "".join(masked).replace("’", "'")
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]*)?", cleaned)
    words = []
    for tok in tokens:
        w = re.sub(r"'s?$", "", tok.lower())
        if w and w not in STOPWORDS and len(w) > 1:
            words.append(w)
    if not words:
        return None
    return " ".join(words[:4])


def _patient_args(text: str, masks: list[tuple[int, int]]) -> dict[str, object]:
    if m := PATIENT_ID_RE.search(text):
        return {"patient_id": f"P{m['num']}", "patient_reference": "named"}
    name = extract_name(text, masks)
    if name:
        return {"patient_name": name, "patient_reference": "named"}
    if PRONOUN_F.search(text):
        return {"patient_reference": "pronoun_female"}
    if PRONOUN_M.search(text):
        return {"patient_reference": "pronoun_male"}
    if PRONOUN_N.search(text):
        return {"patient_reference": "pronoun_neutral"}
    return {"patient_reference": "none"}


def looks_like_dictation(text: str) -> bool:
    if DICTATE_PREFIX_RE.search(text):
        return True
    if COMMAND_START_RE.search(text):
        return False
    return bool(find_findings(text) or find_sites(text))


def looks_like_command(text: str) -> bool:
    return bool(COMMAND_START_RE.search(text) or SUMMARY_RE.search(text))


class RuleBasedInterpreter:
    name = "rule_based"

    def interpret(self, text: str, ctx: InterpretContext, mode: str = "auto") -> ToolCall:
        if mode == "dictation" or (mode == "auto" and looks_like_dictation(text)):
            return ToolCall(name="draft_findings", args={}, source="rule_based")

        date, date_spans = detect_date(text)
        report_args: dict[str, object] = {
            "modality": detect_modality(text),
            "which": detect_which(text, date is not None),
            **({"date": date} if date else {}),
        }
        m = COMMAND_START_RE.search(text)
        verb = m["verb"].lower() if m else None
        has_target = bool(categories_in_question(text) or any(s.type in ("tooth", "unresolved") for s in find_sites(text))
                          or FIELD_RE.search(text))
        is_question = (verb in QUESTION_VERBS) or text.rstrip().endswith("?")

        if SUMMARY_RE.search(text):
            intent = "summarize_report"
        elif is_question and has_target:
            intent = "answer_report_question"
        elif verb in OPEN_VERBS or (is_question and (REPORT_WORD_RE.search(text) or report_args["modality"] != "ANY")):
            intent = "open_report"
        elif is_question:
            intent = "answer_report_question"
        else:
            return ToolCall(name="unsupported_request", args={"reason": "No supported command recognised."},
                            source="rule_based")

        patient = _patient_args(text, date_spans)
        if intent == "answer_report_question":
            return ToolCall(name=intent, args={"question": text[:500], **patient}, source="rule_based")
        return ToolCall(name=intent, args={**report_args, **patient}, source="rule_based")
