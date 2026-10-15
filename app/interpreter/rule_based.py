"""Deterministic interpreter: text -> one ToolCall, no network, no API key.

It produces exactly the same tool calls the LLM interpreter would, so everything downstream
(validation, tools, tests) is shared. It is also the fallback when the LLM is unavailable.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..dental import categories_in_question, find_findings, find_sites, question_term_spans
from .schema import InterpretContext, ToolCall

DICTATE_PREFIX_RE = re.compile(r"^\s*(?:dictate|dictation|findings?|note)\s*[:\-]", re.I)
COMMAND_START_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay)\s+\w+[,\s]+)?(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|"
    r"i\s+(?:want|need|would\s+like|wanna)\s+(?:to\s+)?|let(?:'s|\s+me|\s+us)\s+)?(?:take\s+a\s+)?"
    r"(?P<verb>open|show|display|pull|get|fetch|find|bring|load|retrieve|view|see|summari[sz]e|give|read|"
    r"look|what|what's|whats|does|did|do|is|are|was|were|any|has|have|had|tell|list|check|which|how|where|when|"
    r"should|shall|must)\b",
    re.I)
SUMMARY_RE = re.compile(
    r"\bsummar(?:y|ies|ise|ize|ised|ized|ising|izing)\b|\bbrief\s+me\b|\boverview\b|\bgist\b|\brecap\b", re.I)
QUESTION_VERBS = {"what", "what's", "whats", "does", "did", "do", "is", "are", "was", "were", "any", "has", "have",
                  "had", "tell", "which", "how", "where", "when", "check", "list", "should", "shall", "must"}
OPEN_VERBS = {"open", "show", "display", "pull", "get", "fetch", "find", "bring", "load", "retrieve", "view", "see",
              "give", "read", "look"}
BROAD_RE = re.compile(r"\banything\s+(?:concerning|abnormal|wrong|important|significant|notable|serious|unusual|"
                      r"worrying)\b|\bany\s+(?:concerns?|issues?|problems?|abnormalit(?:y|ies)|red\s+flags?)\b|"
                      r"\b(?:key|main)\s+findings\b|\bwhat\s+are\s+the\s+findings\b|"
                      r"\bexplain\s+(?:the\s+|her\s+|his\s+|their\s+)?(?:findings|report|scan|results?)\b|"
                      r"\bwhat\s+(?:was|were)\s+(?:found|seen|noted)\b", re.I)
FIELD_RE = re.compile(r"\bimpression\b|\brecommend\w*|\badvi[cs]\w*|\bnext\s+steps?\b|\bconclusion\b", re.I)
REPORT_WORD_RE = re.compile(
    r"\breports?\b|\bscans?\b|\bx-?rays?\b|\bradiographs?\b|\bstud(?:y|ies)\b|\bimages?\b|\bfilms?\b", re.I)

MODALITIES: list[tuple[str, re.Pattern[str]]] = [
    ("CBCT", re.compile(r"\bc[\s.]?b[\s.]?c[\s.]?t\b|\bcone[- ]beam\b|\b3d\s+scan\b", re.I)),
    ("LATERAL_CEPH", re.compile(r"\b(?:lateral\s+|lat\s+)?ceph(?:alogram|alometric|alograph)?s?\b", re.I)),
    ("BITEWING", re.compile(r"\bbite[- ]?wings?\b|\bbw\b", re.I)),
    ("IOPA", re.compile(
        r"\biopa\b|\bintra-?oral\s+periapical\b|\b(?:periapical|pa)\s+(?:x-?ray|radiograph|film|view)s?\b", re.I)),
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

# ----------------------------------------------------------------------------- patient names
# A word counts as part of a patient name only with evidence for it: it matches a name in the
# clinic's patient list, or it sits where a name sits ("Priya's", "patient Rahul", "does Kavya have",
# "open Arjun"). Any other unknown word is read as part of the request, so a clinical word the rules
# have never seen ("Is malignancy ruled out?") needs no entry anywhere. The only fixed lists are
# grammar: English function words and the assistant's own command vocabulary.
FUNCTION_WORDS = frozenset("""
a an the this that these those it its it's there there's here what's whats that's let let's lets
who who's whom whose why
i me my mine we us our you your he him his she her hers they them their theirs
is isn't are aren't was wasn't were be been being am do don't does doesn't did didn't done has hasn't have
haven't had having can can't could will would shall should may might must
and or but nor not no yes so if then than as at by for from in into of on onto off over under to up down
out with without about after before since between through again also too just only even still yet ever
never always already really actually maybe perhaps very quite much many more most less least some any all
each every both either neither other another such same own something anything everything nothing else
one ones like please thanks thank ok okay hey hi hello sure right alright well now
""".split())
COMMAND_WORDS = frozenset("""
open show display pull get fetch find bring load retrieve view see look read give tell list check say says said
summary summaries summarize summarise summarized summarised overview gist recap brief explain
report reports scan scans xray xrays x ray rays radiograph radiographs study studies image images film films
file files record records chart result results finding findings note notes details detail
impression conclusion recommendation recommendations recommend recommended advice advise advised plan
latest last recent newest current previous prior earlier older first earliest oldest initial baseline second
today yesterday tomorrow week weeks month months year years day days ago time
patient patients clinic doctor dr radiologist mention mentions mentioned noted seen documented
""".split())
NOT_NAMES = FUNCTION_WORDS | COMMAND_WORDS | OPEN_VERBS | QUESTION_VERBS
NAME_CUES = frozenset({"patient", "named", "called", "mr", "mrs", "ms", "miss", "master"})
AUX_VERBS = frozenset({"does", "did", "do", "has", "have", "is", "was"})
HAVE_VERBS = frozenset({"have", "has", "had", "got", "get", "need", "needs"})
VERB_PARTICLES = frozenset({"me", "up", "us"})  # "show me Arjun", "pull up Rahul"
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?['’]?")


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


@dataclass
class _Token:
    raw: str
    word: str          # lower case, possessive removed
    start: int
    possessive: bool
    masked: bool


def _close(a: str, b: str) -> bool:
    """A near spelling of a name word ("Pria" / "Priya"): long enough, same first letter, similar letters."""
    return (len(a) >= 4 and len(b) >= 4 and a[0] == b[0] and abs(len(a) - len(b)) <= 1
            and SequenceMatcher(None, a, b).ratio() >= 0.8)


def _tokens(text: str, masks: list[tuple[int, int]], known: set[str]) -> list[_Token]:
    out = []
    for m in TOKEN_RE.finditer(text):
        raw = m.group(0).replace("’", "'")
        low = raw.lower()
        word = re.sub(r"'s?$", "", low)
        if low in NOT_NAMES:
            word = low  # "it's", "what's": grammar, not a possessive
        elif word.endswith("s") and word not in known and word[:-1] in known:
            word = word[:-1]  # "sharmas" typed or transcribed without the apostrophe
        out.append(_Token(raw, word, m.start(), word != low.rstrip("'"),
                          any(a < m.end() and m.start() < b for a, b in masks)))
    return out


def _capitalised(tok: _Token, text: str) -> bool:
    """Capitalised mid-sentence ("the report for Meera"); only meaningful when the text is in normal case."""
    if not tok.raw[0].isupper() or tok.raw.isupper() or text.islower() or text.isupper():
        return False
    before = text[:tok.start].rstrip()
    return bool(before) and before[-1] not in ".!?:;"


def find_name(text: str, names: Sequence[str] = (), masks: list[tuple[int, int]] | None = None,
              bare: bool = False) -> str | None:
    """The patient name spoken in `text`, or None.

    `names` are the clinic's patient names (local only; never sent to a model). With `bare=True`
    the whole utterance is expected to be a name (the answer to "which patient?")."""
    known = {w for n in names for w in re.findall(r"[a-z]+", n.lower())}
    spans = list(masks or [])
    for _, rx in MODALITIES:
        spans += [m.span() for m in rx.finditer(text)]
    spans += [m.span() for m in PATIENT_ID_RE.finditer(text)]
    spans += [(f.start, f.end) for f in find_findings(text)]
    spans += [(s.start, s.end) for s in find_sites(text)]
    spans += question_term_spans(text)
    command = COMMAND_START_RE.search(text)
    if command:
        spans.append(command.span())
    toks = _tokens(text, spans, known)

    def usable(t: _Token) -> bool:
        return not t.masked and len(t.word) > 1 and t.word not in NOT_NAMES

    def in_index(t: _Token) -> bool:
        return t.word in known or any(_close(t.word, k) for k in known)

    def own_evidence(i: int) -> bool:
        t, prev = toks[i], toks[i - 1].word if i else ""
        object_of_verb = (command is not None and command["verb"].lower() in OPEN_VERBS | {"summarize", "summarise"}
                          and all(x.word in VERB_PARTICLES for x in toks[:i] if x.start >= command.end()))
        return in_index(t) or t.possessive or prev in NAME_CUES or object_of_verb or _capitalised(t, text)

    runs: list[list[int]] = []
    for i, t in enumerate(toks):
        if not usable(t):
            continue
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])

    found: list[tuple[bool, list[str]]] = []  # (matches the patient list, name words) per run
    for run in runs:
        before = toks[run[0] - 1].word if run[0] else ""
        if before in ("dr", "doctor"):
            continue  # "what did Dr. Rao say": a clinician, not the patient
        words = [toks[i].word for i in run if own_evidence(i)]
        after = toks[run[-1] + 1].word if run[-1] + 1 < len(toks) else ""
        if not words and (bare or (before in AUX_VERBS and after in HAVE_VERBS)):  # "does Kavya Reddy have ..."
            words = [toks[i].word for i in run]
        if words:
            found.append((any(in_index(toks[i]) for i in run), words))
    if not found:
        return None
    words = next((w for indexed, w in found if indexed), found[0][1])  # a known name beats an unknown word
    return " ".join(words[:4])


def _patient_args(text: str, masks: list[tuple[int, int]], names: Sequence[str]) -> dict[str, object]:
    if m := PATIENT_ID_RE.search(text):
        return {"patient_id": f"P{m['num']}", "patient_reference": "named"}
    name = find_name(text, names, masks)
    if name:
        return {"patient_name": name, "patient_reference": "named"}
    if PRONOUN_F.search(text):
        return {"patient_reference": "pronoun_female"}
    if PRONOUN_M.search(text):
        return {"patient_reference": "pronoun_male"}
    if PRONOUN_N.search(text):
        return {"patient_reference": "pronoun_neutral"}
    return {"patient_reference": "none"}


def first_pronoun(text: str) -> tuple[str, int] | None:
    """The earliest patient pronoun in the text, as (patient_reference, character position)."""
    hits = [(m.start(), ref) for ref, rx in (("pronoun_female", PRONOUN_F), ("pronoun_male", PRONOUN_M),
                                             ("pronoun_neutral", PRONOUN_N)) if (m := rx.search(text))]
    if not hits:
        return None
    pos, ref = min(hits)
    return ref, pos


def looks_like_dictation(text: str) -> bool:
    if DICTATE_PREFIX_RE.search(text):
        return True
    if COMMAND_START_RE.search(text) or text.rstrip().endswith("?"):  # a question is never dictation
        return False
    return bool(find_findings(text) or find_sites(text))


def looks_like_command(text: str) -> bool:
    return bool(COMMAND_START_RE.search(text) or SUMMARY_RE.search(text))


class RuleBasedInterpreter:
    name = "rule_based"

    def interpret(self, text: str, ctx: InterpretContext, mode: str = "auto", names: Sequence[str] = ()) -> ToolCall:
        """`names`: the clinician's patient names, used locally to recognise a spoken name."""
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
        has_target = bool(categories_in_question(text) or FIELD_RE.search(text)
                          or any(s.type in ("tooth", "unresolved") for s in find_sites(text)))
        is_question = (verb in QUESTION_VERBS) or text.rstrip().endswith("?")

        if SUMMARY_RE.search(text) or (BROAD_RE.search(text) and not has_target):
            intent = "summarize_report"
        elif (is_question and has_target) or (FIELD_RE.search(text) and report_args["modality"] == "ANY"
                                               and verb in ("read", "give", "show", "tell", "display", "get")):
            intent = "answer_report_question"  # a question about the report, or "read me the impression"
        elif verb in OPEN_VERBS or (is_question and (REPORT_WORD_RE.search(text) or report_args["modality"] != "ANY")):
            intent = "open_report"
        elif is_question:
            intent = "answer_report_question"
        else:
            return ToolCall(name="unsupported_request", args={"reason": "No supported command recognised."},
                            source="rule_based")

        patient = _patient_args(text, date_spans, names)
        if intent == "answer_report_question":
            return ToolCall(name=intent, args={"question": text[:500], **patient}, source="rule_based")
        return ToolCall(name=intent, args={**report_args, **patient}, source="rule_based")
