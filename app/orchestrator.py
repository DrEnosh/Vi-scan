"""Conversation orchestrator.

    transcript -> interpreter (LLM or rules) -> ToolCall
               -> validate_tool_call (schema + business rules)
               -> deterministic tool (patient match, report selection, dictation parser)
               -> response with evidence + trace

Context (the open patient/report, a pending "which patient?" question) lives in the session,
so follow-ups like "summarize her report" need no repeated name when the reference is clear.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .interpreter import InterpretContext, RuleBasedInterpreter, ToolCall
from .interpreter.rule_based import extract_name, looks_like_command, looks_like_dictation
from .interpreter.schema import DraftArgs, PatientRefArgs, QuestionArgs, ReportQueryArgs, UnsupportedArgs
from .models import MODALITY_LABELS, AssistantResponse, Clinician, DraftReport, Evidence, Patient, Report, TraceStep
from .repository import Repository
from .session import DraftStore, PendingClarification, Session, SessionStore
from .tools.dictation import build_draft_text, merge_missing, parse_dictation
from .tools.retrieval import (
    answer_question, match_patients, parse_date_filter, report_listing, select_report, similar_names,
    summarize_report,
)
from .validation import ToolCallInvalid, ground_llm_findings, validate_tool_call

audit_log = logging.getLogger("viscan.audit")
log = logging.getLogger("viscan")

EXAMPLES = ['"Open Priya Sharma\'s latest OPG report"', '"Summarize her report"', '"Does she have any caries?"',
            '"Lower right third molar is horizontally impacted" (dictation)']
REPORT_TOOLS = {"open_report", "summarize_report", "answer_report_question"}
CANCEL_RE = re.compile(r"^\s*(?:no|nope|cancel|neither|none(?:\s+of\s+them)?|stop|never\s*mind|forget\s+it)\W*$", re.I)
YES_RE = re.compile(r"^\s*(?:yes|yeah|yep|yup|correct|right|that'?s\s+(?:right|correct|the\s+one)|that\s+one|confirm(?:ed)?|"
                    r"ok(?:ay)?|sure)\b", re.I)
ORDINALS = [("first", 0), ("1st", 0), ("second", 1), ("2nd", 1), ("third", 2), ("3rd", 2), ("fourth", 3), ("4th", 3),
            ("last", -1)]


@dataclass
class Turn:
    session: Session
    clinician: Clinician
    text: str
    trace: list[TraceStep] = field(default_factory=list)

    def step(self, name: str, t0: float | None = None, **detail: Any) -> None:
        ms = round((time.perf_counter() - t0) * 1000, 2) if t0 is not None else None
        self.trace.append(TraceStep(step=name, detail=detail, ms=ms))


class Assistant:
    def __init__(self, repo: Repository, sessions: SessionStore, drafts: DraftStore, interpreter: Any = None):
        self.repo = repo
        self.sessions = sessions
        self.drafts = drafts
        self.rules = RuleBasedInterpreter()
        self.interpreter = interpreter or self.rules

    # ================================================================== entry point
    def handle(self, text: str, clinician: Clinician, session_id: str | None = None, mode: str = "auto",
               channel: str = "text") -> AssistantResponse:
        t0 = time.perf_counter()
        session = self.sessions.get_or_create(session_id, clinician.clinician_id)
        turn = Turn(session=session, clinician=clinician, text=text)
        turn.step("input", channel=channel, mode=mode, chars=len(text))

        resp = self._resolve_pending(turn) if session.pending else None
        if resp is None:
            call = self._interpret(turn, mode)
            resp = self._execute(call, turn)

        session.turns += 1
        resp.trace = turn.trace
        resp.total_ms = round((time.perf_counter() - t0) * 1000, 2)
        return resp

    # ================================================================== interpretation
    def _interpret(self, turn: Turn, mode: str) -> ToolCall:
        active = self._active_patient(turn)
        ictx = InterpretContext(has_active_patient=active is not None, active_patient_sex=active.sex if active else None)
        t = time.perf_counter()
        if self.interpreter is not self.rules:
            try:
                call = self.interpreter.interpret(turn.text, ictx, mode)
                turn.step("interpret", t, engine=self.interpreter.name, tool=call.name, args=call.args, usage=call.usage)
                return self._cross_check_name(call, turn, ictx)
            except Exception as e:  # network error, timeout, bad output: degrade, don't fail the clinician
                turn.step("interpret_fallback", t, engine=self.interpreter.name, error=type(e).__name__,
                          note="LLM unavailable or returned no usable tool call; using rule-based interpreter")
                t = time.perf_counter()
        call = self.rules.interpret(turn.text, ictx, mode)
        turn.step("interpret", t, engine="rule_based", tool=call.name, args=call.args)
        return call

    def _cross_check_name(self, call: ToolCall, turn: Turn, ictx: InterpretContext) -> ToolCall:
        """If a patient name was spoken but the LLM returned none (e.g. read "Priya's" as a pronoun),
        use the spoken name. Falling back to session context there could open the wrong patient."""
        if call.name not in REPORT_TOOLS or call.args.get("patient_name") or call.args.get("patient_id"):
            return call
        spoken = self.rules.interpret(turn.text, ictx, "command").args
        name, pid = spoken.get("patient_name"), spoken.get("patient_id")
        if not (name or pid):
            return call
        if name and match_patients(str(name), self.repo.patients_for(turn.clinician)).status == "not_found":
            return call
        turn.step("name_cross_check", llm_reference=call.args.get("patient_reference"), spoken=name or pid,
                  note="LLM returned no patient name although one was spoken; using the spoken name")
        args = {**call.args, "patient_name": name, "patient_id": pid, "patient_reference": "named"}
        return ToolCall(name=call.name, args=args, source=call.source, usage=call.usage)

    # ================================================================== validation + dispatch
    def _execute(self, call: ToolCall, turn: Turn) -> AssistantResponse:
        t = time.perf_counter()
        try:
            args = validate_tool_call(call)
        except ToolCallInvalid as e:
            turn.step("validate", t, ok=False, error=str(e))
            if call.source == "llm":  # re-interpret deterministically rather than act on a bad call
                fallback = self.rules.interpret(turn.text, InterpretContext(), "auto")
                turn.step("interpret", engine="rule_based", tool=fallback.name, args=fallback.args, reason="llm_call_invalid")
                return self._execute(fallback, turn)
            return self._unsupported(turn, f"I couldn't act on that request ({e}).")
        turn.step("validate", t, ok=True, tool=call.name)

        if call.name == "open_report":
            return self._open_report(call, args, turn)  # type: ignore[arg-type]
        if call.name == "summarize_report":
            return self._summarize(call, args, turn)  # type: ignore[arg-type]
        if call.name == "answer_report_question":
            return self._answer(call, args, turn)  # type: ignore[arg-type]
        if call.name == "draft_findings":
            return self._dictation(call, args, turn)  # type: ignore[arg-type]
        reason = args.reason if isinstance(args, UnsupportedArgs) else ""
        return self._unsupported(turn, "I can open reports, summarize them, answer questions about them, "
                                       "or turn dictation into a draft report." + (f" ({reason})" if reason else ""))

    # ================================================================== patient resolution
    def _active_patient(self, turn: Turn) -> Patient | None:
        pid = turn.session.active_patient_id
        return self.repo.get_patient(pid, turn.clinician) if pid else None

    def _resolve_patient(self, call: ToolCall, args: PatientRefArgs, turn: Turn) -> tuple[Patient, list[str]] | AssistantResponse:
        t = time.perf_counter()
        accessible = self.repo.patients_for(turn.clinician)

        if args.patient_id:
            p = self.repo.get_patient(args.patient_id, turn.clinician)
            turn.step("tool:find_patient", t, by="id", query=args.patient_id, found=bool(p))
            if p is None:
                return self._respond(turn, "not_found", f"No patient with ID {args.patient_id.upper()} is in your clinic's records.")
            return p, self._similar_warnings(p, accessible)

        if args.patient_name:
            m = match_patients(args.patient_name, accessible)
            turn.step("tool:find_patient", t, by="name", query=args.patient_name, status=m.status, method=m.method,
                      candidates=[c.patient_id for c in m.candidates])
            if m.status == "resolved" and m.patient:
                return m.patient, self._similar_warnings(m.patient, accessible)
            if m.status == "ambiguous":
                q = (f"I found {len(m.candidates)} patients matching \"{args.patient_name.title()}\". "
                     f"Which one do you mean? You can say the full name or the patient ID.")
                return self._ask(turn, call, "choose_patient", m.candidates, q)
            if m.status == "needs_confirmation":
                names = " or ".join(f"{c.name} ({c.patient_id})" for c in m.candidates)
                q = f"I couldn't find an exact match for \"{args.patient_name.title()}\". Did you mean {names}?"
                return self._ask(turn, call, "confirm_patient", m.candidates, q)
            return self._respond(turn, "not_found",
                                 f"No patient named \"{args.patient_name.title()}\" is in your clinic's records. "
                                 f"Check the name or use the patient ID.")

        active = self._active_patient(turn)
        ref = args.patient_reference
        if ref.startswith("pronoun"):
            if active is None:
                return self._ask(turn, call, "name_patient", [], "Which patient do you mean? No patient is open in this conversation yet.")
            expected = {"pronoun_female": "F", "pronoun_male": "M"}.get(ref)
            if expected and active.sex != expected:
                word = "her" if expected == "F" else "his"
                q = (f"You said \"{word}\", but the patient open now is {active.name} ({active.patient_id}, "
                     f"{'female' if active.sex == 'F' else 'male'}). Which patient do you mean?")
                turn.step("context_check", t, result="pronoun_mismatch", active_patient=active.patient_id)
                return self._ask(turn, call, "name_patient", [], q)
            turn.step("context_check", t, result="resolved_from_session", patient_id=active.patient_id, reference=ref)
            return active, []
        if active is not None:
            turn.step("context_check", t, result="no_name_given_using_open_patient", patient_id=active.patient_id)
            return active, [f"Using the patient already open: {active.name} ({active.patient_id})."]
        return self._ask(turn, call, "name_patient", [], "Which patient's report do you want?")

    def _similar_warnings(self, p: Patient, accessible: list[Patient]) -> list[str]:
        sims = similar_names(p, accessible)
        if not sims:
            return []
        others = ", ".join(f"{s.name} ({s.patient_id})" for s in sims)
        return [f"Similar name on file: {others}. Confirm patient ID {p.patient_id} and age {p.age_on()} before acting."]

    def _ask(self, turn: Turn, call: ToolCall, kind: str, candidates: list[Patient], question: str) -> AssistantResponse:
        turn.session.pending = PendingClarification(call=call, kind=kind, candidate_ids=[c.patient_id for c in candidates],  # type: ignore[arg-type]
                                                    question=question)
        cards = [self._candidate_card(c) for c in candidates]
        return self._respond(turn, "clarification", question, data={"kind": kind, "candidates": cards})

    def _candidate_card(self, p: Patient) -> dict[str, Any]:
        reports = self.repo.reports_for(p)
        latest = reports[0] if reports else None
        return {**p.card(), "latest_report": f"{MODALITY_LABELS[latest.modality]} {latest.study_date.isoformat()}" if latest else None,
                "report_count": len(reports)}

    # ================================================================== clarification follow-up
    def _resolve_pending(self, turn: Turn) -> AssistantResponse | None:
        pend = turn.session.pending
        assert pend is not None
        text = turn.text.strip()
        t = time.perf_counter()
        if CANCEL_RE.match(text):
            turn.session.pending = None
            turn.step("clarification", t, result="cancelled")
            return self._respond(turn, "cancelled", "Okay, cancelled. Nothing was opened.")
        if looks_like_command(text) or looks_like_dictation(text):
            # a fresh request, not an answer: drop the question instead of applying the old intent to it
            turn.session.pending = None
            turn.step("clarification", t, result="abandoned_for_new_request")
            return None

        accessible = self.repo.patients_for(turn.clinician)
        cands = [p for p in accessible if p.patient_id in pend.candidate_ids]
        chosen: Patient | None = None
        how = ""

        if m := re.search(r"\bp[-\s]?(\d{3})\b", text, re.I):
            chosen = self.repo.get_patient(f"P{m[1]}", turn.clinician)
            how = "patient_id"
        if chosen is None and cands:
            low = text.lower()
            for word, idx in ORDINALS:
                if re.search(rf"\b{word}\b", low) and len(cands) >= (idx + 1 if idx >= 0 else 1):
                    chosen, how = cands[idx], "ordinal"
                    break
            if chosen is None and re.fullmatch(r"\s*(?:option\s+|number\s+)?([1-4])\s*", low):
                i = int(re.fullmatch(r"\s*(?:option\s+|number\s+)?([1-4])\s*", low)[1]) - 1  # type: ignore[index]
                if i < len(cands):
                    chosen, how = cands[i], "ordinal"
            if chosen is None and len(cands) == 1 and YES_RE.match(text):
                chosen, how = cands[0], "confirmed"
        if chosen is None:
            name = extract_name(text)
            if name:
                pool = cands or accessible
                m2 = match_patients(name, pool)
                if m2.status != "resolved" and cands:
                    m2 = match_patients(name, accessible)  # they may have named someone else entirely
                if m2.status == "resolved" and m2.patient:
                    chosen, how = m2.patient, "name"
                elif m2.status in ("ambiguous", "needs_confirmation"):
                    turn.step("clarification", t, result="still_ambiguous", candidates=[c.patient_id for c in m2.candidates])
                    return self._ask(turn, pend.call, "choose_patient", m2.candidates,
                                     f"Still more than one match for \"{name.title()}\". Please give the full name or patient ID.")

        if chosen is not None:
            turn.session.pending = None
            turn.step("clarification", t, result="resolved", patient_id=chosen.patient_id, how=how)
            args = dict(pend.call.args)
            args.update({"patient_id": chosen.patient_id, "patient_name": None, "patient_reference": "named"})
            return self._execute(ToolCall(name=pend.call.name, args=args, source=pend.call.source), turn)

        turn.step("clarification", t, result="unresolved")
        cards = [self._candidate_card(c) for c in cands]
        return self._respond(turn, "clarification", f"Sorry, I still need to know which patient. {pend.question}",
                             data={"kind": pend.kind, "candidates": cards})

    # ================================================================== tools
    def _open_report(self, call: ToolCall, args: ReportQueryArgs, turn: Turn) -> AssistantResponse:
        res = self._resolve_patient(call, args, turn)
        if isinstance(res, AssistantResponse):
            return res
        patient, notes = res
        t = time.perf_counter()
        reports = self.repo.reports_for(patient)
        sel = select_report(reports, args.modality, args.which, parse_date_filter(args.date))
        turn.step("tool:get_report", t, patient_id=patient.patient_id, modality=args.modality, which=args.which,
                  date=args.date, result=sel.report.report_id if sel.report else None)
        turn.session.active_patient_id = patient.patient_id  # the patient is unambiguous even if the report is missing
        if sel.report is None:
            turn.session.active_report_id = None
            listing = report_listing(sel.available)
            avail = "; ".join(f"{r['modality']} {r['study_date']} ({r['report_id']})" for r in listing)
            msg = f"{patient.name} ({patient.patient_id}): {sel.reason}" + (f" Available: {avail}." if avail else "")
            return self._respond(turn, "not_found", msg, data={"patient": patient.card(), "available_reports": listing},
                                 warnings=notes)
        r = sel.report
        turn.session.active_report_id = r.report_id
        self._audit(turn, "open_report", patient_id=patient.patient_id, report_id=r.report_id)
        return self._respond(
            turn, "report", f"Opened {patient.name} ({patient.patient_id}) - {r.headline()}.",
            data={"patient": patient.card(), "report": r.model_dump(mode="json"),
                  "other_reports": report_listing([x for x in reports if x.report_id != r.report_id])},
            evidence=[Evidence(source="report", report_id=r.report_id, finding_ids=[f.id for f in r.findings])],
            warnings=notes)

    def _pick_report_for_followup(self, patient: Patient, args: ReportQueryArgs | None, turn: Turn) -> tuple[Report | None, str]:
        reports = self.repo.reports_for(patient)
        if args is not None and args.report_specified:
            sel = select_report(reports, args.modality, args.which, parse_date_filter(args.date))
            return sel.report, sel.reason
        if turn.session.active_report_id:
            r = self.repo.get_report(turn.session.active_report_id, turn.clinician)
            if r and r.patient_id == patient.patient_id:
                return r, ""
        sel = select_report(reports, "ANY", "latest", None)
        return sel.report, sel.reason

    def _summarize(self, call: ToolCall, args: ReportQueryArgs, turn: Turn) -> AssistantResponse:
        res = self._resolve_patient(call, args, turn)
        if isinstance(res, AssistantResponse):
            return res
        patient, notes = res
        t = time.perf_counter()
        report, reason = self._pick_report_for_followup(patient, args, turn)
        turn.session.active_patient_id = patient.patient_id
        if report is None:
            turn.step("tool:summarize_report", t, patient_id=patient.patient_id, result=None)
            return self._respond(turn, "not_found", f"{patient.name} ({patient.patient_id}): {reason}", warnings=notes)
        s = summarize_report(report, patient)
        turn.step("tool:summarize_report", t, patient_id=patient.patient_id, report_id=report.report_id, method="extractive")
        turn.session.active_report_id = report.report_id
        self._audit(turn, "summarize_report", patient_id=patient.patient_id, report_id=report.report_id)
        return self._respond(
            turn, "summary", s["message"],
            data={"patient": patient.card(), "report_id": report.report_id, "points": s["points"],
                  "impression": s["impression"], "recommendation": s["recommendation"], "method": s["method"]},
            evidence=[Evidence(source="report", report_id=report.report_id, finding_ids=s["finding_ids"])],
            warnings=notes)

    def _answer(self, call: ToolCall, args: QuestionArgs, turn: Turn) -> AssistantResponse:
        res = self._resolve_patient(call, args, turn)
        if isinstance(res, AssistantResponse):
            return res
        patient, notes = res
        t = time.perf_counter()
        report, reason = self._pick_report_for_followup(patient, None, turn)
        turn.session.active_patient_id = patient.patient_id
        if report is None:
            return self._respond(turn, "not_found", f"{patient.name} ({patient.patient_id}): {reason}", warnings=notes)
        ans = answer_question(args.question, report, self.repo.reports_for(patient))
        turn.step("tool:answer_from_report", t, report_id=report.report_id, matched_findings=ans["finding_ids"],
                  fields=ans["fields"], outcome=ans["type"])
        turn.session.active_report_id = report.report_id
        self._audit(turn, "answer_question", patient_id=patient.patient_id, report_id=report.report_id)
        ev = [Evidence(source="report", report_id=report.report_id, finding_ids=ans["finding_ids"],
                       quote=None if ans["finding_ids"] else ", ".join(ans["fields"]) or None)]
        return self._respond(turn, ans["type"], ans["message"],
                             data={"patient": patient.card(), "report_id": report.report_id,
                                   "matched_findings": ans["finding_ids"], "fields": ans["fields"]},
                             evidence=ev if ans["type"] == "answer" else [], warnings=notes)

    def _dictation(self, call: ToolCall, args: DraftArgs, turn: Turn) -> AssistantResponse:
        t = time.perf_counter()
        rule = parse_dictation(turn.text)
        rejected = []
        added_from_rules = 0
        if call.source == "llm" and args.findings is not None:
            findings, rejected = ground_llm_findings(turn.text, args.findings)
            findings, added_from_rules = merge_missing(findings, rule.findings)
            engine = "llm+grounding" + ("+rules" if added_from_rules else "")
        else:
            findings = rule.findings
            engine = "rule_parser"
        turn.step("tool:draft_findings", t, engine=engine, accepted=len(findings), rejected=len(rejected),
                  added_from_rules=added_from_rules, unparsed=len(rule.unparsed),
                  needs_review=sum(f.needs_review for f in findings))

        patient = self._active_patient(turn)
        warnings: list[str] = []
        if patient:
            warnings.append(f"Linked to the patient open in this conversation: {patient.name} ({patient.patient_id}). "
                            f"Confirm before sign-off.")
        else:
            warnings.append("No patient linked. Open or name a patient before sign-off.")
        if not findings:
            warnings.append("No findings were recognised. The dictation is kept verbatim below.")
        if rejected:
            warnings.append(f"{len(rejected)} proposed finding(s) were rejected as unsupported by the transcript.")

        now = DraftStore.now()
        card = patient.card() if patient else None
        draft = DraftReport(
            draft_id=DraftStore.new_id(), status="draft_pending_review", created_by=turn.clinician.clinician_id,
            created_at=now, updated_at=now, patient=card, transcript=turn.text, findings=findings,
            unparsed_segments=rule.unparsed, rejected_findings=rejected, warnings=warnings,
            draft_text=build_draft_text(findings, rule.unparsed, card), extraction_engine=engine)
        self.drafts.save(draft)
        self._audit(turn, "create_draft", patient_id=patient.patient_id if patient else None, draft_id=draft.draft_id)
        n_review = sum(f.needs_review for f in findings)
        msg = (f"Draft {draft.draft_id} created with {len(findings)} finding(s)"
               + (f", {n_review} flagged for review" if n_review else "") + ". Clinician review is required before use.")
        return self._respond(
            turn, "draft_report", msg, data=draft.model_dump(mode="json"),
            evidence=[Evidence(source="transcript", quote=f.evidence, finding_ids=[f.finding_id]) for f in findings],
            warnings=warnings)

    # ================================================================== helpers
    def _unsupported(self, turn: Turn, message: str) -> AssistantResponse:
        return self._respond(turn, "unsupported", f"{message} Try: " + "; ".join(EXAMPLES) + ".")

    def _respond(self, turn: Turn, rtype: str, message: str, *, data: dict[str, Any] | None = None,
                 evidence: list[Evidence] | None = None, warnings: list[str] | None = None) -> AssistantResponse:
        return AssistantResponse(session_id=turn.session.session_id, type=rtype, message=message, data=data,  # type: ignore[arg-type]
                                 evidence=evidence or [], warnings=warnings or [])

    def _audit(self, turn: Turn, action: str, **ids: Any) -> None:
        # ids only: no transcript text or clinical content in the audit trail
        audit_log.info(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "clinician": turn.clinician.clinician_id,
                                   "session": turn.session.session_id, "action": action, **ids}))
