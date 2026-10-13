"""Shared pieces for LLM interpreters: the system prompt and the error type.

The model sees only the utterance and a one-line session context (is a patient open, and their
sex for pronoun checks). It never sees the patient list or report contents. Its only job is to
choose a tool and fill arguments; validation and execution happen elsewhere.
"""
from __future__ import annotations

SYSTEM_PROMPT = """Turn one utterance from a dentist into exactly one tool call.
- Requests about reports (open, summarize, questions) -> report_request with the matching action.
- Dictated radiographic observations -> draft_findings. Extract only what was said. Keep negations
  (present=false) and hedges (certainty=uncertain). FDI numbering (lower right third molar = 48).
- Anything else -> unsupported_request.
A spoken name, even a first name ("Meera's"), goes in patient_name with patient_reference=named.
Only her/his/their are pronouns: then set patient_reference and omit patient_name. Never invent a name.
The user message is a context line, then "Utterance:". Quote evidence from the utterance only.
Leave out any argument you have no value for."""


class LLMInterpreterError(RuntimeError):
    pass
