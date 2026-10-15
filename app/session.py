"""In-memory conversation context and draft storage (Redis/Postgres in production)."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from .interpreter.schema import ToolCall
from .models import DraftReport


@dataclass
class PendingClarification:
    call: ToolCall
    kind: Literal["choose_patient", "confirm_patient", "name_patient"]
    candidate_ids: list[str]
    question: str


@dataclass
class Session:
    session_id: str
    clinician_id: str
    last_seen: float = field(default_factory=time.time)
    active_patient_id: str | None = None
    active_report_id: str | None = None
    pending: PendingClarification | None = None
    turns: int = 0


class SessionStore:
    def __init__(self, ttl_s: int = 1800):
        self.ttl_s = ttl_s
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None, clinician_id: str) -> Session:
        now = time.time()
        with self._lock:
            for sid in [k for k, s in self._sessions.items() if now - s.last_seen > self.ttl_s]:
                del self._sessions[sid]
            s = self._sessions.get(session_id) if session_id else None
            # a session is bound to one clinician; never reuse another clinician's context
            if s is None or s.clinician_id != clinician_id:
                s = Session(session_id=session_id if session_id and s is None else uuid.uuid4().hex[:16],
                            clinician_id=clinician_id)
                self._sessions[s.session_id] = s
            s.last_seen = now
            return s


class DraftStore:
    def __init__(self) -> None:
        self._drafts: dict[str, DraftReport] = {}
        self._lock = threading.Lock()

    def save(self, draft: DraftReport) -> DraftReport:
        with self._lock:
            self._drafts[draft.draft_id] = draft
        return draft

    def get(self, draft_id: str, clinician_id: str) -> DraftReport | None:
        d = self._drafts.get(draft_id)
        if d is None or d.created_by != clinician_id:
            return None
        return d

    @staticmethod
    def new_id() -> str:
        return "DR-" + uuid.uuid4().hex[:8].upper()

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)


def summarize_pending(p: PendingClarification | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return {"kind": p.kind, "candidate_ids": p.candidate_ids, "original_tool": p.call.name}
