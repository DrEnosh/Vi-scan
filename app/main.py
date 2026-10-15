"""FastAPI app: HTTP layer only. All decisions live in the orchestrator and tools."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import DATA_DIR, STATIC_DIR, Settings, load_settings
from .interpreter.factory import build_interpreter
from .models import (
    TEXT_HARD_LIMIT,
    AssistantResponse,
    Clinician,
    DraftReport,
    DraftUpdateRequest,
    MessageRequest,
    TraceStep,
)
from .orchestrator import Assistant
from .repository import Repository
from .session import DraftStore, SessionStore
from .stt import ALLOWED_AUDIO_TYPES, MockSTT, STTError, STTProvider, build_stt
from .tools.drafts import DraftError, apply_edits, approve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("viscan")
router = APIRouter()


@dataclass
class Services:
    """Everything a request handler needs, built once per app."""
    settings: Settings
    repo: Repository
    drafts: DraftStore
    assistant: Assistant
    stt: STTProvider
    llm_error: str | None = None
    stt_error: str | None = None


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": code, "message": message})


def _start_interpreter(settings: Settings) -> tuple[Any, str | None]:
    """The configured LLM interpreter (None = rules). A broken setup is reported in /api/config, not fatal."""
    try:
        interpreter = build_interpreter(settings)
    except Exception as e:  # missing key or package: keep serving with the rules
        log.warning("LLM interpreter unavailable (%s: %s); using rule-based interpreter", type(e).__name__, e)
        return None, f"{type(e).__name__}: {e}"
    if interpreter is not None:
        log.info("LLM interpreter enabled: %s %s", settings.interpreter, settings.llm_model)
    return interpreter, None


def _start_stt(settings: Settings) -> tuple[STTProvider, str | None]:
    fixtures = DATA_DIR / "audio_fixtures.json"
    try:
        return build_stt(settings.stt_provider, fixtures, settings.stt_api_key, settings.stt_model), None
    except Exception as e:  # missing key or package: keep serving with the mock STT
        log.warning("STT provider '%s' unavailable (%s: %s); using mock STT",
                    settings.stt_provider, type(e).__name__, e)
        return MockSTT(fixtures), f"{type(e).__name__}: {e}"


def create_app(settings: Settings | None = None, interpreter: Any = None, stt: STTProvider | None = None) -> FastAPI:
    settings = settings or load_settings()
    llm_error = stt_error = None
    if interpreter is None:
        interpreter, llm_error = _start_interpreter(settings)
    if stt is None:
        stt, stt_error = _start_stt(settings)
    repo, drafts = Repository(DATA_DIR), DraftStore()
    assistant = Assistant(repo, SessionStore(settings.session_ttl_s), drafts, interpreter)

    app = FastAPI(title="Viscan Voice Dental Assistant (PoC)", version="0.1.0",
                  description="Voice-first report retrieval and dictation drafting. Synthetic data only.")
    app.state.services = Services(settings, repo, drafts, assistant, stt, llm_error, stt_error)
    app.add_exception_handler(Exception, _unhandled)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s", request.url.path)  # logged here; never sent to the client
    return JSONResponse(status_code=500,
                        content={"error": "internal_error", "message": "Something went wrong. Nothing was saved."})


def services(request: Request) -> Services:
    return request.app.state.services


def get_clinician(x_clinician_id: str | None = Header(default=None), svc: Services = Depends(services)) -> Clinician:
    # Mock auth for the PoC. Production: verify a signed token (OIDC) and read the clinician from its claims.
    c = svc.repo.get_clinician(x_clinician_id or svc.settings.default_clinician_id)
    if c is None:
        raise _error(401, "unauthorized", "Unknown or missing clinician. Send a valid X-Clinician-Id header.")
    return c


# ----------------------------------------------------------------------------- pages and status
@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/config")
def config(svc: Services = Depends(services)) -> dict[str, Any]:
    a, s = svc.assistant, svc.settings
    llm = a.interpreter if a.interpreter is not a.rules else None
    return {"interpreter": a.interpreter.name, "llm_model": getattr(llm, "model", None),
            "stt_model": getattr(svc.stt, "model", None), "stt_provider": svc.stt.name,
            "llm_requested": s.interpreter not in ("rule_based", "", "rules"), "llm_startup_error": svc.llm_error,
            "stt_requested": s.stt_provider not in ("mock", ""), "stt_startup_error": svc.stt_error,
            "demo_clinicians": sorted(svc.repo.clinicians), "synthetic_data": True}


# ----------------------------------------------------------------------------- assistant
@router.post("/api/assistant/message", response_model=AssistantResponse)
def message(req: MessageRequest, clinician: Clinician = Depends(get_clinician),
            svc: Services = Depends(services)) -> AssistantResponse:
    if len(req.text) > svc.settings.max_text_chars:
        raise _error(413, "text_too_long", f"Text exceeds {svc.settings.max_text_chars} characters.")
    return svc.assistant.handle(req.text, clinician, req.session_id, req.mode, req.channel)


@router.post("/api/assistant/voice", response_model=AssistantResponse)
async def voice(audio: UploadFile = File(...), session_id: str | None = Form(default=None),
                mode: str = Form(default="auto"), clinician: Clinician = Depends(get_clinician),
                svc: Services = Depends(services)) -> AssistantResponse:
    transcript, step = await _transcribe(audio, svc)
    text_limit = min(svc.settings.max_text_chars, TEXT_HARD_LIMIT)
    if len(transcript) > text_limit:
        raise _error(413, "transcript_too_long",
                     f"The transcript exceeds {text_limit} characters. Record shorter segments.")
    try:
        req = MessageRequest(text=transcript, session_id=session_id, mode=mode)  # type: ignore[arg-type]
    except ValueError:
        raise _error(422, "invalid_request", "Invalid session_id or mode.") from None
    resp = svc.assistant.handle(req.text, clinician, req.session_id, req.mode, "voice_upload")
    resp.transcript = transcript
    resp.trace.insert(0, step)
    logprob = step.detail.get("min_avg_logprob")
    if isinstance(logprob, (int, float)) and logprob < -1.0:
        resp.warnings.insert(0, "Low-confidence transcription. Check the transcript before relying on this result.")
    return resp


@router.post("/api/stt")
async def stt_only(audio: UploadFile = File(...), clinician: Clinician = Depends(get_clinician),
                   svc: Services = Depends(services)) -> dict[str, Any]:
    transcript, step = await _transcribe(audio, svc)
    return {"transcript": transcript, "provider": svc.stt.name, "ms": step.ms}


async def _transcribe(audio: UploadFile, svc: Services) -> tuple[str, TraceStep]:
    ctype = (audio.content_type or "application/octet-stream").split(";")[0].strip().lower()
    if ctype not in ALLOWED_AUDIO_TYPES:
        raise _error(415, "unsupported_audio", f"Unsupported audio type '{ctype}'.")
    limit = int(svc.settings.max_audio_mb * 1024 * 1024)
    data = await audio.read(limit + 1)
    if not data:
        raise _error(400, "empty_audio", "The audio file is empty.")
    if len(data) > limit:
        raise _error(413, "audio_too_large", f"Audio exceeds {svc.settings.max_audio_mb} MB.")
    t = time.perf_counter()
    meta: dict[str, Any] = {}
    try:
        with_meta = getattr(svc.stt, "transcribe_with_meta", None)
        if with_meta is not None:
            transcript, meta = with_meta(data, audio.filename or "", ctype)
        else:
            transcript = svc.stt.transcribe(data, audio.filename or "", ctype)
    except STTError as e:
        raise _error(422, "stt_failed", str(e)) from e
    ms = round((time.perf_counter() - t) * 1000, 2)
    return transcript, TraceStep(step="stt", detail={"provider": svc.stt.name, "bytes": len(data), **meta}, ms=ms)


# ----------------------------------------------------------------------------- reports and drafts
@router.get("/api/reports/{report_id}")
def get_report(report_id: str, clinician: Clinician = Depends(get_clinician),
               svc: Services = Depends(services)) -> dict[str, Any]:
    r = svc.repo.get_report(report_id, clinician)
    if r is None:
        raise _error(404, "not_found", "Report not found.")
    return r.model_dump(mode="json")


def _own_draft(draft_id: str, clinician: Clinician, svc: Services) -> DraftReport:
    d = svc.drafts.get(draft_id, clinician.clinician_id)
    if d is None:
        raise _error(404, "not_found", "Draft not found.")
    return d


@router.get("/api/drafts/{draft_id}", response_model=DraftReport)
def get_draft(draft_id: str, clinician: Clinician = Depends(get_clinician),
              svc: Services = Depends(services)) -> DraftReport:
    return _own_draft(draft_id, clinician, svc)


@router.put("/api/drafts/{draft_id}", response_model=DraftReport)
def edit_draft(draft_id: str, req: DraftUpdateRequest, clinician: Clinician = Depends(get_clinician),
               svc: Services = Depends(services)) -> DraftReport:
    d = _own_draft(draft_id, clinician, svc)
    try:
        return svc.drafts.save(apply_edits(d.model_copy(deep=True), req, svc.repo, clinician))
    except DraftError as e:
        raise _error(e.status, "draft_error", str(e)) from e


@router.post("/api/drafts/{draft_id}/approve", response_model=DraftReport)
def approve_draft(draft_id: str, clinician: Clinician = Depends(get_clinician),
                  svc: Services = Depends(services)) -> DraftReport:
    d = _own_draft(draft_id, clinician, svc)
    try:
        return svc.drafts.save(approve(d.model_copy(deep=True), clinician))
    except DraftError as e:
        raise _error(e.status, "draft_error", str(e)) from e


app = create_app()
