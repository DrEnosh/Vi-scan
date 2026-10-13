"""FastAPI app: HTTP layer only. All decisions live in the orchestrator and tools."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import DATA_DIR, STATIC_DIR, Settings, load_settings
from .interpreter.factory import build_interpreter
from .models import AssistantResponse, Clinician, DraftReport, DraftUpdateRequest, MessageRequest, TraceStep
from .orchestrator import Assistant
from .repository import Repository
from .session import DraftStore, SessionStore
from .stt import ALLOWED_AUDIO_TYPES, MockSTT, STTError, STTProvider, build_stt
from .tools.drafts import DraftError, apply_edits, approve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("viscan")


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": code, "message": message})


def create_app(settings: Settings | None = None, interpreter: Any = None, stt: STTProvider | None = None) -> FastAPI:
    settings = settings or load_settings()
    repo = Repository(DATA_DIR)
    drafts = DraftStore()

    llm_error: str | None = None
    if interpreter is None:
        try:
            interpreter = build_interpreter(settings)
            if interpreter is not None:
                log.info("LLM interpreter enabled: %s %s", settings.interpreter, settings.llm_model)
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"
            log.warning("LLM interpreter unavailable (%s); using rule-based interpreter", llm_error)
            interpreter = None
    assistant = Assistant(repo, SessionStore(settings.session_ttl_s), drafts, interpreter)

    if stt is None:
        try:
            stt = build_stt(settings.stt_provider, DATA_DIR / "audio_fixtures.json", settings.stt_api_key, settings.stt_model)
        except STTError as e:
            log.warning("STT provider '%s' unavailable (%s); using mock STT", settings.stt_provider, e)
            stt = MockSTT(DATA_DIR / "audio_fixtures.json")

    app = FastAPI(title="Viscan Voice Dental Assistant (PoC)", version="0.1.0",
                  description="Voice-first report retrieval and dictation drafting. Synthetic data only.")
    app.state.assistant = assistant
    app.state.repo = repo
    app.state.drafts = drafts
    app.state.stt = stt

    def get_clinician(x_clinician_id: str | None = Header(default=None)) -> Clinician:
        # Mock auth for the PoC. Production: verify a signed token (OIDC) and read the clinician from its claims.
        c = repo.get_clinician(x_clinician_id or settings.default_clinician_id)
        if c is None:
            raise _error(401, "unauthorized", "Unknown or missing clinician. Send a valid X-Clinician-Id header.")
        return c

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # never leak a stack trace to the client
        log.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal_error",
                                                      "message": "Something went wrong. Nothing was saved."})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        llm = assistant.interpreter if assistant.interpreter is not assistant.rules else None
        return {"interpreter": assistant.interpreter.name, "llm_model": getattr(llm, "model", None),
                "stt_model": getattr(stt, "model", None),
                "llm_requested": settings.interpreter not in ("rule_based", "", "rules"), "llm_startup_error": llm_error,
                "stt_provider": stt.name, "demo_clinicians": sorted(repo.clinicians), "synthetic_data": True}

    @app.post("/api/assistant/message", response_model=AssistantResponse)
    def message(req: MessageRequest, clinician: Clinician = Depends(get_clinician)) -> AssistantResponse:
        if len(req.text) > settings.max_text_chars:
            raise _error(413, "text_too_long", f"Text exceeds {settings.max_text_chars} characters.")
        return assistant.handle(req.text, clinician, req.session_id, req.mode, req.channel)

    @app.post("/api/assistant/voice", response_model=AssistantResponse)
    async def voice(audio: UploadFile = File(...), session_id: str | None = Form(default=None),
                    mode: str = Form(default="auto"), clinician: Clinician = Depends(get_clinician)) -> AssistantResponse:
        transcript, step = await _transcribe(audio)
        try:
            req = MessageRequest(text=transcript, session_id=session_id, mode=mode)  # type: ignore[arg-type]
        except Exception:
            raise _error(422, "invalid_request", "Invalid session_id or mode.")
        resp = assistant.handle(req.text, clinician, req.session_id, req.mode, "voice_upload")
        resp.transcript = transcript
        resp.trace.insert(0, step)
        logprob = step.detail.get("min_avg_logprob")
        if isinstance(logprob, (int, float)) and logprob < -1.0:
            resp.warnings.insert(0, "Low-confidence transcription. Check the transcript before relying on this result.")
        return resp

    @app.post("/api/stt")
    async def stt_only(audio: UploadFile = File(...), clinician: Clinician = Depends(get_clinician)) -> dict[str, Any]:
        transcript, step = await _transcribe(audio)
        return {"transcript": transcript, "provider": stt.name, "ms": step.ms}

    async def _transcribe(audio: UploadFile) -> tuple[str, TraceStep]:
        ctype = (audio.content_type or "application/octet-stream").split(";")[0].strip().lower()
        if ctype not in ALLOWED_AUDIO_TYPES:
            raise _error(415, "unsupported_audio", f"Unsupported audio type '{ctype}'.")
        limit = int(settings.max_audio_mb * 1024 * 1024)
        data = await audio.read(limit + 1)
        if not data:
            raise _error(400, "empty_audio", "The audio file is empty.")
        if len(data) > limit:
            raise _error(413, "audio_too_large", f"Audio exceeds {settings.max_audio_mb} MB.")
        t = time.perf_counter()
        meta: dict[str, Any] = {}
        try:
            with_meta = getattr(stt, "transcribe_with_meta", None)
            if with_meta is not None:
                transcript, meta = with_meta(data, audio.filename or "", ctype)
            else:
                transcript = stt.transcribe(data, audio.filename or "", ctype)
        except STTError as e:
            raise _error(422, "stt_failed", str(e))
        ms = round((time.perf_counter() - t) * 1000, 2)
        return transcript, TraceStep(step="stt", detail={"provider": stt.name, "bytes": len(data), **meta}, ms=ms)

    @app.get("/api/reports/{report_id}")
    def get_report(report_id: str, clinician: Clinician = Depends(get_clinician)) -> dict[str, Any]:
        r = repo.get_report(report_id, clinician)
        if r is None:
            raise _error(404, "not_found", "Report not found.")
        return r.model_dump(mode="json")

    @app.get("/api/drafts/{draft_id}", response_model=DraftReport)
    def get_draft(draft_id: str, clinician: Clinician = Depends(get_clinician)) -> DraftReport:
        d = drafts.get(draft_id, clinician.clinician_id)
        if d is None:
            raise _error(404, "not_found", "Draft not found.")
        return d

    @app.put("/api/drafts/{draft_id}", response_model=DraftReport)
    def edit_draft(draft_id: str, req: DraftUpdateRequest, clinician: Clinician = Depends(get_clinician)) -> DraftReport:
        d = drafts.get(draft_id, clinician.clinician_id)
        if d is None:
            raise _error(404, "not_found", "Draft not found.")
        try:
            return drafts.save(apply_edits(d.model_copy(deep=True), req, repo, clinician))
        except DraftError as e:
            raise _error(e.status, "draft_error", str(e))

    @app.post("/api/drafts/{draft_id}/approve", response_model=DraftReport)
    def approve_draft(draft_id: str, clinician: Clinician = Depends(get_clinician)) -> DraftReport:
        d = drafts.get(draft_id, clinician.clinician_id)
        if d is None:
            raise _error(404, "not_found", "Draft not found.")
        try:
            return drafts.save(approve(d.model_copy(deep=True), clinician))
        except DraftError as e:
            raise _error(e.status, "draft_error", str(e))

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
