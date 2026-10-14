"""Speech-to-text providers behind one small interface.

Voice can reach the system three ways:
  1. Web UI mic with a server STT provider configured: the browser records audio and posts it to
     /api/assistant/voice, which transcribes it with Whisper (Groq or OpenAI).
  2. Web UI mic without one: browser speech recognition (Web Speech API, en-IN) sends text.
  3. A typed transcript (always available).

Providers:
  * MockSTT         - deterministic; maps known file names to transcripts (tests, offline demo)
  * OpenAICompatSTT - Whisper via Groq (free tier, whisper-large-v3-turbo) or OpenAI (whisper-1),
                      with a dental-vocabulary prompt
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a",
                       "audio/x-m4a", "audio/webm", "audio/ogg", "video/webm", "application/octet-stream"}
DENTAL_PROMPT = ("Dental radiology dictation. Terms: OPG, IOPA, CBCT, bitewing, cephalogram, FDI tooth numbers 11-48, "
                 "third molar, mesioangular, distoangular, horizontally impacted, periapical radiolucency, caries, "
                 "mandibular canal, root canal treated, obturation.")


class STTError(RuntimeError):
    pass


class STTProvider(Protocol):
    name: str

    def transcribe(self, audio: bytes, filename: str, content_type: str) -> str: ...


class MockSTT:
    name = "mock"

    def __init__(self, fixtures_path: Path):
        data = json.loads(fixtures_path.read_text())
        self.fixtures = {k: v for k, v in data.items() if not k.startswith("_")}

    def transcribe(self, audio: bytes, filename: str, content_type: str) -> str:
        key = Path(filename or "").name
        if key in self.fixtures:
            return self.fixtures[key]
        raise STTError("Mock STT has no transcript for this file. Type the request, or set STT_PROVIDER=groq "
                       "(free key) for real transcription. Known sample files: "
                       + ", ".join(sorted(self.fixtures)))


# Whisper is known to "hear" these on silence or noise. A command that is only one of these is
# treated as no speech rather than sent on as a request.
SILENCE_HALLUCINATIONS = {
    "thank you", "thank you very much", "thanks", "thanks for watching", "thank you for watching",
    "you", "bye", "bye bye", "please subscribe", "subtitles by the amara org community", "so", "okay",
}
NO_SPEECH_PROB = 0.6     # Whisper segment-level probability that the segment is not speech


def _norm_words(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def _seg_get(seg: object, name: str) -> object:
    return seg.get(name) if isinstance(seg, dict) else getattr(seg, name, None)


class OpenAICompatSTT:
    """OpenAI-style /audio/transcriptions. Used for Groq (whisper-large-v3-turbo) and OpenAI (whisper-1)."""

    def __init__(self, api_key: str | None, model: str, base_url: str | None = None, name: str = "openai",
                 client: object = None):
        self.name = name
        if client is None:
            if not api_key:
                raise STTError(f"API key for STT provider '{name}' is not set")
            try:
                from openai import OpenAI  # lazy import: optional dependency
            except ImportError as e:
                raise STTError("The 'openai' package is not installed. "
                               "Run: pip install -r requirements-providers.txt") from e

            client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=1)
        self.client = client
        self.model = model

    def transcribe(self, audio: bytes, filename: str, content_type: str) -> str:
        return self.transcribe_with_meta(audio, filename, content_type)[0]

    def transcribe_with_meta(self, audio: bytes, filename: str, content_type: str) -> tuple[str, dict]:
        verbose = "whisper" in self.model.lower()
        try:
            kwargs = dict(model=self.model, file=(filename or "audio.webm", audio, content_type),
                          language="en", prompt=DENTAL_PROMPT, temperature=0)
            if verbose:
                kwargs["response_format"] = "verbose_json"
            result = self.client.audio.transcriptions.create(**kwargs)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover - network path
            raise STTError(f"Transcription failed ({type(e).__name__}). Try again or type the request.") from e

        text = (getattr(result, "text", "") or "").strip()
        meta: dict = {"model": self.model}
        segments = getattr(result, "segments", None) or []
        if segments:
            kept, dropped = [], 0
            for seg in segments:
                nsp = _seg_get(seg, "no_speech_prob")
                if isinstance(nsp, (int, float)) and nsp >= NO_SPEECH_PROB:
                    dropped += 1
                    continue
                kept.append((_seg_get(seg, "text") or "").strip())
            meta.update(segments=len(segments), dropped_no_speech=dropped,
                        max_no_speech_prob=round(max(float(_seg_get(x, "no_speech_prob") or 0) for x in segments), 3),
                        min_avg_logprob=round(min(float(_seg_get(x, "avg_logprob") or 0) for x in segments), 3))
            if dropped:
                text = " ".join(t for t in kept if t).strip()
        if not text or _norm_words(text) in SILENCE_HALLUCINATIONS:
            meta["rejected_text"] = text or None
            err = STTError("No speech detected in the audio. Please try again.")
            err.meta = meta  # type: ignore[attr-defined]
            raise err
        return text, meta


def build_stt(provider: str, fixtures_path: Path, api_key: str | None, model: str) -> STTProvider:
    if provider == "groq":
        return OpenAICompatSTT(api_key, model or "whisper-large-v3-turbo", "https://api.groq.com/openai/v1", name="groq")
    if provider == "openai":
        return OpenAICompatSTT(api_key, model or "whisper-1", None, name="openai")
    return MockSTT(fixtures_path)
