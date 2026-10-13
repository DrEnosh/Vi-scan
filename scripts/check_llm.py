"""Live check of the configured LLM (Groq or any OpenAI-compatible API) and STT, using .env.

    python scripts/check_llm.py

1. Pre-flight: one direct tool call, so a bad key / model / network shows a clear message.
2. A short conversation through the full app (interpreter -> validation -> tools), printing what
   the model chose, latency, tokens and estimated cost of each call.
3. If STT_PROVIDER is groq/openai: one transcription call to confirm the speech key works.

Keys are never printed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.INFO)

from app.config import DATA_DIR, load_settings  # noqa: E402

CONVERSATION = [
    "Open Priya's latest OPG report",
    "Priya Sharma",
    "Summarize her report",
    "Does she have any caries?",
    "Can you get me the panoramic x-ray we took for Rahul Menon?",
    "Lower right third molar is horizontally impacted with roots close to the mandibular canal. "
    "Possible caries on 46. No periapical radiolucency in relation to 36. Patient is anxious about the procedure.",
    "Book me a CBCT slot for tomorrow",
]

FRIENDLY = {
    "AuthenticationError": "The API key was rejected. Check the key in .env (no quotes or spaces needed).",
    "PermissionDeniedError": "The key works but lacks access to this model.",
    "NotFoundError": "Model not found for this key. Set LLM_MODEL in .env to a model your account can use.",
    "RateLimitError": "Rate limited (free-tier limit or no credits). Wait a minute and retry.",
    "APIConnectionError": "Could not reach the API. Check internet / proxy / firewall.",
    "APITimeoutError": "The request timed out. Try again or raise LLM_TIMEOUT_S in .env.",
    "BadRequestError": "The API rejected the request. The message below has details.",
    "LLMInterpreterError": "The model answered but not with a usable tool call.",
}
KEY_ENV = {"groq": "GROQ_API_KEY", "openai_compat": "LLM_API_KEY"}


def fail(e: Exception) -> None:
    name = type(e).__name__
    print(f"  FAILED: {FRIENDLY.get(name, name)}\n  ({name}: {str(e)[:300]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", action="append", default=[],
                    help="real speech file to run through STT -> LLM -> tools (repeatable)")
    ap.add_argument("--pace", type=float, default=8.0,
                    help="seconds between LLM calls; Groq's free tier allows about 8 requests a minute at this size")
    opts = ap.parse_args()
    s = load_settings()
    if s.interpreter in ("rule_based", "", "rules"):
        print("INTERPRETER is rule_based, so no LLM is configured. Set INTERPRETER=groq and GROQ_API_KEY in .env.")
        return 2
    if not s.llm_api_key:
        print(f"No {KEY_ENV.get(s.interpreter, 'API key')} found. Add it to .env next to README.md.")
        return 2
    print(f"Provider: {s.interpreter}  model: {s.llm_model}  key loaded ({len(s.llm_api_key)} chars)  timeout: {s.llm_timeout_s}s")

    from app.interpreter import InterpretContext
    from app.interpreter.factory import build_interpreter

    try:
        interp = build_interpreter(s)
    except Exception as e:
        fail(e)
        if isinstance(e, ImportError):
            print("  Run: pip install -r requirements-providers.txt")
        return 2

    print("\n[1/3] Pre-flight tool call ...")
    try:
        call = interp.interpret("Open Priya Sharma's latest OPG report", InterpretContext(), "auto")
    except Exception as e:
        fail(e)
        return 1
    print(f"  OK -> {call.name} {call.args}\n  usage={call.usage}")

    print("\n[2/3] Conversation through the full app ...")
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app(s, interpreter=interp))
    sid = None
    calls = fallbacks = 0
    cost = 0.0
    lat: list[float] = []
    for text in CONVERSATION:
        time.sleep(opts.pace)
        r = client.post("/api/assistant/message", json={"text": text, "session_id": sid}).json()
        sid = r["session_id"]
        print(f"\n> {text}")
        for st in r["trace"]:
            d = st["detail"]
            if st["step"] == "interpret" and d.get("engine") not in ("rule_based", None):
                calls += 1
                u = d.get("usage") or {}
                cost += u.get("est_cost_usd") or 0
                lat.append(st["ms"] or 0)
                shown = d["args"] if d["tool"] != "draft_findings" else "(findings below)"
                print(f"  {d['engine']} -> {d['tool']} {shown}")
                print(f"           {st['ms']} ms, {u.get('input_tokens')} in ({u.get('cached_tokens', 0)} cached) + "
                      f"{u.get('output_tokens')} out tokens, ~${u.get('est_cost_usd')}")
            elif st["step"] == "interpret_fallback":
                fallbacks += 1
                print(f"  FALLBACK to rules: {d.get('error')}")
            elif st["step"] == "clarification":
                print(f"  (answered the pending question: {d.get('result')}, no LLM call needed)")
            elif st["step"] == "validate" and not d.get("ok"):
                print(f"  validation rejected the model's call: {d.get('error')}")
        print(f"  => {r['type']}: {r['message'].splitlines()[0][:160]}")
        if r["type"] == "draft_report":
            dr = r["data"]
            dstep = next((x for x in r["trace"] if x["step"] == "tool:draft_findings"), {})
            print(f"     engine={dr['extraction_engine']} added_from_rules={dstep.get('detail', {}).get('added_from_rules')}")
            for f in dr["findings"]:
                print(f"     + {f['statement']}" + (f"  [review: {'; '.join(f['review_reasons'])}]" if f["needs_review"] else ""))
            for x in dr["rejected_findings"]:
                print(f"     - rejected {x['proposed'].get('category')}: {x['reason']}")
            for u in dr["unparsed_segments"]:
                print(f"     ~ kept verbatim: \"{u}\"")

    print("\n[3/3] Speech-to-text ...")
    if s.stt_provider in ("groq", "openai"):
        from app.stt import STTError, build_stt

        try:
            stt = build_stt(s.stt_provider, DATA_DIR / "audio_fixtures.json", s.stt_api_key, s.stt_model)
            wav = (ROOT / "samples" / "dictation_48_impacted.wav").read_bytes()  # 0.5 s of silence
            try:
                out, meta = stt.transcribe_with_meta(wav, "check.wav", "audio/wav")
                print(f"  {s.stt_provider} {s.stt_model}: silent clip transcribed as {out!r}  {meta}")
                print("  WARNING: silence produced text; the no-speech guard did not catch it")
                fallbacks += 1
            except STTError as e:
                if "No speech" in str(e):
                    print(f"  OK ({s.stt_provider} {s.stt_model}) -> key accepted; silent clip correctly rejected as "
                          f"no speech  {getattr(e, 'meta', {})}")
                else:
                    raise
            print("  Use the mic button in the web UI to try real speech.")
        except Exception as e:
            fail(e.__cause__ or e)
            fallbacks += 1
    else:
        print(f"  STT_PROVIDER={s.stt_provider}: skipped (set STT_PROVIDER=groq for real transcription)")

    if opts.audio:
        print("\n[extra] Real speech through the whole pipeline (POST /api/assistant/voice) ...")
        vsid = None
        for path in opts.audio:
            p = Path(path)
            if not p.exists():
                print(f"  {path}: not found, skipped")
                continue
            time.sleep(opts.pace)
            r = client.post("/api/assistant/voice", files={"audio": (p.name, p.read_bytes(), "audio/wav")},
                            data={"session_id": vsid} if vsid else {})
            if r.status_code != 200:
                print(f"  {p.name}: HTTP {r.status_code} {r.text[:200]}")
                fallbacks += 1
                continue
            body = r.json()
            vsid = body["session_id"]
            stt_step = next((x for x in body["trace"] if x["step"] == "stt"), {})
            llm_step = next((x for x in body["trace"] if x["step"] == "interpret"), {})
            print(f"\n  {p.name}")
            sd = stt_step.get("detail", {})
            print(f"    heard ({sd.get('provider')}, {stt_step.get('ms')} ms): {body.get('transcript')!r}")
            print(f"    whisper segments={sd.get('segments')} max_no_speech_prob={sd.get('max_no_speech_prob')} "
                  f"min_avg_logprob={sd.get('min_avg_logprob')}")
            lu = llm_step.get("detail", {}).get("usage") or {}
            print(f"    {llm_step.get('detail', {}).get('engine')} -> {llm_step.get('detail', {}).get('tool')} "
                  f"({llm_step.get('ms')} ms, {lu.get('cached_tokens', 0)}/{lu.get('input_tokens')} input tokens cached)")
            if any(x["step"] == "interpret_fallback" for x in body["trace"]):
                fallbacks += 1
                print("    FALLBACK to rules on this one")
            print(f"    => {body['type']}: {body['message'].splitlines()[0][:160]}")
            if body["type"] == "draft_report":
                for f in body["data"]["findings"]:
                    print(f"       + {f['statement']}")

    print("\n" + "-" * 60)
    if lat:
        print(f"LLM calls: {calls}, avg {sum(lat) / len(lat):.0f} ms, max {max(lat):.0f} ms, est. cost ${cost:.5f}")
    if fallbacks:
        print(f"{fallbacks} problem(s) above. The app still works (rule-based fallback), but check the errors.")
        return 1
    print("Integration working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
