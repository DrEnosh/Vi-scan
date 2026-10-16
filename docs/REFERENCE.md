# Reference

Details that used to live in the README: configuration, the API, the mock data, why each model was chosen, what the live runs taught me, and production notes.

## Configuration

All settings are environment variables, read from `.env` (copy `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `INTERPRETER` | `rule_based` | `rule_based` (offline), `groq` or `openai_compat` |
| `GROQ_API_KEY` | – | Groq key; used by `INTERPRETER=groq` and `STT_PROVIDER=groq` |
| `LLM_BASE_URL`, `LLM_API_KEY` | – | For `INTERPRETER=openai_compat` |
| `LLM_MODEL` | per provider | Groq: `openai/gpt-oss-120b` |
| `LLM_REASONING_EFFORT` | per model | Groq reasoning models: `low` for gpt-oss, `none` for Qwen (keeps latency down) |
| `LLM_TIMEOUT_S` | `8` | Per-call timeout before falling back to rules |
| `LLM_PRICE_IN_PER_MTOK` / `LLM_PRICE_OUT_PER_MTOK` | `0` | USD per million tokens, only for the per-call cost estimate (0 on Groq's free tier) |
| `STT_PROVIDER` | `mock` | `mock` (file-name lookup for tests), `groq` or `openai` |
| `STT_MODEL` | per provider | Groq: `whisper-large-v3-turbo`; OpenAI: `whisper-1` |
| `OPENAI_API_KEY` | – | For `STT_PROVIDER=openai` |
| `MAX_AUDIO_MB` / `MAX_TEXT_CHARS` | `10` / `2000` | Input limits |
| `SESSION_TTL_S` | `1800` | Idle time before conversation context is dropped |
| `DEFAULT_CLINICIAN_ID` | `dr_demo` | Used when no `X-Clinician-Id` header is sent (demo convenience; set empty to require the header) |

**Other providers.** `INTERPRETER=openai_compat` with `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` works with any OpenAI-compatible API (OpenAI, OpenRouter, Cerebras, GitHub Models and others). `STT_PROVIDER=openai` with `OPENAI_API_KEY` uses OpenAI's `whisper-1`.

**Failure behaviour.** If the LLM call fails, times out, hits a rate limit or returns an unusable tool call, the request is re-interpreted by the rule-based interpreter straight away (no retry wait). The trace records why, and the UI shows an "LLM unavailable" note on that answer, so the clinician still gets a response. If the key or the provider package is missing at startup, the server still starts in offline mode, and `/api/config` and the UI header say why.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/assistant/message` | Main entry: `{text, session_id?, mode: auto\|command\|dictation}` |
| `POST` | `/api/assistant/voice` | Multipart `audio` (+ `session_id`, `mode`): STT, then the same pipeline |
| `POST` | `/api/stt` | Transcribe only |
| `GET` | `/api/reports/{id}` | Fetch a report (clinic-scoped) |
| `GET` / `PUT` | `/api/drafts/{id}` | Read or edit a draft (edits are re-validated) |
| `POST` | `/api/drafts/{id}/approve` | Clinician sign-off |
| `GET` | `/api/config`, `/health` | Runtime info (active provider and models) |

Every assistant response has the same shape: `type` (report, clarification, not_found, summary, answer, draft_report, unsupported, cancelled), `message`, `data`, `evidence` (report and finding IDs, or transcript quotes), `warnings`, and a `trace` of each step with timings (plus model and token counts for LLM steps, and confidence data for STT). Interactive docs are at `/docs` when the server runs.

`docs/DEMO.md` is a full sequence of calls with real outputs and curl commands, generated offline by `scripts/run_demo.py`. `docs/LIVE_RUN.md` is the live run against Groq.

## Mock data

Nine synthetic patients, 18 reports and two clinicians (`app/data/`). Every patient has two to four reports except Kiran Das, who has none on purpose. The edge cases are deliberate:

| Patient | Why it's there |
|---|---|
| Priya Sharma (P001), Priya Shetty (P002) | Same first name: ambiguity |
| Ananya Rao (P006), Anaya Rao (P007, age 8) | Sound-alike names, shared surname; P007 has primary-tooth (FDI 75) findings |
| Priya Shah (P008) | Belongs to another clinic (MYS-02): invisible to `dr_demo` |
| Kiran Das (P009) | No reports on file |
| Arjun Nair (P004) | Has OPG and ceph, no CBCT: missing-report path |
| Rahul Menon, Fatima Khan | Root stumps, implant, sinus and bone-loss findings for Q&A |

`samples/` holds silent placeholder WAVs for the mock STT (the file name maps to a transcript) and three spoken clips made with Windows text-to-speech (`scripts/make_speech_samples.ps1`, which `run_check.bat` also runs).

## Model and provider choices

**Interpreter: a small LLM via tool calling, with a rule-based twin.** The interpreter's job is narrow: pick one of three tools (`report_request` with an open/summarize/question action, `draft_findings`, `unsupported_request`) and fill a few arguments. `openai/gpt-oss-120b` on Groq is free to start with, handles tool calling well and answers in about a second, which matters for voice. Reasoning effort is kept low because this is classification, not problem solving. The model never sees the patient list or report contents, only the utterance plus a one-line context ("a patient is open, sex F") for pronoun checks. Swapping models or providers is a config change.

The rule-based interpreter emits the same internal tool calls, so the system runs and is testable without a key, and it is the fallback when the LLM fails. It finds patient names by evidence rather than by elimination: a word is a name only if it matches the clinic's patient list (exact or a close spelling) or sits where a name sits ("Priya's", "patient Rahul", "does Kavya have", "open Arjun"). Unknown clinical words therefore stay part of the question instead of being read as a name, and its only fixed word lists are grammar. `docs/EVAL.md` compares it with the LLM.

**Dictation extraction: LLM proposals, checked by a deterministic vocabulary.** A controlled dental vocabulary (FDI notation including spoken forms like "four eight", 23 finding types, negation and hedge detection) decides what counts as a finding. The model's proposals must pass a grounding check against the transcript, and parser findings the model missed are added. A draft report is where hallucination does the most harm, so the model is never trusted on its own there.

**Summaries: extractive, by design.** "Summarize her report" returns the report's own findings, impression and recommendation, each tagged with its finding ID. It reads less smoothly than a generated paragraph, but every line is traceable, which is what the brief asks for.

**STT: Whisper on Groq.** `whisper-large-v3-turbo` with a dental-terms prompt (OPG, IOPA, mesioangular, periapical radiolucency...), `language=en` and `verbose_json` so segment confidence is available. Browser speech recognition is the fallback when no server STT is configured, and the mock provider keeps the voice endpoint testable offline. Adding Deepgram, Azure or a self-hosted Whisper is one class in `app/stt.py`.

## What live testing caught

Measured with `run_check.bat` against Groq's free tier (full log in `docs/LIVE_RUN.md`):

| Step | Result |
|---|---|
| Interpreter calls | 9 of 9 correct tool choices (6 typed, 3 spoken); 590–1,610 ms each (about 1 s average); about 860 input tokens per call |
| "Open Priya's latest OPG report" | Model passed "Priya"; code found two Priyas and asked which one |
| "Can you get me the panoramic x-ray we took for Rahul Menon?" | Opened R-3001 |
| Dictation through the LLM | 4 grounded findings, "Patient is anxious about the procedure" kept verbatim, "Possible caries" flagged |
| "Book me a CBCT slot for tomorrow" | `unsupported_request` |
| Whisper on spoken clips (Windows TTS) | Exact transcripts in 350–480 ms; "tooth 4-6" mapped to tooth 46 |
| Whisper on a silent clip | Returned "Thank you."; rejected by the no-speech guard |
| Live API tests | 4 passed |

Problems the scripted tests had not shown, and the fixes:

1. **Groq rejected tool calls with `null` arguments.** The model sent `"patient_id": null`, and Groq validates generated calls against the schema, so the request failed with a 400. Optional arguments are now nullable in the schema and nulls are dropped before validation.
2. **Whisper "heard" speech in silence.** A silent clip came back as "Thank you.", a known Whisper failure. `no_speech_prob` was 0.0 on this model, so it couldn't be relied on; the guard also rejects transcripts that consist only of known silence phrases, and low average log-probability adds a "check the transcript" warning.
3. **Free-tier rate limit.** At about 1,300 tokens per call, Groq's free tier (8,000 tokens per minute for this model) allowed about 6 calls a minute. Merging three report tools into one and tightening descriptions cut each call to about 860 tokens; the system prompt is static so Groq's automatic prompt caching can apply; and rate-limited calls fall back to the rules immediately.
4. **The model missed a finding.** In one dictation it skipped "roots close to the mandibular canal", which the deterministic parser caught. Grounded parser findings that the model missed are now merged in (`extraction_engine: llm+grounding+rules`). The model had also attached that phrase to the impaction as a qualifier; qualifiers that describe a separate finding are now dropped.
5. **The model dropped a spoken name.** Once it read "Priya's" as a pronoun. With another patient open that could have resolved to the wrong person. If a name that matches a patient in scope is spoken and the model returns none, the spoken name is used (`name_cross_check` in the trace).

## Production notes

**Privacy.** The LLM receives the utterance and a one-line context, never the patient list, report text or IDs. The audit log records clinician, action and record IDs, not transcript text or clinical content. Free tiers are fine for synthetic data only. With real patients I would sign a DPA with each provider (or self-host Whisper and an open-weight model in-region for DPDP Act compliance), use zero-retention settings, encrypt drafts at rest, and redact PHI from application logs.

**Authorization.** Every lookup is scoped to the clinician's clinics, and out-of-scope patients behave as if they don't exist (no "forbidden" that leaks existence). Sessions are bound to one clinician, and drafts are visible only to their author. The `X-Clinician-Id` header is a stand-in; production would validate an OIDC token and add role checks (for example, only radiologists approve).

**Latency.** Measured live: Whisper 350–480 ms per short clip, interpreter about 1 s (590–1,610 ms), deterministic tools and validation about 1 ms (p50 1.0 ms, p95 2.1 ms over 400 requests). The three spoken commands in the live run took 1.05–1.76 s on the server from audio to answer, about 1.3 s on average; the browser upload comes on top. To go faster: streaming STT, rules first for unambiguous commands, and the LLM for dictation and anything the rules can't parse cleanly.

**Costs.** Free within Groq's free-tier limits. At Groq's published paid rates (mid-2026: $0.15 / $0.60 per million input / output tokens for `gpt-oss-120b`, $0.04 per audio hour for `whisper-large-v3-turbo` with a 10-second minimum per request), a spoken command costs about $0.0003 (about $0.00019 for the interpreter at ~860 input and ~100 output tokens, plus about $0.00011 for STT), or roughly $0.30 per 1,000 commands. The rule-based mode costs nothing.

## Project layout

```
app/
  main.py               FastAPI routes, auth stub, error handling
  orchestrator.py       interpret -> validate -> tool -> respond; session context; clarifications
  interpreter/          schema.py (tool contract), openai_compat.py (Groq / OpenAI-compatible),
                        llm.py (shared prompt), rule_based.py (offline twin), factory.py (picks one from .env)
  validation.py         tool-call schema checks + grounding of LLM findings
  dental.py             FDI notation, site parsing, finding vocabulary, negation/hedging
  tools/                retrieval.py, dictation.py, drafts.py
  stt.py                mock and Whisper (Groq / OpenAI) STT providers, no-speech guard
  repository.py         clinic-scoped data access over the JSON mock data
  data/                 synthetic patients, reports, clinicians, mock STT fixtures
static/index.html       single-page UI (mic recording, chat, evidence panel, draft editor)
tests/                  automated tests + 4 opt-in live API tests
eval/cases.jsonl        134 labelled utterances; scored by scripts/run_eval.py
scripts/                run_eval.py, run_demo.py (writes docs/DEMO.md), check_llm.py (live check)
start.bat               one-click start on Windows
run_check.bat           Windows: setup + tests + live check + live tests -> check_output.txt
docs/                   ARCHITECTURE.md, architecture.pdf, EVAL.md, DEMO.md, LIVE_RUN.md, this file
```
