# Viscan Voice Dental Assistant (proof of concept)

A voice-first assistant for a dental imaging workflow. A clinician can say "Open Priya's latest OPG report", follow up with "Summarize her report", and dictate observations like "Lower right third molar is horizontally impacted" to get an editable, structured draft report.

The design goal was safety before fluency. The LLM only proposes a tool call. Patient identity, report selection and "is this finding actually in the transcript?" are decided by plain code that the tests pin down. Without an API key, a deterministic interpreter produces the same tool calls, so everything also runs offline.

All data in this repository is synthetic.

## What works

| Assignment item | Status | Where |
|---|---|---|
| A. Report retrieval from typed or spoken requests | Done | `app/tools/retrieval.py`, `app/orchestrator.py` |
| A. Ambiguous patients and missing reports handled safely | Done: asks which patient, never guesses; lists what is on file | `match_patients`, `select_report` |
| B. Dictation to editable structured JSON + readable draft | Done, with negation, hedging, FDI numbering, evidence quotes | `app/tools/dictation.py` |
| B. No invented observations, uncertainty labelled, clinician review | Done: draft stays "pending review" until approved; no impression is ever generated | `app/validation.py`, `app/tools/drafts.py` |
| C. Context-dependent follow-up with source evidence | Done: "her"/"his" resolved from session with a sex-consistency check; every answer returns report and finding IDs | `_resolve_patient`, `summarize_report`, `answer_question` |
| Bonus: report-aware Q&A | Done: grounded in the selected report; says when something isn't documented | `answer_question` |
| Tool calling, LLM separated from business rules | Done: Groq (`openai/gpt-oss-120b`) tool calling, or any OpenAI-compatible API; schema validation, grounding validator and identity rules in code | `app/interpreter/`, `app/validation.py` |
| Voice input | Done: the mic records audio and sends it to Whisper (`whisper-large-v3-turbo` on Groq) with a no-speech guard | `static/index.html`, `app/stt.py` |
| Tests | 62 automated tests, plus 4 opt-in live tests against the real APIs (passing) | `tests/` |

## Quick start (no keys needed)

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # Windows: copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

On Windows you can double-click `start.bat` instead: the first run creates `.venv`, installs everything and copies `.env.example` to `.env`; every run opens http://localhost:8000.

Things to try, in order, in one session:

1. `Open Priya's latest OPG report` → asks which Priya
2. `Priya Sharma` (or click the button) → opens R-1003
3. `Summarize her report` → extractive summary with finding IDs
4. `Does she have any caries?` → R-1003 doesn't mention caries, so it says so and points to R-1002/R-1001
5. `Does she have a tumour?` → not something the report documents, so it is neither confirmed nor denied
6. `Lower right third molar is horizontally impacted with roots close to the mandibular canal. Possible caries on 46. No periapical radiolucency in relation to 36.` → draft with four findings; edit the JSON, then approve

Run the tests and regenerate the offline demo log:

```bash
pytest                            # 62 passed, 4 skipped (the live API tests)
python scripts/run_demo.py        # writes docs/DEMO.md from real responses
```

## Turning on the LLM and real voice (free Groq key)

One free Groq key (console.groq.com) covers both the LLM and Whisper transcription.

1. `pip install -r requirements-providers.txt`
2. In `.env`:
   ```
   INTERPRETER=groq
   STT_PROVIDER=groq
   GROQ_API_KEY=gsk_...
   ```
3. Check it end to end:
   ```bash
   python scripts/check_llm.py --audio samples/speech_open_priya.wav
   ```
   A pre-flight call (with a plain-English message if the key, model, rate limit or network is the problem), a short conversation through the whole app showing the tool the model chose with latency and tokens, a silent clip to confirm the no-speech guard, and optionally real speech files through STT → LLM → tools.
4. Restart the server. The header shows `interpreter: Groq (openai/gpt-oss-120b)` and `server STT: groq (whisper-large-v3-turbo)`, and each answer shows model, latency and tokens under it.

On Windows, `run_check.bat` does all of this in one go: it sets up `.venv`, makes three spoken test clips with Windows text-to-speech, runs the tests, the live check and the live API tests, and writes everything to `check_output.txt`. `docs/LIVE_RUN.md` is the output of that script on my machine.

Live API tests (skipped by default so CI never uses quota): `RUN_LIVE=1 pytest -m live` (PowerShell: `$env:RUN_LIVE=1; pytest -m live`).

**Other providers.** `INTERPRETER=openai_compat` with `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` works with any OpenAI-compatible API (OpenAI, OpenRouter, Cerebras, GitHub Models and others). `STT_PROVIDER=openai` with `OPENAI_API_KEY` uses OpenAI's `whisper-1`.

**Failure behaviour.** If the LLM call fails, times out, hits a rate limit or returns an unusable tool call, the request is re-interpreted by the rule-based interpreter straight away (no retry wait). The trace records why, and the UI shows an "LLM unavailable" note on that answer, so the clinician still gets a response. If the key or the provider package is missing at startup, the server still starts in offline mode, and `/api/config` and the UI header say why.

## Live verification

Measured with `run_check.bat` against Groq's free tier (full log in `docs/LIVE_RUN.md`):

| Step | Result |
|---|---|
| Interpreter calls | 9 of 9 correct tool choices (6 typed, 3 spoken); 590–1,610 ms each (about 1 s average); about 860 input tokens per call |
| "Open Priya's latest OPG report" | Model passed "Priya"; code found two Priyas and asked which one |
| "Can you get me the panoramic x-ray we took for Rahul Menon?" | Opened R-3001 (the offline parser needed extra rules for this phrasing; the model handled it directly) |
| Dictation through the LLM | 4 grounded findings, "Patient is anxious about the procedure" kept verbatim, "Possible caries" flagged |
| "Book me a CBCT slot for tomorrow" | `unsupported_request` |
| Whisper on real speech (Windows TTS clips) | Exact transcripts in 350–480 ms; "tooth 4-6" mapped to tooth 46 |
| Whisper on a silent clip | Returned "Thank you."; rejected by the no-speech guard |
| Live API tests | 4 passed |

### What live testing caught (and the fixes)

1. **Groq rejected tool calls with `null` arguments.** The model sent `"patient_id": null`, and Groq validates generated calls against the schema, so the request failed with a 400. Optional arguments are now nullable in the schema and nulls are dropped before validation.
2. **Whisper "heard" speech in silence.** A silent clip came back as "Thank you.", a known Whisper failure. `no_speech_prob` was 0.0 on this model, so it couldn't be relied on; the guard also rejects transcripts that consist only of known silence phrases, and low average log-probability adds a "check the transcript" warning.
3. **Free-tier rate limit.** At about 1,300 tokens per call, Groq's free tier (8,000 tokens per minute for this model) allowed about 6 calls a minute. Merging three report tools into one and tightening descriptions cut each call to about 860 tokens; the system prompt is now static so Groq's automatic prompt caching can apply (cached tokens don't count toward the limit); and rate-limited calls fall back to the rules immediately.
4. **The model missed a finding.** In one dictation it skipped "roots close to the mandibular canal", which the deterministic parser caught. Grounded parser findings that the model missed are now merged in (`extraction_engine: llm+grounding+rules`). The model had also attached that phrase to the impaction as a qualifier; qualifiers that describe a separate finding are now dropped.
5. **The model dropped a spoken name.** Once it read "Priya's" as a pronoun. That was harmless here, but with another patient open it could have resolved to the wrong person. If a name that matches a patient in scope is spoken and the model returns none, the spoken name is now used (`name_cross_check` in the trace).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `INTERPRETER` | `rule_based` | `rule_based` (offline), `groq` or `openai_compat` |
| `GROQ_API_KEY` | – | Groq key; used by `INTERPRETER=groq` and `STT_PROVIDER=groq` |
| `LLM_BASE_URL`, `LLM_API_KEY` | – | For `INTERPRETER=openai_compat` |
| `LLM_MODEL` | per provider | Groq: `openai/gpt-oss-120b` |
| `LLM_REASONING_EFFORT` | per model | Groq reasoning models: `low` for gpt-oss, `none` for Qwen (keeps latency down) |
| `LLM_TIMEOUT_S` | `8` | Per-call timeout before falling back to rules |
| `LLM_PRICE_IN_PER_MTOK` / `LLM_PRICE_OUT_PER_MTOK` | `0` | USD per million tokens, only for the per-call cost estimate (0 on Groq's free tier; set Groq's paid rates to see paid costs) |
| `STT_PROVIDER` | `mock` | `mock` (file-name lookup for tests), `groq` or `openai` |
| `STT_MODEL` | per provider | Groq: `whisper-large-v3-turbo`; OpenAI: `whisper-1` |
| `OPENAI_API_KEY` | – | For `STT_PROVIDER=openai` |
| `MAX_AUDIO_MB` / `MAX_TEXT_CHARS` | `10` / `2000` | Input limits |
| `SESSION_TTL_S` | `1800` | Idle time before conversation context is dropped |
| `DEFAULT_CLINICIAN_ID` | `dr_demo` | Used when no `X-Clinician-Id` header is sent (demo convenience; set empty to require the header) |

## Model and provider choices

**Interpreter: a small LLM via tool calling, with a rule-based twin.** The interpreter's job is narrow: pick one of three tools (`report_request` with an open/summarize/question action, `draft_findings`, `unsupported_request`) and fill a few arguments. `openai/gpt-oss-120b` on Groq is free to start with, handles tool calling well and answers in about a second, which matters for voice. Reasoning effort is kept low because this is classification, not problem solving. The model never sees the patient list or report contents, only the utterance plus a one-line context ("a patient is open, sex F") for pronoun checks. Swapping models or providers is a config change. The rule-based interpreter emits the same internal tool calls, so the system runs and is testable without a key, and it is the fallback.

**Dictation extraction: LLM proposals, checked by a deterministic vocabulary.** A controlled dental vocabulary (FDI notation including spoken forms like "four eight", 23 finding types, negation and hedge detection) decides what counts as a finding. The model's proposals must pass a grounding check against the transcript, and parser findings the model missed are added. A draft report is where hallucination does the most harm, so the model is never trusted on its own there.

**Summaries: extractive, by design.** "Summarize her report" returns the report's own findings, impression and recommendation, each tagged with its finding ID. It reads less smoothly than a generated paragraph, but every line is traceable, which is what the brief asks for.

**STT: Whisper on Groq.** `whisper-large-v3-turbo` with a dental-terms prompt (OPG, IOPA, mesioangular, periapical radiolucency...), `language=en` and `verbose_json` so segment confidence is available. Browser speech recognition is the fallback when no server STT is configured, and the mock provider keeps the voice endpoint testable offline. Adding Deepgram, Azure or a self-hosted Whisper is one class in `app/stt.py`.

## How safety is enforced

1. **Identity is never guessed.** First name only with two matches → the assistant asks. A fuzzy match ("Pria Sharma") → asks for confirmation. Exact matches still carry a look-alike name alert (Priya Sharma / Priya Shetty, Ananya Rao / Anaya Rao) with patient ID and age as the second identifier.
2. **Pronouns must fit, and spoken names win.** "Summarize his report" while a female patient is open is questioned, not executed. If a patient's name was spoken, session context is never used in its place, and a name that matches nobody is never swapped for the open patient without a yes ("Does Kavya have caries?" → "Did you mean Priya Sharma (P001), who is open now?").
3. **Every LLM output is validated before it acts.** Tool arguments are schema-checked; an invalid call is re-interpreted by the rules instead of executed. The patient must come from what was said: a name or patient ID the model returns that isn't in the utterance is discarded, so a model slip can't open someone else's report.
4. **Every LLM finding is grounded.** It survives only if its quoted evidence is in the transcript, that evidence mentions the claimed finding, the tooth agrees with what was said, and negation matches the wording. Qualifiers that were never spoken, or that describe a different finding, are dropped. Rejections are returned to the clinician with reasons.
5. **Unrecognised speech is kept, not interpreted.** "Patient is anxious about the procedure" goes into `unparsed_segments` verbatim.
6. **Unspecified sides stay unspecified.** "Lower third molar is impacted" becomes "could be 38 / 48" and blocks approval until a tooth is picked.
7. **Silence is not a command.** Whisper's silence hallucinations are rejected before they reach the interpreter.
8. **Drafts need a human.** No impression or treatment advice is generated. Approval requires a linked patient and resolved teeth; approved drafts are frozen.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/assistant/message` | Main entry: `{text, session_id?, mode: auto\|command\|dictation}` |
| `POST` | `/api/assistant/voice` | Multipart `audio` (+ `session_id`, `mode`): STT then the same pipeline |
| `POST` | `/api/stt` | Transcribe only |
| `GET` | `/api/reports/{id}` | Fetch a report (clinic-scoped) |
| `GET` / `PUT` | `/api/drafts/{id}` | Read or edit a draft (edits are re-validated) |
| `POST` | `/api/drafts/{id}/approve` | Clinician sign-off |
| `GET` | `/api/config`, `/health` | Runtime info (active provider and models) |

Every assistant response has the same shape: `type` (report, clarification, not_found, summary, answer, draft_report, unsupported, cancelled), `message`, `data`, `evidence` (report and finding IDs, or transcript quotes), `warnings`, and a `trace` of each step with timings (plus model and token counts for LLM steps, and confidence data for STT). Interactive docs are at `/docs`.

`docs/DEMO.md` is a full sequence of calls with real outputs and curl commands, generated offline so anyone can reproduce it without a key. `docs/LIVE_RUN.md` is the live run against Groq.

## Mock data

Nine synthetic patients, 18 reports, two clinicians (`app/data/`). Every patient has two to four reports except Kiran Das, who has none on purpose. The edge cases are deliberate:

| Patient | Why it's there |
|---|---|
| Priya Sharma (P001), Priya Shetty (P002) | Same first name: ambiguity |
| Ananya Rao (P006), Anaya Rao (P007, age 8) | Sound-alike names, shared surname; P007 has primary-tooth (FDI 75) findings |
| Priya Shah (P008) | Belongs to another clinic (MYS-02): invisible to `dr_demo` |
| Kiran Das (P009) | No reports on file |
| Arjun Nair (P004) | Has OPG and ceph, no CBCT: missing-report path |
| Rahul Menon, Fatima Khan | Root stumps, implant, sinus and bone-loss findings for Q&A |

`samples/` holds silent placeholder WAVs for the mock STT (the file name maps to a transcript) and three real spoken clips made with Windows text-to-speech (`scripts/make_speech_samples.ps1`, which `run_check.bat` also runs).

## Production notes

**Privacy.** The LLM receives the utterance and a one-line context, never the patient list, report text or IDs. The audit log records clinician, action and record IDs, not transcript text or clinical content. Free tiers are fine for synthetic data only. With real patients I would sign a DPA with each provider (or self-host Whisper and an open-weight model in-region for DPDP Act compliance), use zero-retention settings, encrypt drafts at rest, and redact PHI from application logs.

**Authorization.** Every lookup is scoped to the clinician's clinics, and out-of-scope patients behave as if they don't exist (no "forbidden" that leaks existence). Sessions are bound to one clinician, and drafts are visible only to their author. The `X-Clinician-Id` header is a stand-in; production would validate an OIDC token and add role checks (e.g. only radiologists approve).

**Latency.** Measured live: Whisper 350–480 ms per short clip, interpreter about 1 s (590–1,610 ms), deterministic tools and validation about 1 ms (p50 1.0 ms, p95 2.1 ms over 400 requests). The three spoken commands in the live run took 1.05–1.76 s on the server from audio to answer, about 1.3 s on average; the browser upload comes on top. To go faster in production: streaming STT, rules first for unambiguous commands, and the LLM for dictation and anything the rules can't parse cleanly.

**Costs.** Free within Groq's free-tier limits. At Groq's published paid rates (mid-2026: $0.15 / $0.60 per million input / output tokens for `gpt-oss-120b`, $0.04 per audio hour for `whisper-large-v3-turbo` with a 10-second minimum per request), a spoken command costs about $0.0003 (about $0.00019 for the interpreter at ~860 input and ~100 output tokens, plus about $0.00011 for STT), or roughly $0.30 per 1,000 commands. The rule-based mode costs nothing.

## Known limitations

- The dictation vocabulary covers common radiographic findings, not all of dental radiology. Anything outside it lands in `unparsed_segments` for the clinician (safe, but more manual work).
- The offline interpreter is pattern-based. An unusual phrasing may get "which patient?" or "I can only answer from what the report documents" instead of an answer: it asks rather than guesses. The LLM interpreter handles free phrasing better.
- FDI numbering only. Palmer ("UR6") and spoken forms ("lower right eight", "four eight", "tooth 4-6") are handled; Universal numbering is not.
- A bare two-digit number in dictation is read as a tooth (e.g. "34" is tooth 34). Ages and measurements with units are excluded, but this is a known trade-off.
- Sessions and drafts are in memory and are lost on restart.
- Free-tier rate limits apply; when one is hit, that request falls back to the rule-based interpreter.
- English only. Clinicians in Bengaluru will code-switch; that needs an STT model and vocabulary tuned for it.
- No real PACS/EHR integration, image viewing, or authentication, as the brief asked.

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
tests/                  62 tests + 4 opt-in live API tests
scripts/run_demo.py     generates docs/DEMO.md (offline)
scripts/check_llm.py    live check of the key, model, latency, STT and real speech
start.bat               one-click start on Windows
run_check.bat           Windows: setup + tests + live check + live tests -> check_output.txt
docs/                   ARCHITECTURE.md, architecture.pdf, DEMO.md, LIVE_RUN.md
```
