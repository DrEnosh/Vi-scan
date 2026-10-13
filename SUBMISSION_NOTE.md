# Submission note

**Candidate:** Dr. Enosh A Paulson
**Role:** AI Engineer, VI-SCAN Diagnostics
**Contents:** source code with mock data, README, one-page architecture (`docs/architecture.pdf`, also `docs/ARCHITECTURE.md`), documented API calls with real outputs (`docs/DEMO.md`, offline and reproducible), a live run against Groq (`docs/LIVE_RUN.md`), 57 automated tests plus 4 live API tests.

## What works

- **Report retrieval** from typed or spoken requests, with modality, latest/previous/earliest and spoken dates. When two patients share a name the assistant asks which one; a near-miss name ("Pria Sharma") needs confirmation; an exact match still shows a look-alike name alert with patient ID and age. Missing reports return what is on file instead of an error.
- **Dictation to draft report.** Spoken findings become structured JSON with FDI tooth numbers, qualifiers, present/absent, stated/uncertain, the exact words each finding came from, and review flags. Negation ("no periapical radiolucency") and hedging ("possible caries") are kept. Speech outside the vocabulary is kept verbatim rather than interpreted, an unspecified side ("lower third molar") is not guessed, and no impression or advice is generated. Clinicians can edit the JSON (edits are re-validated) and approve.
- **Follow-up.** "Summarize her report" uses the patient already open, checks that the pronoun fits, and returns an extractive summary in which every line cites its finding ID. Questions such as "Does she have any caries?" are answered only from the selected report; if it isn't documented, the answer says so and points to the reports that do mention it. This also covers the report-aware chatbot bonus.
- **Tool calling with a clear boundary.** The LLM (Groq `openai/gpt-oss-120b`, or any OpenAI-compatible API by config) picks one of three tools; Pydantic validation and deterministic tools decide identity, scope and evidence. LLM-proposed findings must pass a grounding check against the transcript, and rejected ones are shown with reasons. A rule-based interpreter emits the same tool calls, so everything runs offline and the LLM has a fallback.
- **Voice.** The mic in the web UI records audio and sends it to Whisper (`whisper-large-v3-turbo` on Groq) with a dental-terms prompt and a guard against Whisper's silence hallucinations.
- **Verified live.** On Groq's free tier: correct tool choice on every test utterance, about 1 s per interpreter call and 350–480 ms per transcription, and all 4 live API tests passing.
- **Production basics.** Input limits, clean error responses, clinic-scoped authorization, sessions bound to a clinician, an ID-only audit log, per-step latency in every response, and a cost note.

## What live testing changed

Running against the real APIs exposed five problems that the scripted tests had not: Groq rejecting `null` tool arguments, Whisper transcribing silence as "Thank you.", the free-tier tokens-per-minute limit, the model missing one dictated finding, and the model once dropping a spoken patient name. Each now has a fix and a regression test; details are in the README under "What live testing caught".

## What is incomplete

- The dental vocabulary covers common radiographic findings, not the whole field. Uncovered phrases are safe (kept verbatim) but not structured.
- Sessions and drafts are in memory. Auth is a header stand-in for real tokens.
- Live testing used clean synthetic speech (Windows text-to-speech). Real clinic audio with accents and background noise still needs evaluating.

## What I would improve with more time

1. Build an evaluation set of real dictations (Indian-English accents, code-switching, clinic noise) and track extraction precision/recall, rejection rates and STT word error rate on dental terms.
2. Streaming STT with a custom dental vocabulary, and rules-first routing so the LLM is only called for dictation or when rules can't parse cleanly, to cut latency and cost.
3. Let the LLM write a fluent summary, but keep the current check: every sentence must map to a cited finding or it is dropped.
4. Persist drafts with version history and add role-based sign-off (for example, only a radiologist can approve).
5. Expand the vocabulary with a clinician (bony vs soft-tissue impaction classes, Pell and Gregory, periodontal staging) and add Universal notation.
