# Submission note

**Candidate:** Dr. Enosh A Paulson
**Role:** AI Engineer, VI-SCAN Diagnostics
**Contents:** source code with mock data, README, a one-page architecture document (`docs/architecture.pdf`, also as `docs/ARCHITECTURE.md`, with a two-page appendix in `docs/architecture-appendix.pdf`), a 3-minute demo video (`docs/demo.mp4`), documented API calls with real outputs (`docs/DEMO.md`, offline and reproducible), a live run against Groq (`docs/LIVE_RUN.md`), 90 automated tests plus 4 live API tests, a 134-case evaluation set with results (`docs/EVAL.md`), and CI on GitHub Actions.

## What works

- **Report retrieval** from typed or spoken requests, with modality, latest/previous/earliest and spoken dates. When two patients share a name the assistant asks which one; a near-miss name ("Pria Sharma") needs confirmation; an exact match still shows a look-alike name alert with patient ID and age. Missing reports return what is on file instead of an error.
- **Dictation to draft report.** Spoken findings become structured JSON with FDI tooth numbers, qualifiers, present/absent, stated/uncertain, the exact words each finding came from, and review flags. Negation ("no periapical radiolucency") and hedging ("possible caries") are kept. Speech outside the vocabulary is kept verbatim rather than interpreted, an unspecified side ("lower third molar") is not guessed, and no impression or advice is generated. Clinicians can edit the JSON (edits are re-validated) and approve.
- **Follow-up.** "Summarize her report" uses the patient already open, checks that the pronoun fits, and returns an extractive summary in which every line cites its finding ID. Questions such as "Does she have any caries?" are answered only from the selected report; if it isn't documented, the answer says so and points to the reports that do mention it. Something the report doesn't cover ("Does she have a tumour?") is neither confirmed nor denied, and a treatment question gets the report's own recommendation, quoted. This also covers the report-aware chatbot bonus.
- **Tool calling with a clear boundary.** The LLM (Groq `openai/gpt-oss-120b`, or any OpenAI-compatible API by config) picks one of three tools; Pydantic validation and deterministic tools decide identity, scope and evidence. LLM-proposed findings must pass a grounding check against the transcript, and rejected ones are shown with reasons. A rule-based interpreter emits the same tool calls, so everything runs offline and the LLM has a fallback.
- **Voice.** The mic in the web UI records audio and sends it to Whisper (`whisper-large-v3-turbo` on Groq) with a dental-terms prompt and a guard against Whisper's silence hallucinations.
- **Verified live.** On Groq's free tier: correct tool choice on every test utterance, about 1 s per interpreter call and 350–480 ms per transcription, and all 4 live API tests passing.
- **Measured.** On a labelled set of 134 utterances the offline interpreter takes the right action in 96% of conversation turns and finds dictated findings with F1 0.97; the live model takes the right action in 92% with findings F1 0.96. The first live run found one reply that answered for the wrong patient; it is fixed, with a regression test, and the re-run had none. Every failed case is listed in `docs/EVAL.md`.
- **Production basics.** Input limits, clean error responses, clinic-scoped authorization, sessions bound to a clinician, an ID-only audit log, per-step latency in every response, and a cost note.

## What testing changed

Running against the real APIs exposed five problems that the scripted tests had not: Groq rejecting `null` tool arguments, Whisper transcribing silence as "Thank you.", the free-tier tokens-per-minute limit, the model missing one dictated finding, and the model once dropping a spoken patient name. Each now has a fix and a regression test; details are in `docs/REFERENCE.md` under "What live testing caught".

A final check from a clean clone found two more. Following the Quick start exactly, one test needed the optional provider package, and the offline interpreter read some question words as patient names ("Does she have a tumour?" came back as "no patient named Tumour"). Both are fixed, with regression tests.

An adversarial pass with a scripted model then showed that a patient name or ID returned by the model was trusted even when nobody had said it. The patient now has to come from the utterance, and duplicate model findings are merged. Both have regression tests.

Finally, I reviewed the repository as an outsider would. That led to the evaluation set, a rewrite of how the offline interpreter finds patient names (by evidence from the patient list and sentence position, instead of a long list of words to ignore), lint and CI, pinned dependencies, and a shorter README. `CHANGELOG.md` has the day-by-day record.

## What is incomplete

- The dental vocabulary covers common radiographic findings, not the whole field. Uncovered phrases are safe (kept verbatim) but not structured.
- The offline interpreter is pattern-based, so an unusual phrasing gets a clarifying question rather than an answer (seven of the 134 evaluation cases still fail this way). The LLM interpreter handles free phrasing better.
- Sessions and drafts are in memory. Auth is a header stand-in for real tokens.
- Live testing used clean synthetic speech (Windows text-to-speech). Real clinic audio with accents and background noise still needs evaluating.

## What I would improve with more time

1. Extend the evaluation set with real dictations recorded by clinicians (Indian-English accents, code-switching, clinic noise) and track STT word error rate on dental terms alongside the current measures.
2. Streaming STT with a custom dental vocabulary, and rules-first routing so the LLM is only called for dictation or when rules can't parse cleanly, to cut latency and cost.
3. Let the LLM write a fluent summary, but keep the current check: every sentence must map to a cited finding or it is dropped.
4. Persist drafts with version history and add role-based sign-off (for example, only a radiologist can approve).
5. Expand the vocabulary with a clinician (bony vs soft-tissue impaction classes, Pell and Gregory, periodontal staging) and add Universal notation.
