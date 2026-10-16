# Architecture (one page)

```mermaid
flowchart LR
  subgraph Input
    AUD[Mic recording / audio upload] --> STT[Whisper on Groq<br/>+ no-speech guard] -->|transcript| MSG
    MIC[Browser speech<br/>fallback] -->|text| MSG
    TYP[Typed text] --> MSG
  end
  MSG[/POST /api/assistant/message/] --> ORCH[Orchestrator<br/>session context]
  ORCH --> INT{Interpreter}
  INT -->|tool call| LLM[LLM tool calling<br/>Groq gpt-oss-120b or<br/>any OpenAI-compatible API]
  INT -->|fallback / offline| RB[Rule-based twin]
  LLM --> VAL[Validation<br/>schema, name cross-check,<br/>business rules]
  RB --> VAL
  VAL --> T1[find_patient<br/>exact / ambiguous / fuzzy]
  VAL --> T2[select_report]
  VAL --> T3[summarize / answer<br/>extractive, cited]
  VAL --> T4[draft_findings<br/>grounding + parser merge]
  T1 & T2 --> REPO[(Clinic-scoped<br/>mock data)]
  T4 --> DRAFT[(Draft store)<br/>edit / approve]
  T1 & T2 & T3 & T4 --> RESP[Response<br/>message, data, evidence, warnings, trace]
```

## Flow

1. **Voice to text.** The UI mic records audio and posts it to `POST /api/assistant/voice`, which transcribes it with Whisper (`whisper-large-v3-turbo` on Groq, dental-terms prompt, `verbose_json`). Transcripts that are only a known Whisper silence phrase ("Thank you.") are rejected, and low-confidence transcripts carry a warning. Without a server STT provider the UI falls back to browser speech recognition, and tests use a mock. Typed input skips this step; everything after it is identical.
2. **Interpretation (LLM side).** The utterance goes to the LLM (Groq `openai/gpt-oss-120b` by default, any OpenAI-compatible API by config) with three tools: `report_request` (action open / summarize / question), `draft_findings` and `unsupported_request`. Tool choice is forced, so it must answer with exactly one call. The system prompt and tools are static (so the provider can cache them); the user message carries a one-line context (is a patient open, and their sex for pronoun checks) and the utterance. The model never sees names, IDs or report text. With no key, or on timeout, rate limit or a bad call, the rule-based interpreter emits the same internal tool calls; it recognises a patient name from the clinic's patient list or from its position in the sentence.
3. **Validation (business-rule side).** Arguments are validated with Pydantic (enums, ID formats, dates; `null` treated as absent). A bad LLM call is not executed: it is re-interpreted by the rules. If a patient name was spoken but the model returned none, the spoken name is used; a name or patient ID the model returns that was never said is discarded. Then the deterministic tools apply the policies the LLM cannot override:
   - **Patient identity**: exact match opens; several matches or a fuzzy match asks the clinician; look-alike names get an alert with patient ID and age as a second identifier. Only patients in the clinician's clinics are searchable.
   - **Context**: "her/his/their" resolves to the open patient only if the pronoun fits; otherwise the assistant asks. A spoken name always beats context, and a name that matches nobody is never swapped for the open patient without a yes. A pending "which patient?" question carries the original intent, so "Priya Sharma" completes "open her latest OPG".
   - **Reports**: modality, latest/previous/earliest and spoken dates select a report; a missing one returns what *is* on file.
4. **Report generation.** Dictation becomes findings `{category, site (FDI), qualifiers, present, certainty, evidence, review_reasons}`. Every model-proposed finding must pass the grounding check: quote present in the transcript, quote mentions the finding, tooth consistent with the words, negation consistent with the words, qualifiers actually spoken for that finding. Failures are returned as `rejected_findings` with reasons. Findings from the deterministic parser that the model missed are merged in. Unmapped speech is kept verbatim. The readable draft restates findings only; no impression or advice is generated. Status stays `draft_pending_review` until a clinician edits (re-validated) and approves (requires a linked patient and resolved teeth).
5. **Answers with evidence.** Summaries and Q&A are extractive: each line carries `report_id#finding_id`. If the report doesn't mention something, the answer says so and names other reports that do.

## Production considerations

| Concern | In this PoC | Next step for production |
|---|---|---|
| Privacy | LLM never sees records or IDs; audit log holds IDs only; free tiers used with synthetic data only | In-region or self-hosted STT/LLM, zero-retention API terms, encrypted draft storage, PHI-scrubbed logs (DPDP Act) |
| Authorization | Clinic-scoped reads, sessions bound to a clinician, drafts private to author; header-based mock identity | OIDC tokens, role-based approval (e.g. radiologist sign-off), break-glass logging |
| Latency | Measured: STT 350–480 ms, interpreter about 1 s, tools about 1 ms; LLM capped by an 8 s timeout with instant rule fallback | Streaming STT, rules-first routing so the LLM handles dictation and unclear commands only |
| Cost | Groq free tier: $0 within rate limits. At Groq's paid rates about $0.0003 per spoken command (interpreter plus STT) | Prompt caching, self-hosted Whisper and open-weight model at volume |
| Failure handling | LLM error or rate limit → rule fallback; STT error → clear 4xx; input limits; no stack traces to clients | Retry budgets, circuit breaker, monitoring on rejection and clarification rates |
