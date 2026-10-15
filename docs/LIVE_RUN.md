# Live run against Groq

Output of `run_check.bat` on Windows (Python 3.14) with `INTERPRETER=groq` and `STT_PROVIDER=groq`, 13 October 2026. Model `openai/gpt-oss-120b` for interpretation and `whisper-large-v3-turbo` for speech, both on Groq's free tier. All patient data is synthetic.

**Re-run on 15 October 2026** on the final code (commit `68e754c`): same outcome. 9 of 9 tool calls correct, the silent clip rejected, 4 of 4 live API tests passed, and all 62 offline tests passing on Windows. Model calls took 607–1,255 ms and Whisper 317–386 ms. The log below is the original 13 October run.

What it shows:

- **[1/3]** one direct tool call to confirm the key, model and tool calling.
- **[2/3]** a conversation through the full app (LLM → validation → tools). "Priya Sharma" answers a pending clarification in code, so it needs no LLM call.
- **[3/3]** a 0.5-second silent clip. Whisper returns "Thank you." and the no-speech guard rejects it.
- **[extra]** three spoken clips (made with Windows text-to-speech by `scripts/make_speech_samples.ps1`) sent through `POST /api/assistant/voice`: Whisper → LLM → tools.
- The 4 live API tests in `tests/test_live_llm.py`.

Calls are spaced about 8 seconds apart to stay within the free tier's tokens-per-minute limit. Cost shows $0 because the free tier is not billed. Draft IDs are random per run.

```text
=== Live check ===
Provider: groq  model: openai/gpt-oss-120b  key loaded  timeout: 8.0s

[1/3] Pre-flight tool call ...
  OK -> open_report {'date': None, 'modality': 'OPG', 'patient_id': None, 'patient_name': 'Priya Sharma', 'patient_reference': 'named', 'which': 'latest'}
  usage={'model': 'openai/gpt-oss-120b', 'input_tokens': 855, 'cached_tokens': 0, 'output_tokens': 111, 'est_cost_usd': 0.0}

[2/3] Conversation through the full app ...

> Open Priya's latest OPG report
  groq -> open_report {'date': None, 'modality': 'OPG', 'patient_id': None, 'patient_name': 'Priya', 'patient_reference': 'named', 'which': 'latest'}
           717.49 ms, 854 in (0 cached) + 105 out tokens, ~$0.0
  => clarification: I found 2 patients matching "Priya". Which one do you mean? You can say the full name or the patient ID.

> Priya Sharma
  (answered the pending question: resolved, no LLM call needed)
  => report: Opened Priya Sharma (P001) - OPG dated 2026-08-21 (R-1003).

> Summarize her report
  groq -> summarize_report {'date': None, 'modality': 'ANY', 'patient_id': None, 'patient_name': None, 'patient_reference': 'pronoun_female', 'which': None}
           742.83 ms, 857 in (768 cached) + 98 out tokens, ~$0.0
  => summary: Priya Sharma (P001) - OPG dated 2026-08-21 (R-1003)

> Does she have any caries?
  groq -> answer_report_question {'date': None, 'modality': None, 'patient_id': None, 'patient_name': None, 'patient_reference': 'pronoun_female', 'question': 'Does she have any caries?', 'which': None}
           1612.55 ms, 859 in (0 cached) + 136 out tokens, ~$0.0
  => answer: R-1003 (OPG, 2026-08-21) does not mention caries. I can't confirm or rule it out from this report.

> Can you get me the panoramic x-ray we took for Rahul Menon?
  groq -> open_report {'date': None, 'modality': 'OPG', 'patient_id': None, 'patient_name': 'Rahul Menon', 'patient_reference': 'named', 'which': None}
           587.48 ms, 867 in (0 cached) + 104 out tokens, ~$0.0
  => report: Opened Rahul Menon (P003) - OPG dated 2025-07-09 (R-3001).

> Lower right third molar is horizontally impacted with roots close to the mandibular canal. Possible caries on 46. No periapical radiolucency in relation to 36. Patient is anxious about the procedure.
  groq -> draft_findings (findings below)
           1294.53 ms, 897 in (0 cached) + 337 out tokens, ~$0.0
  => draft_report: Draft DR-78C1F180 created with 4 finding(s), 1 flagged for review. Clinician review is required before use.
     engine=llm+grounding+rules added_from_rules=1
     + Tooth 48 (lower right third molar): Impaction (horizontal).
     + Tooth 46 (lower right first molar): Caries. [uncertain as dictated: "Possible"]  [review: Dictated with uncertainty ("Possible")]
     + Tooth 36 (lower left first molar): No periapical radiolucency.
     + Tooth 48 (lower right third molar): Root proximity to mandibular canal.
     ~ kept verbatim: "Patient is anxious about the procedure"

> Book me a CBCT slot for tomorrow
  groq -> unsupported_request {'reason': 'Scheduling request not supported'}
           1086.51 ms, 860 in (0 cached) + 50 out tokens, ~$0.0
  => unsupported: I can open reports, summarize them, answer questions about them, or turn dictation into a draft report. (Scheduling request not supported) Try: "Open Priya Shar

[3/3] Speech-to-text ...
  OK (groq whisper-large-v3-turbo) -> key accepted; silent clip correctly rejected as no speech  {'model': 'whisper-large-v3-turbo', 'segments': 1, 'dropped_no_speech': 0, 'max_no_speech_prob': 0.0, 'min_avg_logprob': -0.968, 'rejected_text': 'Thank you.'}
  Use the mic button in the web UI to try real speech.

[extra] Real speech through the whole pipeline (POST /api/assistant/voice) ...

  speech_open_priya.wav
    heard (groq, 387.57 ms): 'Can you pull up the latest OPG for Priya Sharma?'
    whisper segments=1 max_no_speech_prob=0.0 min_avg_logprob=-0.117
    groq -> open_report (667.48 ms, 0/859 input tokens cached)
    => report: Opened Priya Sharma (P001) - OPG dated 2026-08-21 (R-1003).

  speech_summarize.wav
    heard (groq, 353.33 ms): 'Summarize her report.'
    whisper segments=1 max_no_speech_prob=0.0 min_avg_logprob=-0.396
    groq -> summarize_report (696.73 ms, 0/858 input tokens cached)
    => summary: Priya Sharma (P001) - OPG dated 2026-08-21 (R-1003)

  speech_dictation.wav
    heard (groq, 480.01 ms): 'Lower right third molar is horizontally impacted. Possible caries on tooth 4-6. No periapical radiolucency in relation to tooth 3-6.'
    whisper segments=1 max_no_speech_prob=0.0 min_avg_logprob=-0.049
    groq -> draft_findings (1283.74 ms, 768/888 input tokens cached)
    => draft_report: Draft DR-01136ECF created with 3 finding(s), 1 flagged for review. Clinician review is required before use.
       + Tooth 48 (lower right third molar): Impaction (horizontal).
       + Tooth 46 (lower right first molar): Caries. [uncertain as dictated: "Possible"]
       + Tooth 36 (lower left first molar): No periapical radiolucency.

------------------------------------------------------------
LLM calls: 6, avg 1007 ms, max 1613 ms, est. cost $0.00000
Integration working.

=== Live API tests (pytest -m live) ===
....                                                                     [100%]
```
