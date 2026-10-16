# Evaluation

`eval/cases.jsonl` holds 134 labelled utterances, each with the outcome a careful assistant should produce. `scripts/run_eval.py` runs them through the real pipeline (interpreter, validation, tools) and scores the replies.

```bash
python scripts/run_eval.py                       # offline rules, a few seconds
python scripts/run_eval.py --interpreter groq    # live model; about 20 minutes on Groq's free tier
python scripts/run_eval.py --markdown --verbose  # tables like the ones below, every case listed
```

## What is in the set

| Group | Cases | What a correct reply looks like |
|---|---|---|
| Retrieval | 30 | Right patient and report, including dates, "previous", misspellings and lower-case speech-to-text output |
| Identity | 14 | Asks or stops when the patient is unclear (two Priyas, "Pria Sharma", another clinic's patient, a pronoun with nobody open) |
| Follow-ups | 32 | A patient is already open: questions, summaries, pronoun checks, clinical words the vocabulary doesn't know |
| Clarification answers | 6 | "Priya Sharma", "the second one", "P002", "yes", "cancel" after a "which patient?" question |
| Out of scope | 8 | Refuses booking, deleting, prescribing, small talk |
| Dictation | 44 | The findings that were said, with 56 labelled findings: tooth, present or absent, stated or uncertain |

For follow-ups the open patient is set directly in the session, so no extra model call is spent setting the scene. Dictated findings are matched on category and site; a finding that wasn't said counts against precision.

The number that must stay at zero is **a report shown for the wrong patient**. The script exits with an error if it is ever above zero, and CI runs it on every push.

## Results: offline rules

| Measure | Before the name change | Now |
|---|---|---|
| Cases fully correct | 123 / 134 | 127 / 134 |
| Right action (open, summarize, answer, ask, refuse) | 82 / 90 (91%) | 86 / 90 (96%) |
| Right patient, when one was named or open | 57 / 63 (90%) | 61 / 63 (97%) |
| Right report | 50 / 56 (89%) | 53 / 56 (95%) |
| Asked or stopped when the patient was unclear | 12 / 12 | 12 / 12 |
| Report shown for the wrong patient | 0 | 0 |
| Dictated findings: recall / precision / F1 | 0.98 / 0.96 / 0.97 | 0.98 / 0.96 / 0.97 |
| Said as absent and kept absent ("no ...") | 14 / 14 | 14 / 14 |
| Said as uncertain and kept uncertain ("possible ...") | 9 / 9 | 9 / 9 |

"Before" is the offline interpreter that found names by removing a long list of known words; "now" finds names by evidence (the clinic's patient list, or a name position such as "Priya's" or "does Kavya have"). The four cases it fixed were all follow-ups or lower-case transcripts where a word was misread as a name.

The seven cases the rules still get wrong:

| Case | What happened | Why |
|---|---|---|
| "I need the most recent scan we have on file for Fatima Khan" | Refused | The rules expect a command verb ("open", "show"); "I need" isn't one |
| "Show me the scan before the latest one for Priya Sharma" | Opened the latest | "before the latest one" isn't in the rules' phrasing for "previous" |
| "And the one before that?" | Refused | No verb and no report word, so the rules can't tell it's a report request |
| "What's the weather like?" | Asked which patient | Safe, but a plain refusal would be better |
| "Change her diagnosis to caries" | Made a draft with "caries" | Read as dictation; the draft still needs clinician review |
| "Retained deciduous 75 with the successor 35 unerupted." | Two extra findings | Both terms applied to both teeth |
| "Crown on 21 with no periapical changes." | Missed the crown | "Crown" alone usually means the anatomical crown, so the vocabulary needs "crown present" or a material |

These are phrasing gaps in the offline fallback, which the LLM is there to cover. None of them showed a report for the wrong patient.

## Results: live model

The same 134 cases through Groq `openai/gpt-oss-120b`, run on a Windows laptop with `--interpreter groq` on 16 October. Every reply still goes through the same validation and deterministic tools; only the interpreter changes.

| Measure | Offline rules | Live model |
|---|---|---|
| Cases fully correct | 127 / 134 | 120 / 134 |
| Right action (open, summarize, answer, ask, refuse) | 86 / 90 (96%) | 83 / 90 (92%) |
| Right patient, when one was named or open | 61 / 63 (97%) | 54 / 63 (86%) |
| Right report | 53 / 56 (95%) | 48 / 56 (86%) |
| Asked or stopped when the patient was unclear | 12 / 12 | 12 / 12 |
| Refused out-of-scope requests | 11 / 13 (85%) | 13 / 13 (100%) |
| Report shown for the wrong patient | 0 | 0 |
| Dictated findings: recall / precision / F1 | 0.98 / 0.96 / 0.97 | 0.96 / 0.95 / 0.96 |
| Model call latency, median / p90 | | 764 / 1204 ms |
| Tokens per call (mean) | | 972 |
| Calls that fell back to the rules | | 0 |

**What the first live run caught.** The first live run, on the evening of 15 October, had one reply that answered for the wrong patient: with Priya Sharma open, "Does Kavya have caries?" was answered from Priya's report. There is no Kavya in the clinic; the model left the name out of its tool call, and the check that restores a dropped name skipped names that match nobody, so the question fell through to the open patient. The spoken name now always wins, so the assistant says it has no patient called Kavya and asks; a regression test replays the model's output. The table above is the run after the fix.

**Where the two differ.** The model handles free phrasing the rules miss ("I need the most recent scan we have on file for Fatima Khan", "Change her diagnosis to caries" refused instead of drafted). It is also more cautious than it needs to be: it refused several follow-ups that the report can answer ("Does she need an extraction?", "What are the key findings?"). A refusal is safe but unhelpful; the fix would be a line in the system prompt saying that questions about what the report says are always in scope, with the rules as a second opinion.

The cases the live model got wrong:

| Case | Utterance | What happened |
|---|---|---|
| R18 | "What does Arjun Nair's OPG show?" | action: got unsupported, expected summary/report |
| F07 | "Show me her previous OPG" | report: got R-1003, expected R-1001 |
| F10 | "Does she have oral cancer?" | patient: got None, expected P001 |
| F12 | "Is she healthy?" | patient: got None, expected P001 |
| F13 | "Does she need an extraction?" | action: got unsupported, expected answer; patient: got None, expected P001; report: got None, expected R-1003 |
| F14 | "Is there anything concerning in her report?" | action: got unsupported, expected summary |
| F15 | "What are the key findings?" | action: got unsupported, expected summary |
| F20 | "Read me the impression" | action: got summary, expected answer |
| F23 | "Should we extract 48?" | action: got unsupported, expected answer; patient: got None, expected P001; report: got None, expected R-1003 |
| F24 | "And the one before that?" | action: got unsupported, expected report; report: got R-1003, expected ['R-1001', 'R-1002'] |
| D13 | "Retained deciduous 75 with the successor 35 unerupted." | added retained_deciduous @ 35 (not dictated); added unerupted @ 75 (not dictated) |
| D19 | "Crown on 21 with no periapical changes." | missed crown @ 21 |
| D34 | "Query root fracture of 21." | action: got clarification, expected draft_report |
| D39 | "Caries on 36 but no periapical lesion." | added periapical @ None (not dictated) |

The model is not fully repeatable at temperature 0.1: between the two live runs, three cases changed between right and wrong with no code change (R18, F07, F10). F16 changed because of the fix.

## Limits of this evaluation

- I wrote the cases myself, after building the system. They cover the phrasings I expected, so they measure regressions and the gap between the rules and the model, not accuracy on real clinicians' speech.
- Text only. Speech recognition is tested separately (`docs/LIVE_RUN.md`) with text-to-speech clips, not real voices.
- 134 cases is small; one case moves a percentage by about one point.
- The next step is a set recorded by clinicians (Indian-English accents, code-switching, clinic noise), with word error rate on dental terms.
