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

The same set can be run against the live model with `--interpreter groq`. Earlier live runs on 13 and 15 October (`docs/LIVE_RUN.md`) got 9 of 9 tool choices right, at about 1 s per call.

## Limits of this evaluation

- I wrote the cases myself, after building the system. They cover the phrasings I expected, so they measure regressions and the gap between the rules and the model, not accuracy on real clinicians' speech.
- Text only. Speech recognition is tested separately (`docs/LIVE_RUN.md`) with text-to-speech clips, not real voices.
- 134 cases is small; one case moves a percentage by about one point.
- The next step is a set recorded by clinicians (Indian-English accents, code-switching, clinic noise), with word error rate on dental terms.
