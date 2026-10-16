# Changelog

What changed and why, day by day. The assignment arrived on 13 October 2026.

## 16 October, morning

- **Live evaluation.** Ran the 134 cases against Groq `gpt-oss-120b`. The first run (the evening before) had one reply that answered for the wrong patient: "Does Kavya have caries?" with Priya open, after the model dropped a name that matches nobody in the clinic. The spoken name now always wins, with a regression test. The re-run: 120 of 134 fully correct and 0 wrong-patient replies. Live results are in `docs/EVAL.md`.

## 15 October, evening

Reviewing the repository as an outsider would showed four weak spots: no way to measure accuracy, an offline name finder built on a long list of words to ignore, lint noise, and a README that was hard to scan.

- **Evaluation set.** `eval/cases.jsonl` holds 134 labelled utterances (retrieval, identity, follow-ups, clarification answers, out-of-scope requests, and 44 dictations with 56 findings). `scripts/run_eval.py` scores the right action, patient, report and findings, and fails if any reply shows a report for the wrong patient. Results and every failed case are in `docs/EVAL.md`.
- **Offline name finding rewritten.** A word now counts as a patient name only with evidence: it matches the clinic's patient list (exact or a close spelling), or it sits where a name sits ("Priya's", "patient Rahul", "does Kavya have", "open Arjun"). The 524-word ignore list is gone; the only fixed lists are grammar words. On the evaluation set the offline interpreter went from 123 to 127 of 134 cases fully correct, and follow-ups like "What happened to 46?" or "Is malignancy ruled out?" no longer come back as "no patient named ...".
- **Code quality.** Ruff configuration with all findings fixed. `create_app` and `ground_llm_findings` were split into small functions (a differential check over 3,000 generated model outputs gave identical results before and after). Dependencies are pinned to the versions tested on Python 3.11 to 3.14. GitHub Actions runs lint, tests and the evaluation on every push.
- **Docs.** Shorter README with a demo GIF and screenshots; configuration, API and production notes moved to `docs/REFERENCE.md`; this changelog.

## 15 October, morning

An adversarial pass with a scripted model, trying to make it act on the wrong patient or invent findings.

- The patient must come from what was said: a name or patient ID returned by the model that isn't in the utterance is discarded.
- Duplicate findings from the model are merged, keeping the more cautious reading, and a vaguer finding for the same tooth is dropped.
- Re-ran the live checks on the final code (`docs/LIVE_RUN.md`); architecture document v1.2.

## 14 October

A check from a clean clone, following the README exactly.

- One test needed the optional provider package; tests no longer depend on it, and a missing package or key no longer stops the server from starting.
- The offline interpreter read some question words as patient names ("Does she have a tumour?" came back as "no patient named Tumour"). Fixed for those cases, with tests; the evening of 15 October replaced the approach.
- Treatment questions ("Does she need an extraction?") quote the report's own recommendation.
- Windows line endings kept for `.bat` files, so the scripts run from a downloaded ZIP.

## 13 October

First version: retrieval with clarification, follow-ups with evidence, dictation to a structured draft, tool calling with a rule-based twin, Whisper speech input, tests, demo log and architecture document.

Live testing against Groq the same day caught five problems that the scripted tests had not: tool calls with `null` arguments rejected by the API, Whisper transcribing silence as "Thank you.", the free-tier tokens-per-minute limit, a dictated finding the model missed, and a spoken patient name the model dropped. Each has a fix and a regression test (details in `docs/REFERENCE.md`).
