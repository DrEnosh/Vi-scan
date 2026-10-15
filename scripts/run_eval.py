"""Scores the assistant on the labelled utterances in eval/cases.jsonl.

    python scripts/run_eval.py                       # offline, rule-based interpreter
    python scripts/run_eval.py --interpreter groq    # live LLM, key from .env (about 20 min on the free tier)
    python scripts/run_eval.py --markdown            # tables for docs/EVAL.md
    python scripts/run_eval.py --only F --verbose    # one group (id prefix), with every case printed

Each case is one utterance with the expected outcome: the action (open, summary, answer, ask,
refuse, draft), the patient and the report, or for dictation the findings that were said.
Follow-ups start with a patient already open (set directly in the session, so no extra model
call is spent on setting the scene). Exit code 1 if any reply disclosed a patient's report when
it should not have: that is the one result that must stay at zero.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from app.config import DATA_DIR, load_settings  # noqa: E402
from app.interpreter.factory import build_interpreter  # noqa: E402
from app.models import AssistantResponse  # noqa: E402
from app.orchestrator import Assistant  # noqa: E402
from app.repository import Repository  # noqa: E402
from app.session import DraftStore, SessionStore  # noqa: E402

CASES = ROOT / "eval" / "cases.jsonl"
DISCLOSES_REPORT = {"report", "summary", "answer"}
NAMES = {p["patient_id"]: p["name"] for p in json.loads((DATA_DIR / "patients.json").read_text(encoding="utf-8"))}


class PacedInterpreter:
    """Spaces out LLM calls and waits out rate limits, so a busy free tier is not scored as a model error.
    Times only the successful call, so pacing and waiting never count as latency."""

    def __init__(self, inner: Any, pace_s: float):
        self.inner, self.pace_s, self.name = inner, pace_s, inner.name
        self.last_ms: float | None = None
        self._last_start = 0.0

    def interpret(self, text: str, ctx: Any, mode: str = "auto") -> Any:
        for attempt in range(6):
            wait = self.pace_s - (time.monotonic() - self._last_start)
            if wait > 0:
                time.sleep(wait)
            self._last_start = time.monotonic()
            try:
                call = self.inner.interpret(text, ctx, mode)
                self.last_ms = (time.monotonic() - self._last_start) * 1000
                return call
            except Exception as e:
                if type(e).__name__ != "RateLimitError" or attempt == 5:
                    raise
                time.sleep(_retry_after(e))
        raise RuntimeError("unreachable")


def _retry_after(e: Exception) -> float:
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        return min(float(headers.get("retry-after", 20)), 90.0)
    except ValueError:
        return 20.0


# ----------------------------------------------------------------------------- one case
@dataclass
class Result:
    case: dict[str, Any]
    resp: AssistantResponse
    llm_ms: float | None = None
    fell_back: bool = False
    tokens: int | None = None
    problems: list[str] = field(default_factory=list)
    finding_counts: Counter[str] = field(default_factory=Counter)

    @property
    def passed(self) -> bool:
        return not self.problems


def _patient(resp: AssistantResponse) -> str | None:
    p = (resp.data or {}).get("patient")
    return p.get("patient_id") if isinstance(p, dict) else None


def _report(resp: AssistantResponse) -> str | None:
    d = resp.data or {}
    return d["report"].get("report_id") if isinstance(d.get("report"), dict) else d.get("report_id")


def _site(site: dict[str, Any] | None) -> str | None:
    if not site:
        return None
    if site.get("type") == "tooth":
        return str(site.get("fdi"))
    if site.get("type") in ("unresolved", "generalized"):
        return str(site["type"])
    return str(site.get("label") or "").lower() or None


def _same_site(gold: str | None, pred: str | None) -> bool:
    if gold in (None, "generalized") and pred in (None, "generalized"):
        return True
    if gold is None or pred is None:
        return False
    if gold == pred:
        return True
    g = set(gold.split()) - {"region", "the"}
    p = set(pred.split()) - {"region", "the"}
    return not gold[0].isdigit() and not pred[0].isdigit() and (g <= p or p <= g)


def score_findings(gold: list[dict[str, Any]], findings: list[dict[str, Any]]) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    problems: list[str] = []
    pred = [{"category": f["category"], "site": _site(f.get("site")), "present": f["present"],
             "certainty": f["certainty"]} for f in findings]
    unmatched = list(range(len(pred)))
    for g in gold:
        hit = next((i for i in unmatched if pred[i]["category"] == g["category"]
                    and _same_site(g["site"], pred[i]["site"])), None)
        if hit is None:
            counts["fn"] += 1
            problems.append(f"missed {g['category']} @ {g['site']}")
            continue
        unmatched.remove(hit)
        counts["tp"] += 1
        p = pred[hit]
        if not g["present"]:
            counts["negated"] += 1
            counts["negated_kept"] += not p["present"]
        if g["certainty"] == "uncertain":
            counts["hedged"] += 1
            counts["hedged_kept"] += p["certainty"] == "uncertain"
        if p["present"] != g["present"]:
            counts["flipped"] += 1
            problems.append(f"{g['category']} @ {g['site']}: present={p['present']}, said {g['present']}")
        if p["certainty"] != g["certainty"]:
            counts["flipped"] += g["certainty"] == "stated"  # a stated finding turned uncertain
            problems.append(f"{g['category']} @ {g['site']}: {p['certainty']}, said {g['certainty']}")
    for i in unmatched:
        counts["fp"] += 1
        problems.append(f"added {pred[i]['category']} @ {pred[i]['site']} (not dictated)")
    return counts, problems


def check(case: dict[str, Any], resp: AssistantResponse) -> tuple[list[str], Counter[str]]:
    exp = case["expect"]
    problems: list[str] = []
    want_types = exp["type"] if isinstance(exp["type"], list) else [exp["type"]]
    if resp.type not in want_types:
        problems.append(f"action: got {resp.type}, expected {'/'.join(want_types)}")
    if resp.type == "draft_report":
        counts, fp = score_findings(exp.get("findings", []), (resp.data or {}).get("findings", []))
        return problems + fp, counts
    pid, rid = _patient(resp), _report(resp)
    if exp.get("patient") and pid != exp["patient"]:
        problems.append(f"patient: got {pid}, expected {exp['patient']}")
    reports = exp.get("report")
    if reports and rid not in (reports if isinstance(reports, list) else [reports]):
        problems.append(f"report: got {rid}, expected {reports}")
    cands = [c.get("patient_id") for c in (resp.data or {}).get("candidates", [])]
    if exp.get("candidates") and not set(exp["candidates"]) <= set(cands):
        problems.append(f"options offered: {cands}, expected to include {exp['candidates']}")
    shown = json.dumps([resp.data, [e.model_dump(mode="json") for e in resp.evidence or []]], default=str)
    leaked = [p for p in exp.get("not_patient", []) if p in shown
              or (NAMES[p].lower() in resp.message.lower() and NAMES[p].lower() not in case["text"].lower())]
    if leaked:
        problems.append(f"showed {leaked}, a patient this clinician must not see")
    return problems, Counter()


def run_case(assistant: Assistant, interp: Any, case: dict[str, Any], clinician: Any) -> Result:
    sid = f"eval-{case['id']}"
    session = assistant.sessions.get_or_create(sid, clinician.clinician_id)
    if ctx := case.get("context"):
        session.active_patient_id, session.active_report_id = ctx["patient"], ctx.get("report")
    for text in case.get("setup", []):
        assistant.handle(text, clinician, session.session_id)
    if isinstance(interp, PacedInterpreter):
        interp.last_ms = None
    resp = assistant.handle(case["text"], clinician, session.session_id, case.get("mode", "auto"))
    res = Result(case, resp)
    for st in resp.trace:
        if st.step == "interpret_fallback":
            res.fell_back = True
        if st.step == "interpret" and (usage := st.detail.get("usage")):
            res.tokens = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    if isinstance(interp, PacedInterpreter) and not res.fell_back:
        res.llm_ms = interp.last_ms
    res.problems, res.finding_counts = check(case, resp)
    return res


# ----------------------------------------------------------------------------- summary
def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def summarize(results: list[Result]) -> dict[str, Any]:
    conv = [r for r in results if r.case["group"] != "dictation"]
    dic = [r for r in results if r.case["group"] == "dictation"]

    def ok(r: Result, prefix: str) -> bool:
        return not any(p.startswith(prefix) for p in r.problems)

    with_patient = [r for r in conv if r.case["expect"].get("patient")]
    with_report = [r for r in conv if r.case["expect"].get("report")]
    must_ask = [r for r in conv if r.case["expect"]["type"] in ("clarification", ["clarification", "not_found"])]
    must_refuse = [r for r in conv if r.case["expect"]["type"] == "unsupported"]
    wrong_disclosure = [r for r in conv if r.resp.type in DISCLOSES_REPORT
                        and _patient(r.resp) != r.case["expect"].get("patient")]
    fc: Counter[str] = sum((r.finding_counts for r in dic), Counter())
    prec = fc["tp"] / (fc["tp"] + fc["fp"]) if fc["tp"] + fc["fp"] else 0.0
    rec = fc["tp"] / (fc["tp"] + fc["fn"]) if fc["tp"] + fc["fn"] else 0.0
    by_group: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_group[r.case["group"]].append(r)
    lat = [r.llm_ms for r in results if r.llm_ms]
    toks = [r.tokens for r in results if r.tokens]
    return {
        "cases": len(results), "conversation": len(conv), "dictation": len(dic),
        "action": (sum(ok(r, "action") for r in conv), len(conv)),
        "patient": (sum(ok(r, "patient") and ok(r, "action") for r in with_patient), len(with_patient)),
        "report": (sum(ok(r, "report") and ok(r, "action") for r in with_report), len(with_report)),
        "asked": (sum(r.resp.type in ("clarification", "not_found") for r in must_ask), len(must_ask)),
        "refused": (sum(r.resp.type == "unsupported" for r in must_refuse), len(must_refuse)),
        "wrong_disclosure": [r.case["id"] for r in wrong_disclosure],
        "findings": {"gold": fc["tp"] + fc["fn"], "tp": fc["tp"], "fp": fc["fp"], "fn": fc["fn"],
                     "precision": round(prec, 3), "recall": round(rec, 3),
                     "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
                     "negated": (fc["negated_kept"], fc["negated"]), "hedged": (fc["hedged_kept"], fc["hedged"]),
                     "changed": fc["flipped"]},
        "groups": {g: (sum(r.passed for r in rs), len(rs)) for g, rs in by_group.items()},
        "passed": (sum(r.passed for r in results), len(results)),
        "fallbacks": sum(r.fell_back for r in results),
        "latency_ms": {"median": round(statistics.median(lat)), "p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))]),
                       "max": round(max(lat))} if lat else None,
        "tokens_per_call": round(statistics.mean(toks)) if toks else None,
    }


def print_report(s: dict[str, Any], results: list[Result], engine: str, markdown: bool) -> None:
    f = s["findings"]
    rows = [
        ("Right action (open / summarize / answer / ask / refuse)", pct(*s["action"])),
        ("Right patient, when one was named or open", pct(*s["patient"])),
        ("Right report", pct(*s["report"])),
        ("Asked or stopped when the patient was unclear", pct(*s["asked"])),
        ("Refused out-of-scope requests", pct(*s["refused"])),
        ("Report shown for the wrong patient", str(len(s["wrong_disclosure"]))),
        (f"Dictated findings found (recall, {f['gold']} said)", f"{f['recall']:.2f}"),
        ("Findings in the draft that were said (precision)", f"{f['precision']:.2f}"),
        ("Findings F1", f"{f['f1']:.2f}"),
        ("Findings added that were not said", str(f["fp"])),
        ("Said as absent and kept absent (\"no ...\")", pct(*f["negated"])),
        ("Said as uncertain and kept uncertain (\"possible ...\")", pct(*f["hedged"])),
        ("Present turned absent, or stated turned uncertain", str(f["changed"])),
    ]
    if s["latency_ms"]:
        lat = s["latency_ms"]
        rows.append(("Model call latency, median / p90", f"{lat['median']} / {lat['p90']} ms"))
        rows.append(("Tokens per call (mean)", str(s["tokens_per_call"])))
        rows.append(("Calls that fell back to the rules", str(s["fallbacks"])))
    title = f"Interpreter: {engine}  |  {s['cases']} cases ({s['conversation']} conversation, {s['dictation']} dictation)"
    if markdown:
        print(f"**{title}**\n\n| Measure | Result |\n|---|---|")
        for k, v in rows:
            print(f"| {k} | {v} |")
        print("\n| Group | Cases fully correct |\n|---|---|")
        for g, (a, b) in s["groups"].items():
            print(f"| {g} | {pct(a, b)} |")
    else:
        print(title)
        for k, v in rows:
            print(f"  {k:<58} {v}")
        print("  By group: " + ", ".join(f"{g} {a}/{b}" for g, (a, b) in s["groups"].items()))
    failed = [r for r in results if not r.passed]
    print(f"\n{'#### ' if markdown else ''}Cases not fully correct ({len(failed)})\n")
    for r in failed:
        print(f"{'- ' if markdown else '  '}{r.case['id']} \"{r.case['text']}\": " + "; ".join(r.problems))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interpreter", choices=["rule_based", "groq", "openai_compat"], default="rule_based")
    ap.add_argument("--pace", type=float, default=8.0, help="seconds between model calls (free tier: about 8/min)")
    ap.add_argument("--only", default="", help="run only case ids starting with this prefix (e.g. F, D0, R1)")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", type=Path, help="write per-case results here")
    opts = ap.parse_args()

    os.environ["INTERPRETER"] = opts.interpreter
    settings = load_settings()
    interp = None
    if opts.interpreter != "rule_based":
        if not settings.llm_api_key:
            print("No API key found. Put GROQ_API_KEY (or LLM_API_KEY) in .env next to README.md.")
            return 2
        interp = PacedInterpreter(build_interpreter(settings), opts.pace)
    engine = f"{opts.interpreter} {settings.llm_model}".strip() if interp else "rule_based (offline)"

    repo = Repository(DATA_DIR)
    clinician = repo.get_clinician("dr_demo")
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [c for c in cases if c["id"].startswith(opts.only)]
    results = []
    for i, case in enumerate(cases, 1):
        assistant = Assistant(repo, SessionStore(), DraftStore(), interp)
        r = run_case(assistant, interp, case, clinician)
        results.append(r)
        if opts.verbose or (interp and not r.passed):
            mark = "ok  " if r.passed else "FAIL"
            print(f"[{i}/{len(cases)}] {mark} {case['id']} {case['text'][:60]!r} -> {r.resp.type}"
                  + (f"  ({'; '.join(r.problems)})" if r.problems else ""), flush=True)
        elif interp:
            print(f"[{i}/{len(cases)}] ok   {case['id']}", flush=True)

    s = summarize(results)
    print()
    print_report(s, results, engine, opts.markdown)
    if opts.json:
        opts.json.write_text(json.dumps({"engine": engine, "summary": s, "cases": [
            {"id": r.case["id"], "text": r.case["text"], "type": r.resp.type, "patient": _patient(r.resp),
             "report": _report(r.resp), "passed": r.passed, "problems": r.problems, "llm_ms": r.llm_ms,
             "fell_back": r.fell_back} for r in results]}, indent=1), encoding="utf-8")
    return 1 if s["wrong_disclosure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
