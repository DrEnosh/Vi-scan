"""Clinician edits and sign-off for dictation drafts."""
from __future__ import annotations

from ..dental import CATEGORY_LABELS, FINDING_BY_KEY, is_valid_fdi, site_display
from ..models import Clinician, DraftFinding, DraftReport, DraftUpdateRequest, Site
from ..repository import Repository
from ..session import DraftStore
from .dictation import build_draft_text, finding_statement


class DraftError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def apply_edits(draft: DraftReport, req: DraftUpdateRequest, repo: Repository, clinician: Clinician) -> DraftReport:
    if draft.status == "clinician_approved":
        raise DraftError(409, "This draft is already approved; create a new draft to change it.")
    if req.patient_id:
        p = repo.get_patient(req.patient_id, clinician)
        if p is None:
            raise DraftError(404, f"No patient with ID {req.patient_id} in your clinic's records.")
        draft.patient = p.card()

    originals = {f.finding_id: f for f in draft.findings}
    new: list[DraftFinding] = []
    for i, e in enumerate(req.findings, 1):
        if e.category not in FINDING_BY_KEY and e.category != "other":
            raise DraftError(422, f"Finding {i}: unknown category '{e.category}'.")
        if e.tooth_fdi and not is_valid_fdi(e.tooth_fdi):
            raise DraftError(422, f"Finding {i}: '{e.tooth_fdi}' is not a valid FDI tooth number.")
        if e.tooth_fdi:
            site: Site | None = Site(type="tooth", fdi=e.tooth_fdi, label=site_display(e.tooth_fdi))
        elif e.site_label:
            site = Site(type="region", label=e.site_label.strip()[:80])
        else:
            site = None
        orig = originals.get(e.finding_id or "")
        f = DraftFinding(
            finding_id="", category=e.category,
            finding=(e.finding or CATEGORY_LABELS.get(e.category, "Other finding")).strip()[:120],
            qualifiers=[q.strip()[:40] for q in e.qualifiers if q.strip()],
            present=e.present, certainty=e.certainty,
            certainty_cue=orig.certainty_cue if orig and e.certainty == "uncertain" else None,
            site=site,
            evidence=orig.evidence if orig else (e.evidence or "Added by clinician during review"),
            source="clinician_edit",
        )
        if orig and _same(orig, f):
            f = orig
        else:
            f.review_reasons = (["Marked uncertain by clinician"] if f.certainty == "uncertain" else []) + \
                               (["No anatomical site"] if site is None else [])
            f.needs_review = bool(f.review_reasons)
            f.statement = finding_statement(f)
        new.append(f)
    for i, f in enumerate(new, 1):
        f.finding_id = f"D{i}"
        f.statement = finding_statement(f)
    draft.findings = new
    draft.status = "edited_pending_review"
    draft.version += 1
    draft.updated_at = DraftStore.now()
    draft.draft_text = build_draft_text(draft.findings, draft.unparsed_segments, draft.patient, draft.status)
    return draft


def _same(a: DraftFinding, b: DraftFinding) -> bool:
    site_a = (a.site.type, a.site.fdi, a.site.label) if a.site else None
    site_b = (b.site.type, b.site.fdi, b.site.label) if b.site else None
    return (a.category, a.finding, a.qualifiers, a.present, a.certainty, site_a) == \
           (b.category, b.finding, b.qualifiers, b.present, b.certainty, site_b)


def approve(draft: DraftReport, clinician: Clinician) -> DraftReport:
    if draft.status == "clinician_approved":
        raise DraftError(409, "Draft is already approved.")
    if draft.patient is None:
        raise DraftError(409, "Link a patient (patient_id) before approving this draft.")
    unresolved = [f.finding_id for f in draft.findings if f.site and f.site.type == "unresolved"]
    if unresolved:
        raise DraftError(409, f"Pick the exact tooth for {', '.join(unresolved)} before approving.")
    draft.status = "clinician_approved"
    draft.approved_by = clinician.clinician_id
    draft.approved_at = DraftStore.now()
    draft.updated_at = draft.approved_at
    draft.requires_clinician_review = False
    draft.draft_text = build_draft_text(draft.findings, draft.unparsed_segments, draft.patient, draft.status)
    return draft
