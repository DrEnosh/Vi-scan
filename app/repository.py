"""Mock data access layer.

Every read goes through the clinician's clinic scope, so a patient from another clinic is
invisible rather than "forbidden" (no leak that the record exists). In production this is
where the PACS/EHR adapter and row-level authorization would live.
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import Clinician, Patient, Report


class Repository:
    def __init__(self, data_dir: Path):
        self.clinicians: dict[str, Clinician] = {
            cid: Clinician(clinician_id=cid, **c)
            for cid, c in json.loads((data_dir / "clinicians.json").read_text()).items()
        }
        self._patients: dict[str, Patient] = {
            p["patient_id"]: Patient(**p) for p in json.loads((data_dir / "patients.json").read_text())
        }
        reports = [Report(**r) for r in json.loads((data_dir / "reports.json").read_text())]
        self._reports: dict[str, Report] = {r.report_id: r for r in reports}
        self._reports_by_patient: dict[str, list[Report]] = {}
        for r in reports:
            self._reports_by_patient.setdefault(r.patient_id, []).append(r)
        for lst in self._reports_by_patient.values():
            lst.sort(key=lambda r: r.study_date, reverse=True)  # newest first

    # ------------------------------------------------------------------ clinicians
    def get_clinician(self, clinician_id: str | None) -> Clinician | None:
        if not clinician_id:
            return None
        return self.clinicians.get(clinician_id)

    # ------------------------------------------------------------------ patients
    def patients_for(self, clinician: Clinician) -> list[Patient]:
        return [p for p in self._patients.values() if p.clinic_id in clinician.clinic_ids]

    def get_patient(self, patient_id: str, clinician: Clinician) -> Patient | None:
        p = self._patients.get(patient_id.upper())
        if p is None or p.clinic_id not in clinician.clinic_ids:
            return None
        return p

    # ------------------------------------------------------------------ reports
    def reports_for(self, patient: Patient) -> list[Report]:
        return list(self._reports_by_patient.get(patient.patient_id, []))

    def get_report(self, report_id: str, clinician: Clinician) -> Report | None:
        r = self._reports.get(report_id.upper())
        if r is None:
            return None
        if self.get_patient(r.patient_id, clinician) is None:
            return None
        return r
