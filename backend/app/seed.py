from __future__ import annotations

import hashlib

from sqlalchemy.orm import Session

from .models import Patient


DEMO_PATIENT_TEMPLATES = [
    {
        "conditions": "Type 2 diabetes, hypertension",
        "medications": "Metformin 500 mg twice daily, Lisinopril 10 mg daily",
    },
    {
        "conditions": "Atrial fibrillation, hyperlipidemia",
        "medications": "Warfarin 5 mg daily, Atorvastatin 20 mg nightly",
    },
    {
        "conditions": "Asthma, seasonal allergies",
        "medications": "Albuterol inhaler as needed, Cetirizine 10 mg daily",
    },
    {
        "conditions": "Chronic back pain, acid reflux",
        "medications": "Ibuprofen 400 mg as needed, Omeprazole 20 mg daily",
    },
    {
        "conditions": "Migraine history, anxiety",
        "medications": "Sumatriptan 50 mg as needed, Sertraline 50 mg daily",
    },
    {
        "conditions": "Heart failure, fluid retention",
        "medications": "Furosemide 20 mg daily, Carvedilol 6.25 mg twice daily",
    },
]


def build_demo_patient_profile(full_name: str) -> dict[str, str]:
    normalized_name = " ".join(full_name.split()).strip()
    seed = int(hashlib.sha256(normalized_name.lower().encode("utf-8")).hexdigest()[:8], 16)
    template = DEMO_PATIENT_TEMPLATES[seed % len(DEMO_PATIENT_TEMPLATES)]

    month = (seed % 12) + 1
    day = (seed % 27) + 1
    year = 1958 + (seed % 36)
    phone_suffix = 1000 + (seed % 9000)

    return {
        "full_name": normalized_name,
        "phone_number": f"+1 (602) 555-{phone_suffix:04d}",
        "dob": f"{year:04d}-{month:02d}-{day:02d}",
        "medications": template["medications"],
        "conditions": template["conditions"],
    }


def seed_patients(db: Session) -> None:
    if db.query(Patient).count() > 0:
        return
    db.add_all([
        Patient(
            full_name="Maria Johnson",
            phone_number="+1 (602) 555-0112",
            dob="1979-04-16",
            medications="Metformin 500 mg, Lisinopril 10 mg",
            conditions="Type 2 diabetes, hypertension",
        ),
        Patient(
            full_name="James Lee",
            phone_number="+1 (480) 555-0107",
            dob="1968-11-03",
            medications="Warfarin 5 mg, Atorvastatin 20 mg",
            conditions="Atrial fibrillation, hyperlipidemia",
        ),
    ])
    db.commit()
