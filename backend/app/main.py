from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import func
from sqlalchemy.orm import Session
from .audio_cleaner import clean_audio_bytes

from .audio_utils import convert_to_telephony_wav, save_audio_bytes
from .config import settings
from .database import Base, engine, get_db
from .medical_logic import decide_priority, entities_to_json, extract_entities_with_ai
from .models import CallRecord, Patient
from .schemas import PatientCreate, PatientRead, PreviewResponse, TranscriptResponse
from .seed import build_demo_patient_profile, seed_patients
from .stt import STTService
from .telephony import build_twiml_stream_response

app = FastAPI(title="CareCaller Backend", version="1.0.0")

origins = (
    [item.strip() for item in settings.CORS_ALLOW_ORIGINS.split(",")]
    if settings.CORS_ALLOW_ORIGINS
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUDIO_DIR = Path("./uploaded_audio")


def _extract_spoken_name(*texts: str | None) -> str | None:
    patterns = [
        r"\bmy name is ([A-Za-z]+(?: [A-Za-z]+){0,2})",
        r"\bthis is ([A-Za-z]+(?: [A-Za-z]+){0,2})",
    ]
    for text in texts:
        if not text:
            continue
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = " ".join(match.group(1).split()).strip(" .,?!")
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in {"taking", "feeling", "calling", "currently", "still"}:
                continue
            return candidate.title()
    return None


def _build_medication_summary(
    entities: list[dict[str, str]],
    *texts: str | None,
) -> str | None:
    drug_names = [item["value"].strip() for item in entities if item.get("label") == "Drug Name" and item.get("value")]
    dosages = [item["value"].strip() for item in entities if item.get("label") == "Dosage" and item.get("value")]

    if not drug_names:
        for text in texts:
            if not text:
                continue
            match = re.search(
                r"\b(?:taking|on|using|used)\s+([A-Za-z][A-Za-z\-]+)",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                candidate = match.group(1).strip(" .,?!").title()
                if candidate.lower() not in {"medication", "medicine", "drug", "dose"}:
                    drug_names.append(candidate)
                    break

    if not drug_names:
        return None

    unique_drugs: list[str] = []
    seen_drugs: set[str] = set()
    for drug in drug_names:
        key = drug.lower()
        if key in seen_drugs:
            continue
        seen_drugs.add(key)
        unique_drugs.append(drug)

    unique_dosages: list[str] = []
    seen_dosages: set[str] = set()
    for dosage in dosages:
        key = dosage.lower()
        if key in seen_dosages:
            continue
        seen_dosages.add(key)
        unique_dosages.append(dosage)

    if len(unique_drugs) == 1 and unique_dosages:
        return f"{unique_drugs[0]} {' / '.join(unique_dosages)}"
    return ", ".join(unique_drugs)


def _build_condition_summary(entities: list[dict[str, str]]) -> str | None:
    findings = [
        item["value"].strip()
        for item in entities
        if item.get("label") in {"Symptom", "Side Effect"} and item.get("value")
    ]
    if not findings:
        return None

    deduped: list[str] = []
    seen: set[str] = set()
    for finding in findings:
        key = finding.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding.lower())

    return f"Reported concerns: {', '.join(deduped)}"


def _update_patient_from_call(
    patient: Patient,
    transcript: str,
    normalized: str,
    entities: list[dict[str, str]],
) -> None:
    spoken_name = _extract_spoken_name(transcript, normalized)
    if spoken_name:
        patient.full_name = spoken_name

    medication_summary = _build_medication_summary(entities, transcript, normalized)
    if medication_summary:
        patient.medications = medication_summary

    condition_summary = _build_condition_summary(entities)
    if condition_summary:
        patient.conditions = condition_summary


def _get_or_create_patient_by_name(full_name: str, db: Session) -> Patient:
    normalized_name = " ".join(full_name.split()).strip()
    existing = (
        db.query(Patient)
        .filter(func.lower(Patient.full_name) == normalized_name.lower())
        .first()
    )
    if existing:
        return existing

    profile = build_demo_patient_profile(normalized_name)
    patient = Patient(**profile)
    db.add(patient)
    db.flush()
    return patient


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        seed_patients(db)
    finally:
        db.close()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/capabilities")
def capabilities() -> dict:
    return {
        "telephony_audio_profile": f"{settings.TELEPHONY_SAMPLE_RATE}hz-{settings.TELEPHONY_ENCODING}",
        "medical_stt_model": settings.DEEPGRAM_MODEL,
        "gemini_medical_extraction_enabled": settings.ENABLE_GEMINI_MEDICAL_EXTRACTION,
        "near_real_time_preview_enabled": True,
        "benchmark_metrics_available": ["WER", "CER", "numeric_accuracy", "keyword_accuracy"],
        "twilio_voice_webhook_enabled": settings.TWILIO_ENABLE_VOICE_WEBHOOK,
        "twilio_voice_webhook_url": (
            f"{settings.PUBLIC_BASE_URL.rstrip('/')}/api/telephony/voice"
            if settings.TWILIO_ENABLE_VOICE_WEBHOOK and settings.PUBLIC_BASE_URL
            else None
        ),
    }


@app.get("/api/patients", response_model=list[PatientRead])
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).order_by(Patient.full_name.asc()).all()


@app.post("/api/patients", response_model=PatientRead)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    normalized_name = " ".join(payload.full_name.split()).strip()
    if not normalized_name:
        raise HTTPException(status_code=400, detail="Patient name is required")

    existing = (
        db.query(Patient)
        .filter(func.lower(Patient.full_name) == normalized_name.lower())
        .first()
    )
    if existing:
        return existing

    profile = build_demo_patient_profile(normalized_name)
    payload_data = payload.model_dump()
    payload_data["full_name"] = normalized_name

    for field_name in ("phone_number", "dob", "medications", "conditions"):
        if not payload_data.get(field_name):
            payload_data[field_name] = profile[field_name]

    patient = Patient(**payload_data)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@app.get("/api/patients/{patient_id}/timeline")
def patient_timeline(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    calls = (
        db.query(CallRecord)
        .filter(CallRecord.patient_id == patient_id)
        .order_by(CallRecord.created_at.desc())
        .all()
    )

    return {
        "patient": PatientRead.model_validate(patient).model_dump(),
        "calls": [
            {
                "id": call.id,
                "created_at": call.created_at.isoformat(),
                "priority": call.priority,
                "decision_title": call.decision_title,
                "decision_description": call.decision_description,
                "transcript": call.transcript,
                "normalized_transcript": call.normalized_transcript,
                "extracted_entities": call.extracted_entities,
            }
            for call in calls
        ],
    }


@app.post("/api/transcribe", response_model=TranscriptResponse)
async def transcribe_audio(
    patient_id: int = Form(...),
    scenario: str | None = Form(default=None),
    important_terms: str | None = Form(default=None),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        raw_bytes = await audio.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        patient_context = (
            f"Patient Name: {patient.full_name or ''}. "
            f"Medications: {patient.medications or ''}. "
            f"Conditions: {patient.conditions or ''}."
        )

        wav_bytes = convert_to_telephony_wav(raw_bytes, audio.filename)
        cleaned_wav_bytes = clean_audio_bytes(wav_bytes)
        stt_result = STTService.transcribe_bytes(
            cleaned_wav_bytes,
            patient_context=patient_context,
            important_terms=important_terms,
        )
        transcript = stt_result["transcript"]

        if not transcript:
            raise HTTPException(status_code=422, detail="Could not transcribe audio")

        normalized, entities, analysis_notes, ai_decision = extract_entities_with_ai(
            transcript,
            patient_context=patient_context,
            utterances=stt_result.get("utterances") or [],
            important_terms=important_terms,
        )
        spoken_name = _extract_spoken_name(transcript, normalized)
        target_patient = patient
        if spoken_name and spoken_name.lower() != (patient.full_name or "").strip().lower():
            target_patient = _get_or_create_patient_by_name(spoken_name, db)

        fallback_decision = decide_priority(normalized, entities)
        decision = {
            "priority": fallback_decision.priority,
            "title": fallback_decision.title,
            "description": fallback_decision.description,
        }
        if ai_decision:
            priority_rank = {"Low": 0, "Medium": 1, "High": 2, "Urgent": 3}
            ai_priority = ai_decision.get("priority", "Low")
            if priority_rank.get(ai_priority, -1) >= priority_rank[fallback_decision.priority]:
                decision = ai_decision
            else:
                safety_note = (
                    f"{analysis_notes}\nDecision safety override applied: "
                    f"fallback rules raised priority to {fallback_decision.priority}."
                ).strip()
                analysis_notes = safety_note

        audio_filename = f"patient_{target_patient.id}_{audio.filename or 'call.wav'}"
        save_audio_bytes(AUDIO_DIR / audio_filename, cleaned_wav_bytes)

        record = CallRecord(
            patient_id=target_patient.id,
            scenario=scenario,
            audio_filename=audio_filename,
            transcript=transcript,
            normalized_transcript=normalized,
            priority=decision["priority"],
            decision_title=decision["title"],
            decision_description=decision["description"],
            extracted_entities=entities_to_json(entities),
        )
        db.add(record)
        _update_patient_from_call(target_patient, transcript, normalized, entities)
        db.commit()

        return TranscriptResponse(
            transcript=transcript,
            normalized_transcript=normalized,
            entities=entities,
            decision=decision,
            patient_context_used=patient_context,
            analysis_notes=analysis_notes,
            stt_model_used=stt_result.get("model"),
            audio_profile_used=f"cleaned-{stt_result.get('sample_rate', 16000)}hz",
        )

    except HTTPException:
        raise
    except Exception:
        import traceback
        print("=== TRANSCRIBE ERROR START ===")
        traceback.print_exc()
        print("=== TRANSCRIBE ERROR END ===")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/transcribe-preview", response_model=PreviewResponse)
async def transcribe_preview(
    patient_id: int | None = Form(default=None),
    important_terms: str | None = Form(default=None),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        patient_context = ""
        if patient_id is not None:
            patient = db.query(Patient).filter(Patient.id == patient_id).first()
            if patient:
                patient_context = (
                    f"Patient Name: {patient.full_name or ''}. "
                    f"Medications: {patient.medications or ''}. "
                    f"Conditions: {patient.conditions or ''}."
                )

        raw_bytes = await audio.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        wav_bytes = convert_to_telephony_wav(raw_bytes, audio.filename)
        cleaned_wav_bytes = clean_audio_bytes(wav_bytes)
        stt_result = STTService.transcribe_bytes(
            cleaned_wav_bytes,
            patient_context=patient_context,
            important_terms=important_terms,
        )
        return PreviewResponse(
            transcript=stt_result["transcript"],
            stt_model_used=stt_result.get("model"),
            audio_profile_used=f"cleaned-{stt_result.get('sample_rate', 16000)}hz",
        )
    except HTTPException:
        raise
    except Exception:
        return PreviewResponse(
            transcript="",
            stt_model_used=None,
            audio_profile_used=None,
        )


@app.post("/api/telephony/voice", response_class=PlainTextResponse)
def telephony_voice_webhook() -> str:
    return build_twiml_stream_response()


@app.post("/api/telephony/media-stream")
def telephony_media_stream_placeholder() -> dict[str, str]:
    return {
        "status": "ready",
        "message": "WebSocket media streaming is not implemented in this HTTP placeholder yet.",
    }
