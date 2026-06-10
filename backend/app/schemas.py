from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PatientCreate(BaseModel):
    full_name: str
    phone_number: Optional[str] = None
    dob: Optional[str] = None
    medications: Optional[str] = None
    conditions: Optional[str] = None


class PatientRead(PatientCreate):
    id: int

    class Config:
        from_attributes = True


class Entity(BaseModel):
    label: str
    value: str


class Decision(BaseModel):
    priority: str = Field(pattern="^(Low|Medium|High|Urgent)$")
    title: str
    description: str


class TranscriptResponse(BaseModel):
    transcript: str
    normalized_transcript: str
    entities: List[Entity]
    decision: Decision
    patient_context_used: Optional[str] = None
    analysis_notes: Optional[str] = None
    stt_model_used: Optional[str] = None
    audio_profile_used: Optional[str] = None


class PreviewResponse(BaseModel):
    transcript: str
    stt_model_used: Optional[str] = None
    audio_profile_used: Optional[str] = None


class CallRead(BaseModel):
    id: int
    patient_id: int
    created_at: datetime
    scenario: Optional[str] = None
    audio_filename: Optional[str] = None
    transcript: str
    normalized_transcript: str
    priority: str
    decision_title: str
    decision_description: str
    extracted_entities: str

    class Config:
        from_attributes = True
