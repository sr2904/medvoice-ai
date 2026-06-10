from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(200), nullable=False)
    phone_number = Column(String(50), nullable=True)
    dob = Column(String(50), nullable=True)
    medications = Column(Text, nullable=True)
    conditions = Column(Text, nullable=True)

    calls = relationship("CallRecord", back_populates="patient", cascade="all, delete-orphan")


class CallRecord(Base):
    __tablename__ = "call_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    scenario = Column(String(100), nullable=True)
    audio_filename = Column(String(255), nullable=True)
    transcript = Column(Text, nullable=False)
    normalized_transcript = Column(Text, nullable=False)
    priority = Column(String(20), nullable=False)
    decision_title = Column(String(255), nullable=False)
    decision_description = Column(Text, nullable=False)
    extracted_entities = Column(Text, nullable=False)

    patient = relationship("Patient", back_populates="calls")
