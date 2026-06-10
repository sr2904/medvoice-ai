from __future__ import annotations

import json
from typing import Any

import requests

from .config import settings

SYSTEM_INSTRUCTION = """
You analyze healthcare phone call transcripts for clinical signal extraction.

Goals:
- Correct obvious transcription mistakes for medication names, allergies, symptoms, side effects, dosages, frequencies, and durations.
- Preserve uncertainty. Never invent facts that are not grounded in the transcript.
- Extract medically important terms even if they appear misspelled or partially recognized.
- Prefer generic or clearly spoken drug names if you can infer them with high confidence.
- For entities, resolve near-duplicate wording into one clinically appropriate mention. Do not emit both "nausea" and "nauseous", or both "dizzy" and "dizziness", for the same underlying symptom unless the transcript clearly distinguishes them.
- Be conservative with cross-talk: ignore likely background speech, room noise, partial interruptions, and non-patient fragments unless they add medically relevant context.
- If the patient appears to mention a medication, allergy, side effect, or symptom, include it even when the STT wording is slightly imperfect.
- Return every clinically relevant mention, not just one example per category.
- The normalized_transcript must be a patient-focused clinical summary transcript, not a raw transcript clone.
- Remove obvious background phrases, friend comments, emotional exclamations, filler chatter, and irrelevant fragments such as "oh my god", "bro", "this is so annoying", unless they directly communicate a symptom or risk.
- If a word like "disease" appears as a frustrated exclamation or stray fragment and not as an actual diagnosis, exclude it from normalized_transcript and do not extract it as a condition.
- Do not label reassuring wellness statements as symptoms. Phrases like "feeling fine", "feeling okay", "doing well", "better now", or "no symptoms" are not symptoms or side effects.
- Preserve person names when clearly spoken or provided in the important terms context. Do not drop a caller name just because it is uncommon.
- Distinguish between a current medication dose and a requested additional dose. If the caller says they are already taking one dose and want more, keep those as separate facts.
- If the caller asks for something urgently using phrases like "immediately", "right now", "as soon as possible", or "urgent", reflect that in both the normalized transcript and the decision priority.
- If the caller says they are dying, about to die, cannot breathe, are unresponsive, or uses other life-threatening language, the decision priority must be Urgent.
- Do not rewrite "want 1 mg more" as if the patient already started taking that extra amount.
- Prefer a single canonical symptom phrase that best matches the patient meaning, not multiple grammatical variants of the same symptom.

Return strict JSON with this shape:
{
  "normalized_transcript": "string",
  "entities": [{"label": "Drug Name|Dosage|Symptom|Side Effect", "value": "string"}],
  "decision": {
    "priority": "Low|Medium|High|Urgent",
    "title": "string",
    "description": "string"
  },
  "notes": "short string"
}
""".strip()


def _extract_text(response_json: dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates")

    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    text = "".join(text_parts).strip()
    if not text:
        raise ValueError("Gemini returned empty content")
    return text


def _coerce_entities(raw_entities: Any) -> list[dict[str, str]]:
    allowed_labels = {
        "medication": "Drug Name",
        "drug name": "Drug Name",
        "dosage": "Dosage",
        "current dosage": "Dosage",
        "requested dosage": "Dosage",
        "symptom": "Symptom",
        "side effect": "Side Effect",
    }
    entities: list[dict[str, str]] = []
    if not isinstance(raw_entities, list):
        return entities

    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = str(item.get("value", "")).strip()
        if not label or not value:
            continue
        canonical_label = allowed_labels.get(label.lower(), label)
        entities.append({"label": canonical_label, "value": value})
    return entities


def user_facing_gemini_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status_code = exc.response.status_code
        if status_code == 429:
            return "Gemini analysis is temporarily rate-limited, so the app used its built-in clinical fallback."
        if 500 <= status_code <= 599:
            return "Gemini analysis is temporarily unavailable, so the app used its built-in clinical fallback."
        return f"Gemini analysis failed with HTTP {status_code}, so the app used its built-in clinical fallback."

    if isinstance(exc, requests.Timeout):
        return "Gemini analysis timed out, so the app used its built-in clinical fallback."

    if isinstance(exc, requests.RequestException):
        return "Gemini analysis could not be reached, so the app used its built-in clinical fallback."

    return "Gemini analysis failed, so the app used its built-in clinical fallback."


def analyze_with_gemini(
    transcript: str,
    patient_context: str | None = None,
    utterances: list[dict[str, str]] | None = None,
    important_terms: str | None = None,
) -> dict[str, Any] | None:
    api_key = (settings.GEMINI_API_KEY or "").strip()
    if not settings.ENABLE_GEMINI_MEDICAL_EXTRACTION or not api_key:
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": json.dumps(
                            {
                                "transcript": transcript,
                                "utterances": utterances or [],
                                "patient_context": patient_context or "",
                                "important_terms": important_terms or "",
                                "focus": [
                                    "person names",
                                    "medication names",
                                    "allergies",
                                    "symptoms",
                                    "dosages",
                                    "current dose vs requested extra dose",
                                    "side effects",
                                    "durations",
                                    "frequencies",
                                    "urgency language",
                                ],
                                "instruction_hint": "Return a clinically useful patient-only normalized transcript. Drop off-speaker chatter and non-clinical interjections. Preserve names and separate current dose from requested extra dose.",
                            },
                            ensure_ascii=False,
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    response = requests.post(url, json=payload, timeout=45)
    response.raise_for_status()
    raw_text = _extract_text(response.json())
    data = json.loads(raw_text)

    normalized = str(data.get("normalized_transcript", transcript)).strip() or transcript
    entities = _coerce_entities(data.get("entities"))
    decision = data.get("decision")
    notes = str(data.get("notes", "")).strip()
    return {
        "normalized_transcript": normalized,
        "entities": entities,
        "decision": decision if isinstance(decision, dict) else None,
        "notes": notes,
    }
