from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .gemini_medical import analyze_with_gemini, user_facing_gemini_error


@dataclass
class Decision:
    priority: str
    title: str
    description: str


COMMON_MEDICATIONS = [
    "aspirin",
    "lisinopril",
    "metformin",
    "amoxicillin",
    "ibuprofen",
    "acetaminophen",
    "tylenol",
    "advil",
    "atorvastatin",
    "omeprazole",
    "albuterol",
    "insulin",
]

SYMPTOMS = [
    "dizzy",
    "dizziness",
    "lightheaded",
    "lightheadedness",
    "nausea",
    "nauseous",
    "rash",
    "fever",
    "headache",
    "weak",
    "weakness",
    "fatigue",
    "shortness of breath",
    "chest pain",
    "fainting",
    "swelling",
]

NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "fifteen": "15",
    "twenty": "20",
    "fifty": "50",
    "hundred": "100",
}

NON_CLINICAL_REASSURANCE_PHRASES = {
    "feeling fine",
    "feel fine",
    "feeling okay",
    "feel okay",
    "doing okay",
    "doing well",
    "better now",
    "feeling better",
    "no symptoms",
    "no side effects",
}

HIGH_RISK_SYMPTOMS = {
    "chest pain",
    "shortness of breath",
    "fainting",
    "swelling",
    "trouble breathing",
    "allergic reaction",
    "anaphylaxis",
    "passed out",
}

IMMEDIATE_DANGER_PHRASES = {
    "about to die",
    "i am dying",
    "i'm dying",
    "going to die",
    "cannot breathe",
    "can't breathe",
    "not breathing",
    "stopped breathing",
    "unresponsive",
    "passing out",
    "passed out",
    "severe bleeding",
    "worst headache of my life",
}

MEDIUM_RISK_SYMPTOMS = {
    "dizzy",
    "dizziness",
    "lightheaded",
    "lightheadedness",
    "nausea",
    "nauseous",
    "rash",
    "fever",
    "headache",
    "weak",
    "weakness",
    "fatigue",
}


def _parse_patient_medications(patient_context: str | None) -> list[str]:
    if not patient_context:
        return []
    match = re.search(r"Medications:\s*([^.]*)", patient_context, flags=re.IGNORECASE)
    if not match:
        return []
    segment = match.group(1).strip()
    if not segment:
        return []

    medications: list[str] = []
    for piece in re.split(r"[,/;]|\band\b", segment, flags=re.IGNORECASE):
        cleaned = " ".join(piece.strip().split())
        if cleaned:
            medications.append(cleaned)
    return medications


def _candidate_medication_terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-]{4,}", text)


def _best_medication_match(candidate: str, known_meds: list[str]) -> str | None:
    candidate_lower = candidate.lower()
    best_name = ""
    best_score = 0.0
    for med in known_meds:
        score = SequenceMatcher(None, candidate_lower, med.lower()).ratio()
        if score > best_score:
            best_score = score
            best_name = med
    if best_name and best_score >= 0.72:
        return best_name
    return None


def apply_contextual_medication_corrections(text: str, patient_context: str | None = None) -> str:
    known_meds: list[str] = []
    seen: set[str] = set()
    for med in _parse_patient_medications(patient_context) + COMMON_MEDICATIONS:
        cleaned = " ".join(med.strip().split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        known_meds.append(cleaned)

    if not known_meds:
        return text

    corrected = text
    for candidate in _candidate_medication_terms(text):
        lower_candidate = candidate.lower()
        if lower_candidate in {"patient", "should", "continue", "medication", "milligram", "milligrams"}:
            continue
        if any(lower_candidate == med.lower() for med in known_meds):
            continue
        best_med = _best_medication_match(candidate, known_meds)
        if not best_med:
            continue
        corrected = re.sub(rf"\b{re.escape(candidate)}\b", best_med, corrected)

    return corrected

def normalize_transcript(text: str) -> str:
    normalized = " ".join(text.split())
    for word, digit in NUMBER_WORDS.items():
        normalized = re.sub(rf"\b{word}\b", digit, normalized, flags=re.IGNORECASE)
    return normalized


def canonicalize_entity(label: str, value: str) -> tuple[str, str]:
    clean_label = label.strip()
    clean_value = normalize_transcript(value.strip())

    if clean_label.lower() in {"dosage", "current dosage", "requested dosage"}:
        clean_value = re.sub(r"\bmilligrams?\b", "mg", clean_value, flags=re.IGNORECASE)
        clean_value = re.sub(r"\bmcgs?\b", "mcg", clean_value, flags=re.IGNORECASE)
        clean_value = re.sub(r"\s+", " ", clean_value).strip().lower()
        clean_value = clean_value.replace(" mg", " mg").replace(" mcg", " mcg")
    elif clean_label.lower() in {"medication", "drug name"}:
        clean_value = re.sub(r"\s+", " ", clean_value).strip().title()
    elif clean_label.lower() in {"symptom", "side effect"}:
        clean_value = re.sub(r"\s+", " ", clean_value).strip().lower()
    else:
        clean_value = re.sub(r"\s+", " ", clean_value).strip()

    return clean_label, clean_value


def is_clinically_meaningful_entity(label: str, value: str) -> bool:
    normalized_value = normalize_transcript(value).strip().lower()
    if label.lower() in {"symptom", "side effect"} and normalized_value in NON_CLINICAL_REASSURANCE_PHRASES:
        return False
    return True


def project_demo_entity(label: str, value: str) -> tuple[str, str] | None:
    label_lower = label.lower()
    if label_lower in {"medication", "drug name"}:
        return ("Drug Name", value)
    if label_lower in {"dosage", "current dosage", "requested dosage"}:
        return ("Dosage", value)
    if label_lower == "symptom":
        return ("Symptom", value)
    if label_lower == "side effect":
        return ("Side Effect", value)
    return None


def extract_entities(text: str, patient_context: str | None = None) -> list[dict[str, str]]:
    normalized = normalize_transcript(text)
    lowered = normalized.lower()
    entities: list[dict[str, str]] = []

    for med in COMMON_MEDICATIONS:
        if re.search(rf"\b{re.escape(med)}\b", lowered, flags=re.IGNORECASE):
            entities.append({"label": "Drug Name", "value": med.title()})

    dosage_matches = re.findall(
        r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|milligram|milligrams)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    for match in dosage_matches:
        clean = (
            match.replace("milligrams", "mg")
            .replace("milligram", "mg")
            .replace("MG", "mg")
        )
        entities.append({"label": "Dosage", "value": clean})

    for symptom in SYMPTOMS:
        if symptom in lowered:
            entities.append({"label": "Symptom", "value": symptom})

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        canonical_label, canonical_value = canonicalize_entity(entity["label"], entity["value"])
        if not is_clinically_meaningful_entity(canonical_label, canonical_value):
            continue
        projected = project_demo_entity(canonical_label, canonical_value)
        if not projected:
            continue
        entity["label"], entity["value"] = projected
        key = (canonical_label.lower(), canonical_value.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(entity)

    return deduped


def extract_entities_with_ai(
    text: str,
    patient_context: str | None = None,
    utterances: list[dict[str, str]] | None = None,
    important_terms: str | None = None,
) -> tuple[str, list[dict[str, str]], str, dict[str, str] | None]:
    context_corrected = apply_contextual_medication_corrections(text, patient_context=patient_context)
    normalized = normalize_transcript(context_corrected)
    fallback_entities = extract_entities(normalized, patient_context=patient_context)

    try:
        ai_result = analyze_with_gemini(
            normalized,
            patient_context=patient_context,
            utterances=utterances,
            important_terms=important_terms,
        )
    except Exception as exc:
        return normalized, fallback_entities, user_facing_gemini_error(exc), None

    if not ai_result:
        return normalized, fallback_entities, "Gemini extraction disabled", None

    ai_normalized = normalize_transcript(ai_result.get("normalized_transcript", normalized))
    ai_entities = ai_result.get("entities") or []

    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    seen_labels: set[str] = set()

    for entity in ai_entities:
        label = str(entity.get("label", "")).strip()
        value = str(entity.get("value", "")).strip()
        if not label or not value:
            continue
        canonical_label, canonical_value = canonicalize_entity(label, value)
        if not is_clinically_meaningful_entity(canonical_label, canonical_value):
            continue
        projected = project_demo_entity(canonical_label, canonical_value)
        if not projected:
            continue
        canonical_label, canonical_value = projected
        key = (canonical_label.lower(), canonical_value.lower())
        if key not in seen:
            seen.add(key)
            seen_labels.add(canonical_label.lower())
            merged.append({"label": canonical_label, "value": canonical_value})

    for item in fallback_entities:
        canonical_label, canonical_value = canonicalize_entity(item["label"], item["value"])
        if not is_clinically_meaningful_entity(canonical_label, canonical_value):
            continue
        projected = project_demo_entity(canonical_label, canonical_value)
        if not projected:
            continue
        canonical_label, canonical_value = projected
        label_key = canonical_label.lower()
        key = (label_key, canonical_value.lower())
        if key in seen:
            continue
        if label_key in seen_labels:
            continue
        seen.add(key)
        seen_labels.add(label_key)
        merged.append({"label": canonical_label, "value": canonical_value})

    raw_decision = ai_result.get("decision")
    decision: dict[str, str] | None = None
    if isinstance(raw_decision, dict):
        priority = str(raw_decision.get("priority", "")).strip()
        title = str(raw_decision.get("title", "")).strip()
        description = str(raw_decision.get("description", "")).strip()
        if priority in {"Low", "Medium", "High", "Urgent"} and title and description:
            decision = {
                "priority": priority,
                "title": title,
                "description": description,
            }

    return ai_normalized, merged, ai_result.get("notes", ""), decision


def decide_priority(text: str, entities: list[dict[str, str]]) -> Decision:
    lowered = text.lower()

    symptom_values = {
        str(entity.get("value", "")).strip().lower()
        for entity in entities
        if entity.get("label") in {"Symptom", "Side Effect"}
    }
    has_drug = any(entity.get("label") == "Drug Name" for entity in entities)
    has_dosage = any(entity.get("label") == "Dosage" for entity in entities)
    has_side_effect = any(entity.get("label") == "Side Effect" for entity in entities)
    has_symptom = any(entity.get("label") == "Symptom" for entity in entities)

    if any(phrase in lowered for phrase in IMMEDIATE_DANGER_PHRASES):
        return Decision(
            priority="Urgent",
            title="Emergency escalation",
            description="Possible life-threatening language was detected. Escalate immediately and direct the patient to emergency care.",
        )

    if any(term in lowered for term in HIGH_RISK_SYMPTOMS) or symptom_values.intersection(HIGH_RISK_SYMPTOMS):
        return Decision(
            priority="High",
            title="Escalate to clinician",
            description="Potentially serious symptoms detected and should be reviewed urgently.",
        )

    urgent_language = any(
        phrase in lowered
        for phrase in {"immediately", "right now", "as soon as possible", "urgent", "emergency"}
    )
    guidance_language = any(
        phrase in lowered
        for phrase in {
            "should i continue",
            "should i stop",
            "can i keep taking",
            "can someone call me back",
            "should i go to the hospital",
        }
    )
    multiple_clinical_findings = len(symptom_values) >= 2
    medium_risk_signal = bool(symptom_values.intersection(MEDIUM_RISK_SYMPTOMS))

    if urgent_language and (has_symptom or has_side_effect or has_drug):
        return Decision(
            priority="High",
            title="Urgent clinician review",
            description="The patient reports a clinically relevant issue with urgent language that should be reviewed quickly.",
        )

    if has_side_effect and has_drug:
        return Decision(
            priority="Medium",
            title="Route to nurse review",
            description="A possible medication-related side effect was reported and should receive follow-up.",
        )

    if has_drug and (has_symptom or medium_risk_signal):
        return Decision(
            priority="Medium",
            title="Medication symptom follow-up",
            description="The patient mentioned both a medication and a symptom, so the call should be reviewed.",
        )

    if has_symptom and (guidance_language or multiple_clinical_findings):
        return Decision(
            priority="Medium",
            title="Symptom follow-up needed",
            description="The patient reported symptoms that should receive clinical follow-up.",
        )

    if guidance_language and (has_drug or has_dosage):
        return Decision(
            priority="Medium",
            title="Medication guidance requested",
            description="The patient is asking for medication guidance and should receive follow-up.",
        )

    return Decision(
        priority="Low",
        title="Monitor and document",
        description="No urgent concern detected. Save the interaction in the patient timeline.",
    )


def entities_to_json(entities: list[dict[str, str]]) -> str:
    return json.dumps(entities, ensure_ascii=False)
