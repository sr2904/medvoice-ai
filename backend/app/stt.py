from __future__ import annotations

import os
import re
from typing import Any

from deepgram import DeepgramClient

from .config import settings


def _read_attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _extract_transcript(response: Any) -> str:
    results = _read_attr(response, "results")
    channels = _read_attr(results, "channels", []) or []
    if not channels:
        return ""
    alternatives = _read_attr(channels[0], "alternatives", []) or []
    if not alternatives:
        return ""
    return str(_read_attr(alternatives[0], "transcript", "")).strip()


def _extract_utterances(response: Any) -> list[dict[str, str]]:
    results = _read_attr(response, "results")
    utterances = _read_attr(results, "utterances", []) or []
    cleaned: list[dict[str, str]] = []
    for utterance in utterances:
        transcript = str(_read_attr(utterance, "transcript", "")).strip()
        if not transcript:
            continue
        speaker = _read_attr(utterance, "speaker", None)
        cleaned.append(
            {
                "speaker": "" if speaker is None else str(speaker),
                "transcript": transcript,
            }
        )
    return cleaned


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?", text))


def _patient_name_terms(patient_context: str | None) -> set[str]:
    if not patient_context:
        return set()
    match = re.search(r"Patient Name:\s*([^.]*)", patient_context, flags=re.IGNORECASE)
    if not match:
        return set()
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z]+", match.group(1))
        if len(token) >= 2
    }


def _clinical_terms(patient_context: str | None, important_terms: str | None) -> set[str]:
    terms = {
        "medication",
        "medicine",
        "dose",
        "dosage",
        "mg",
        "mcg",
        "milligram",
        "milligrams",
        "dizzy",
        "nausea",
        "nauseous",
        "rash",
        "swelling",
        "pain",
        "breathing",
    }
    for source in (patient_context, important_terms):
        if not source:
            continue
        for token in re.findall(r"[A-Za-z]+", source):
            if len(token) >= 3:
                terms.add(token.lower())
    return terms


def _speaker_score(
    transcript: str,
    patient_name_terms: set[str],
    clinical_terms: set[str],
) -> int:
    lowered = transcript.lower()
    tokens = re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?", lowered)
    score = len(tokens)
    score += sum(1 for token in tokens if token in {"i", "my", "me", "feel", "feeling", "taking"}) * 4
    score += sum(1 for token in tokens if token in clinical_terms) * 5
    score += sum(1 for token in tokens if token in patient_name_terms) * 6
    if "should i" in lowered or "can i" in lowered:
        score += 6
    if re.search(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml)\b", lowered):
        score += 6
    return score


def _should_keep_primary_utterance(
    transcript: str,
    patient_name_terms: set[str],
    clinical_terms: set[str],
) -> bool:
    score = _speaker_score(transcript, patient_name_terms, clinical_terms)
    word_count = _word_count(transcript)
    lowered = transcript.lower()
    has_numeric_dose = bool(re.search(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml)\b", lowered))
    has_clinical_term = any(token in lowered for token in clinical_terms)
    has_patient_name = any(token in lowered for token in patient_name_terms)

    if word_count >= 6:
        return True
    if has_numeric_dose or has_clinical_term or has_patient_name:
        return score >= 8
    return score >= 12


def _build_primary_speaker_transcript(
    utterances: list[dict[str, str]],
    fallback: str,
    patient_context: str | None = None,
    important_terms: str | None = None,
) -> str:
    if not utterances:
        return fallback

    patient_name_terms = _patient_name_terms(patient_context)
    clinical_terms = _clinical_terms(patient_context, important_terms)
    speaker_scores: dict[str, int] = {}
    for utterance in utterances:
        speaker = utterance.get("speaker", "")
        transcript = utterance.get("transcript", "")
        if not transcript:
            continue
        speaker_scores[speaker] = speaker_scores.get(speaker, 0) + _speaker_score(
            transcript,
            patient_name_terms,
            clinical_terms,
        )

    if not speaker_scores:
        return fallback

    primary_speaker = max(speaker_scores.items(), key=lambda item: item[1])[0]
    filtered = [
        utterance["transcript"]
        for utterance in utterances
        if utterance.get("speaker", "") == primary_speaker
        and _should_keep_primary_utterance(
            utterance.get("transcript", ""),
            patient_name_terms,
            clinical_terms,
        )
    ]
    if not filtered:
        return fallback
    return " ".join(filtered).strip()


def _build_keyterms(patient_context: str | None = None, important_terms: str | None = None) -> list[str]:
    seeds = [
        "allergy",
        "allergic reaction",
        "side effect",
        "dose",
        "dosage",
        "milligram",
        "milligrams",
        "shortness of breath",
        "chest pain",
        "dizziness",
        "nausea",
        "rash",
        "swelling",
    ]
    if patient_context:
        phrases = re.split(r"[,.;\n]|(?:Medications:)|(?:Conditions:)", patient_context, flags=re.IGNORECASE)
        for phrase in phrases:
            cleaned = " ".join(phrase.strip().split())
            if not cleaned:
                continue
            if len(cleaned) > 60:
                continue
            seeds.append(cleaned)
    if important_terms:
        for phrase in re.split(r"[,;\n]", important_terms):
            cleaned = " ".join(phrase.strip().split())
            if cleaned and len(cleaned) <= 60:
                seeds.append(cleaned)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in seeds:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item.strip())
    return deduped[:50]


class STTService:
    @staticmethod
    def transcribe_bytes(
        audio_bytes: bytes,
        patient_context: str | None = None,
        important_terms: str | None = None,
    ) -> dict[str, Any]:
        api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY missing in .env")

        dg = DeepgramClient(api_key=api_key)
        response = dg.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=settings.DEEPGRAM_MODEL,
            language=settings.DEEPGRAM_LANGUAGE,
            smart_format=True,
            punctuate=True,
            numerals=True,
            utterances=True,
            diarize=True,
            filler_words=False,
            keyterm=_build_keyterms(patient_context, important_terms),
        )

        transcript = _extract_transcript(response)
        utterances = _extract_utterances(response)
        speaker_focused_transcript = _build_primary_speaker_transcript(
            utterances,
            transcript,
            patient_context=patient_context,
            important_terms=important_terms,
        )
        if not transcript:
            raise ValueError("Deepgram returned an empty transcript")

        return {
            "transcript": speaker_focused_transcript or transcript,
            "full_transcript": transcript,
            "utterances": utterances,
            "model": settings.DEEPGRAM_MODEL,
            "sample_rate": 16000,
        }
