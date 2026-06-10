from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio_utils import convert_to_telephony_wav
from app.medical_logic import extract_entities_with_ai, normalize_transcript
from app.stt import STTService


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def tokenize_chars(text: str) -> list[str]:
    return list("".join(char for char in text.lower() if not char.isspace()))


def levenshtein(seq1: list[str], seq2: list[str]) -> int:
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)

    prev = list(range(len(seq2) + 1))
    for i, token1 in enumerate(seq1, start=1):
        curr = [i]
        for j, token2 in enumerate(seq2, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if token1 == token2 else 1),
                )
            )
        prev = curr
    return prev[-1]


def error_rate(reference_tokens: list[str], hypothesis_tokens: list[str]) -> float:
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    return levenshtein(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", normalize_transcript(text))


def numeric_accuracy(reference_text: str, hypothesis_text: str) -> float:
    reference_numbers = extract_numbers(reference_text)
    hypothesis_numbers = extract_numbers(hypothesis_text)
    if not reference_numbers:
        return 1.0
    matches = sum(
        1 for index, value in enumerate(reference_numbers)
        if index < len(hypothesis_numbers) and hypothesis_numbers[index] == value
    )
    return matches / len(reference_numbers)


def keyword_accuracy(expected_keywords: list[str], searchable_text: str) -> float:
    if not expected_keywords:
        return 1.0
    lowered = searchable_text.lower()
    matches = sum(1 for keyword in expected_keywords if keyword.lower() in lowered)
    return matches / len(expected_keywords)


def confusion_pair_accuracy(confusions: list[dict], hypothesis_text: str) -> float:
    if not confusions:
        return 1.0
    lowered = hypothesis_text.lower()
    matches = 0
    for pair in confusions:
        expected = str(pair.get("expected", "")).strip().lower()
        if expected and expected in lowered:
            matches += 1
    return matches / len(confusions)


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate_one(entry: dict) -> dict:
    audio_path = (ROOT / entry["audio_path"]).resolve() if not Path(entry["audio_path"]).is_absolute() else Path(entry["audio_path"])
    reference = entry["reference_transcript"]
    patient_context = entry.get("patient_context", "")
    expected_keywords = entry.get("medical_keywords", [])
    numeric_confusions = entry.get("numeric_confusions", [])
    result = {
        "id": entry.get("id", audio_path.stem),
        "category": entry.get("category", "uncategorized"),
        "reference": reference,
        "hypothesis": "",
        "normalized": "",
        "wer": 1.0,
        "cer": 1.0,
        "numeric_accuracy": 0.0 if extract_numbers(reference) else 1.0,
        "keyword_accuracy": 0.0 if expected_keywords else 1.0,
        "numeric_confusion_accuracy": 0.0 if numeric_confusions else 1.0,
        "entities": [],
        "analysis_notes": "",
        "status": "ok",
    }

    try:
        audio_bytes = audio_path.read_bytes()
        telephony_wav = convert_to_telephony_wav(audio_bytes, audio_path.name)
        stt_result = STTService.transcribe_bytes(telephony_wav, patient_context=patient_context)
        hypothesis = stt_result["transcript"]
        normalized, entities, notes, _ai_decision = extract_entities_with_ai(
            hypothesis,
            patient_context=patient_context,
            utterances=stt_result.get("utterances") or [],
        )
        entity_text = " ".join(entity["value"] for entity in entities)
        searchable_text = f"{hypothesis} {normalized} {entity_text}"
        result.update(
            {
                "hypothesis": hypothesis,
                "normalized": normalized,
                "wer": error_rate(tokenize_words(reference), tokenize_words(hypothesis)),
                "cer": error_rate(tokenize_chars(reference), tokenize_chars(hypothesis)),
                "numeric_accuracy": numeric_accuracy(reference, hypothesis),
                "keyword_accuracy": keyword_accuracy(expected_keywords, searchable_text),
                "numeric_confusion_accuracy": confusion_pair_accuracy(numeric_confusions, searchable_text),
                "entities": entities,
                "analysis_notes": notes,
            }
        )
    except Exception as exc:
        result["status"] = f"failed: {exc}"

    return result


def summarize(results: list[dict]) -> dict:
    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else math.nan

    categories: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        categories[result["category"]].append(result)

    summary = {
        "overall": {
            "count": len(results),
            "failed": sum(1 for row in results if row["status"] != "ok"),
            "wer": avg([row["wer"] for row in results]),
            "cer": avg([row["cer"] for row in results]),
            "numeric_accuracy": avg([row["numeric_accuracy"] for row in results]),
            "keyword_accuracy": avg([row["keyword_accuracy"] for row in results]),
            "numeric_confusion_accuracy": avg([row["numeric_confusion_accuracy"] for row in results]),
        },
        "by_category": {},
    }
    for category, rows in categories.items():
        summary["by_category"][category] = {
            "count": len(rows),
            "failed": sum(1 for row in rows if row["status"] != "ok"),
            "wer": avg([row["wer"] for row in rows]),
            "cer": avg([row["cer"] for row in rows]),
            "numeric_accuracy": avg([row["numeric_accuracy"] for row in rows]),
            "keyword_accuracy": avg([row["keyword_accuracy"] for row in rows]),
            "numeric_confusion_accuracy": avg([row["numeric_confusion_accuracy"] for row in rows]),
        }
    return summary


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python backend/scripts/evaluate_benchmarks.py backend/benchmarks/manifest.example.jsonl")
        return 1

    manifest_path = Path(sys.argv[1]).resolve()
    rows = load_manifest(manifest_path)
    results = [evaluate_one(row) for row in rows]
    print(json.dumps({"summary": summarize(results), "results": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
