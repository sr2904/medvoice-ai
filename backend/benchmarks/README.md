Benchmark manifest format for CareCaller telephony STT evaluation.

Each line is JSON:

```json
{
  "id": "sample-1",
  "category": "medical-noisy",
  "audio_path": "uploaded_audio/sample.wav",
  "reference_transcript": "I have been taking Allegra 10 milligrams and I still feel dizzy.",
  "patient_context": "Medications: Allegra. Conditions: seasonal allergies.",
  "medical_keywords": ["Allegra", "10", "dizzy"],
  "numeric_confusions": [{"expected": "15", "spoken_pair": "fifteen/fifty"}]
}
```

Run:

```bash
cd backend
.venv/bin/python scripts/evaluate_benchmarks.py benchmarks/manifest.example.jsonl
```

Reported metrics:
- `WER`
- `CER`
- `numeric_accuracy`
- `keyword_accuracy`
- `numeric_confusion_accuracy`

Recommended categories:
- `clean`
- `noisy`
- `accented`
- `medical-noisy`
- `telephony-narrowband`
