# MedVoice AI — Healthcare Voice Triage System

<div align="center">

**Real-time patient call analysis — speech to structured clinical insights**

[![GitHub](https://img.shields.io/badge/GitHub-sr2904%2Fmedvoice--ai-blue?logo=github)](https://github.com/sr2904/medvoice_ai)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red?logo=youtube)](https://youtu.be/zr2TYypYerY)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-green?logo=fastapi)
![Deepgram](https://img.shields.io/badge/STT-Deepgram-purple)
![Gemini](https://img.shields.io/badge/AI-Gemini-orange?logo=google)

</div>

---

## What This Is

MedVoice AI is a real-time healthcare call analysis system that converts patient audio into structured clinical insights. It helps healthcare teams understand patient concerns and prioritize follow-ups by automatically extracting medications, dosages, symptoms, and urgency signals from phone calls.

---

## The Problem

Healthcare providers receive a high volume of patient calls daily. These calls contain critical information about medications, symptoms, and side effects — but manual review is slow, details get missed, and urgent cases can fall through the cracks.

---

## Full Pipeline

```
Patient speaks into phone / uploads audio
                │
                ▼
┌───────────────────────────────────────────────────┐
│            Audio Preprocessing                    │
│  pydub + ffmpeg                                   │
│  • Convert any format → 16kHz mono WAV            │
│  • Normalize volume, trim silence                 │
│  • Background noise reduction                     │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│         Speech-to-Text (Deepgram API)             │
│  stt.py                                           │
│  • Medical STT model (nova-2-medical)             │
│  • Speaker diarization → isolate patient voice    │
│  • Smart formatting, numerics, punctuation        │
│  • Keyterm boosting for medications + symptoms    │
│  • Utterance-level speaker scoring               │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│         Clinical Entity Extraction                │
│  medical_logic.py + gemini_medical.py             │
│                                                   │
│  Step 1 — Medication correction                   │
│    Fuzzy-match OCR errors against known med list  │
│    SequenceMatcher similarity ≥ 0.72 threshold    │
│                                                   │
│  Step 2 — Gemini AI extraction (primary)          │
│    Normalize transcript (clinical summary)        │
│    Extract: Drug Name, Dosage, Symptom, Side Eff  │
│    Detect urgency language                        │
│    Temperature 0.1 (deterministic outputs)        │
│                                                   │
│  Step 3 — Deterministic fallback (always runs)    │
│    Regex dosage extraction (mg, mcg, ml, g)       │
│    Keyword symptom matching                       │
│    Known medication list matching                 │
│                                                   │
│  Step 4 — Safety merge                            │
│    Gemini + fallback deduplicated                 │
│    Fallback priority raised if more severe        │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│              Triage Engine                        │
│  medical_logic.py → decide_priority()             │
│                                                   │
│  Urgent  → life-threatening language detected     │
│            "can't breathe", "I'm dying", etc.     │
│                                                   │
│  High    → severe symptom detected               │
│            chest pain, shortness of breath,       │
│            fainting, swelling, anaphylaxis        │
│            OR urgent language + clinical entity   │
│                                                   │
│  Medium  → medication side effect reported        │
│            drug + symptom together                │
│            guidance question asked               │
│            2+ symptoms detected                  │
│                                                   │
│  Low     → no urgent concern detected            │
│            document and monitor                  │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│           Patient Timeline (SQLite)               │
│  models.py — Patient + CallRecord                 │
│  • Every call stored with full transcript         │
│  • Extracted entities saved as JSON               │
│  • Patient profile auto-updated from call         │
│  • Timeline accessible per patient                │
└───────────────────────────────────────────────────┘
```

---

## Example

**Input audio:**
> *"Hi, I started taking Benadryl 10 mg yesterday and now I feel dizzy."*

**Output:**
```json
{
  "normalized_transcript": "Patient started taking Benadryl 10 mg yesterday and is now experiencing dizziness.",
  "entities": [
    { "label": "Drug Name", "value": "Benadryl" },
    { "label": "Dosage",    "value": "10 mg" },
    { "label": "Symptom",   "value": "dizziness" }
  ],
  "decision": {
    "priority": "Medium",
    "title": "Medication symptom follow-up",
    "description": "Patient mentioned both a medication and a symptom. Call should be reviewed."
  }
}
```

---

## Triage Rules

| Priority | Trigger | Action |
|---|---|---|
| **Urgent** | "can't breathe", "I'm dying", "passed out", "unresponsive" | Emergency escalation — direct to 911 |
| **High** | Chest pain, shortness of breath, fainting, swelling, anaphylaxis | Escalate to clinician immediately |
| **High** | Urgent language ("right now", "emergency") + any clinical entity | Urgent clinician review |
| **Medium** | Medication side effect reported | Route to nurse review |
| **Medium** | Drug + symptom together | Medication symptom follow-up |
| **Medium** | 2+ symptoms detected | Symptom follow-up needed |
| **Medium** | Guidance question asked | Medication guidance requested |
| **Low** | No urgent signals | Monitor and document |

---

## Speaker Isolation Logic

The STT layer uses Deepgram's diarization to identify multiple speakers and isolate the patient's voice:

```
All utterances from call
         │
         ▼
Score each speaker utterance:
  +word_count        base score
  +4 per I/my/me     patient language signals
  +5 per clinical    medication/symptom terms
  +6 per patient     name terms from context
  +6 for "should I"  guidance questions
  +6 for numeric     dosage patterns (10 mg)
         │
         ▼
Primary speaker = highest total score
Filter utterances below clinical threshold
Stitch patient-only transcript
```

---

## Gemini AI Prompt Design

The Gemini integration uses a carefully engineered system prompt to ensure safe, conservative clinical extraction:

- **Never invents facts** — only extracts grounded information
- **Resolves near-duplicates** — "dizzy" and "dizziness" collapse to one entity
- **Separates current dose from requested dose** — "taking 1mg, want 1mg more" → two distinct facts
- **Urgency detection** — life-threatening language forces Urgent priority
- **Reassurance filtering** — "feeling fine", "doing well" are NOT extracted as symptoms
- **Temperature 0.1** — near-deterministic outputs for clinical reliability
- **Fallback safety** — if Gemini is rate-limited or unavailable, deterministic rules take over and priority can only be raised, never lowered

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Audio Processing | pydub, ffmpeg |
| Speech-to-Text | Deepgram API (nova-2-medical model) |
| AI Extraction | Google Gemini API |
| Deterministic Fallback | Custom regex + keyword matching |
| Fuzzy Matching | Python difflib SequenceMatcher |
| Database | SQLite + SQLAlchemy ORM |
| Frontend | HTML, CSS, Vanilla JavaScript |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/capabilities` | GET | Feature flags and config |
| `/api/patients` | GET | List all patients |
| `/api/patients` | POST | Create a patient |
| `/api/patients/{id}/timeline` | GET | Full call history for a patient |
| `/api/transcribe` | POST | Full pipeline: audio → triage decision |
| `/api/transcribe-preview` | POST | Quick STT preview without saving |
| `/api/telephony/voice` | POST | Twilio voice webhook |

---

## File Structure

```
medvoice_ai/
├── backend/
│   ├── app/
│   │   ├── main.py              — FastAPI routes and pipeline orchestration
│   │   ├── medical_logic.py     — Entity extraction, normalization, triage engine
│   │   ├── gemini_medical.py    — Gemini AI integration and prompt engineering
│   │   ├── stt.py               — Deepgram STT + speaker isolation logic
│   │   ├── audio_cleaner.py     — Noise reduction and audio normalization
│   │   ├── audio_utils.py       — Format conversion (pydub + ffmpeg)
│   │   ├── models.py            — SQLAlchemy Patient + CallRecord models
│   │   ├── schemas.py           — Pydantic request/response schemas
│   │   ├── database.py          — SQLite engine and session
│   │   ├── config.py            — Settings and environment variables
│   │   ├── seed.py              — Demo patient seeding
│   │   └── telephony.py         — Twilio TwiML response builder
│   ├── benchmarks/              — WER/CER/numeric accuracy benchmarks
│   ├── scripts/                 — Common Voice benchmark tools
│   └── requirements.txt
├── frontend/
│   ├── index.html               — UI with live demo, timeline, architecture
│   └── styles.css
└── README.md
```

---

## Running Locally

```bash
# Clone
git clone https://github.com/sr2904/medvoice_ai.git
cd medvoice_ai

# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add DEEPGRAM_API_KEY to .env
uvicorn app.main:app --reload --port 8001

# Frontend (separate terminal)
cd frontend
python3 -m http.server 8080
```

Open `http://127.0.0.1:8080`

**Required:** Only `DEEPGRAM_API_KEY` is needed for the core pipeline. Gemini is optional and disabled by default.

---

## Future Work

- Integration with Electronic Health Records (EHR)
- Multilingual patient support
- Real-time WebSocket audio streaming via Twilio
- Advanced clinician review dashboard
