# Notary Everyday - ID Verify AI

A hackathon-ready codebase for the Notary Everyday x VillageHacks 2026 challenge.

## What this demo does

- **Level 1:** Authenticity-oriented presentation check using visual heuristics for `genuine`, `screen`, and `print`
- **Level 2:** OCR-based structured field extraction into JSON
- **Level 3:** Cross-document matching across important identity fields
- **Partial Level 4:** Compliance and edge-case flags such as expired ID, expiring soon, and low-confidence extraction

## Important honesty note

This build is designed to be a **strong working demo** with clear explainability and a polished flow. The authenticity classifier is currently **heuristic-based**, not a trained deep learning model. That makes it easy to run and demo quickly. If you want to push it further, connect your KID34K-trained classifier where noted below.

## Project structure

```text
notaryeveryday_ai/
├── backend/
│   ├── app/
│   │   ├── services/
│   │   ├── config.py
│   │   ├── main.py
│   │   └── models.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── README.md
```

## Quick start

### 1. Create a virtual environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

From the **project root**:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Recommended demo flow

1. Upload one ID image and show the fraud label, compliance flags, and JSON output.
2. Explain that the system is **language-agnostic** because OCR text is handled structurally.
3. Upload a second document and show field-by-field matching.
4. End with the compliance rule explanation and human review recommendation.

## Where to add a trained KID34K model

If you train a classifier on the KID34K dataset, swap it into:

- `backend/app/services/fraud_checks.py`

Best approach:

- load your model once at startup
- pass the image into the model inside `analyze_id`
- return `genuine`, `screen`, or `print`
- keep the current rule-based explanations as your explainability layer

## Why this is strong for judging

- clean end-to-end working demo
- machine-readable JSON output
- clear edge-case handling
- easy to explain in 3 minutes
- visually polished enough for presentation

## Suggested hackathon talking points

- “We built a language-agnostic notary verification assistant.”
- “It checks presentation fraud, extracts fields into JSON, compares documents, and flags compliance risk.”
- “We designed the system to keep a human notary in the loop instead of pretending to replace them.”

