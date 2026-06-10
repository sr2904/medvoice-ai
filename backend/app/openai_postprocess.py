from __future__ import annotations

import json
from typing import Any

from .config import settings

SYSTEM_PROMPT = (
    "You clean up healthcare phone-call transcripts. Preserve meaning, fix obvious errors for medication names, dosages, and symptoms, and never invent facts. "
    "Return strict JSON with keys normalized_transcript and notes."
)


async def maybe_postprocess(transcript: str, patient_context: str | None = None) -> dict[str, Any]:
    if not settings.ENABLE_LLM_POSTPROCESS or not settings.OPENAI_API_KEY:
        return {"normalized_transcript": transcript, "notes": "LLM post-processing disabled"}

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return {"normalized_transcript": transcript, "notes": "openai package not installed"}

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.responses.create(
        model=settings.OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"transcript": transcript, "patient_context": patient_context})},
        ],
        text={"format": {"type": "json_object"}},
    )
    data = json.loads(response.output_text)
    return {"normalized_transcript": data.get("normalized_transcript", transcript), "notes": data.get("notes", "")}
