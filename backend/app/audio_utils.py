from __future__ import annotations

import io
import tempfile
from pathlib import Path

from pydub import AudioSegment


def convert_to_telephony_wav(audio_bytes: bytes, filename: str | None = None) -> bytes:
    suffix = Path(filename).suffix.lower().lstrip(".") if filename else None

    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=suffix or None)
    except Exception as exc:
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{suffix or 'audio'}") as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                audio = AudioSegment.from_file(temp_file.name, format=suffix or None)
        except Exception:
            raise ValueError(f"Could not read uploaded audio. Format={suffix or 'unknown'}. Error: {exc}")

    # Convert every upload into a consistent mono 16k WAV before the cleaning stage.
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    out = io.BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


def save_audio_bytes(path: Path, audio_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio_bytes)
