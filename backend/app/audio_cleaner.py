from __future__ import annotations

import io
import tempfile

import soundfile as sf
from pydub import AudioSegment, effects, silence

try:
    import noisereduce as nr
except ImportError:  # pragma: no cover - optional dependency during demo setup
    nr = None


def clean_audio_bytes(audio_bytes: bytes) -> bytes:
    """
    Input: WAV bytes
    Output: cleaned WAV bytes (mono, 16kHz, normalized, silence-trimmed, denoised)
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio = audio.high_pass_filter(120).low_pass_filter(3800)
    audio = effects.compress_dynamic_range(audio, threshold=-22.0, ratio=3.0, attack=5.0, release=50.0)
    audio = effects.normalize(audio)

    silence_threshold = max(audio.dBFS - 14, -42) if audio.dBFS != float("-inf") else -42
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=250,
        silence_thresh=silence_threshold,
        keep_silence=120,
    )
    if chunks:
        audio = sum(chunks[1:], chunks[0])

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        audio.export(temp_file.name, format="wav")
        data, sample_rate = sf.read(temp_file.name)

    if nr is not None and len(data) > 0:
        noise_sample = data[: min(len(data), max(int(sample_rate * 0.35), 1))]
        reduced = nr.reduce_noise(
            y=data,
            y_noise=noise_sample,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.9,
        )
    else:
        reduced = data

    output_buffer = io.BytesIO()
    sf.write(output_buffer, reduced, sample_rate, format="WAV")
    return output_buffer.getvalue()
