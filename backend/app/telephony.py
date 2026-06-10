from __future__ import annotations

from .config import settings


def build_twiml_stream_response() -> str:
    stream_url = ""
    if settings.PUBLIC_BASE_URL:
        base = settings.PUBLIC_BASE_URL.rstrip("/")
        stream_url = f"{base}/api/telephony/media-stream"

    if stream_url:
        body = (
            "<Response>"
            "<Say>Welcome to CareCaller. Connecting your call for live transcription.</Say>"
            f"<Connect><Stream url=\"{stream_url}\" /></Connect>"
            "</Response>"
        )
    else:
        body = (
            "<Response>"
            "<Say>CareCaller voice webhook is enabled, but PUBLIC_BASE_URL is not configured yet.</Say>"
            "</Response>"
        )
    return body
