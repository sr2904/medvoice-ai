from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini-transcribe"
    ENABLE_LLM_POSTPROCESS: bool = False

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    ENABLE_GEMINI_MEDICAL_EXTRACTION: bool = False

    DEEPGRAM_MODEL: str = "nova-3-medical"
    DEEPGRAM_LANGUAGE: str = "en-US"

    TELEPHONY_SAMPLE_RATE: int = 8000
    TELEPHONY_ENCODING: str = "linear16"

    PUBLIC_BASE_URL: str | None = None
    TWILIO_ENABLE_VOICE_WEBHOOK: bool = False

    CORS_ALLOW_ORIGINS: str = "*"
    DB_URL: str = "sqlite:///./carecaller.db"
    WHISPER_MODEL: str = "base"


settings = Settings()
