from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-5.4"
    openai_vector_store_id: str | None = None

    supabase_url: str
    supabase_secret_key: str

    ask_api_key: str | None = None

    redis_url: str | None = None
    redis_ca_cert: str | None = None
    session_ttl_seconds: int = 1200

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
