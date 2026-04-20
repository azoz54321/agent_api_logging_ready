import os


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "agent-api")
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "1200"))


settings = Settings()