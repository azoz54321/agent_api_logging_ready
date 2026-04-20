from redis.asyncio import Redis
from app.core.config import settings


def create_redis_client() -> Redis:
    if not settings.REDIS_URL:
        raise ValueError("REDIS_URL is missing in environment variables.")

    return Redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )