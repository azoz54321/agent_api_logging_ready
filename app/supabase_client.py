from functools import lru_cache
from typing import Any

from supabase import Client, create_client

from .settings import settings


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_secret_key)


def log_unanswered_question(
    *,
    question_text: str,
    channel: str,
    user_id: str | None,
    case_id: str,
    status: str,
    reason: str,
    reason_code: str,
    reason_detail: str | None,
    missing_field: str | None,
    category: str,
    matched_docs_count: int | None,
    top_score: float | None,
    agent_version: str,
) -> None:
    payload: dict[str, Any] = {
        "question_text": question_text,
        "channel": channel,
        "user_id": user_id,
        "case_id": case_id,
        "status": status,
        "reason": reason,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "missing_field": missing_field,
        "category": category,
        "matched_docs_count": matched_docs_count,
        "top_score": top_score,
        "agent_version": agent_version,
    }

    clean_payload = {key: value for key, value in payload.items() if value is not None}
    get_supabase().table("unanswered_questions").insert(clean_payload).execute()
