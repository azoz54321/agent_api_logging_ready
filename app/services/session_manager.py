import json
from datetime import datetime, timezone
from typing import Any, Optional


class SessionManager:
    def __init__(self, redis_client, ttl_seconds: int = 1200):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    def _key(self, channel: str, chat_id: str | int) -> str:
        return f"session:{channel}:{chat_id}"

    async def get_session(self, channel: str, chat_id: str | int) -> dict[str, Any]:
        raw = await self.redis.get(self._key(channel, chat_id))
        if not raw:
            return {
                "channel": channel,
                "chat_id": str(chat_id),
                "recent_messages": [],
                "pending_question": None,
                "pending_intent": None,
                "missing_fields": [],
                "last_complete_question": None,
                "updated_at": None,
            }

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "channel": channel,
                "chat_id": str(chat_id),
                "recent_messages": [],
                "pending_question": None,
                "pending_intent": None,
                "missing_fields": [],
                "last_complete_question": None,
                "updated_at": None,
            }

    async def save_session(self, channel: str, chat_id: str | int, session: dict[str, Any]) -> None:
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        key = self._key(channel, chat_id)
        await self.redis.set(key, json.dumps(session, ensure_ascii=False), ex=self.ttl_seconds)

    async def append_recent_message(
        self,
        channel: str,
        chat_id: str | int,
        role: str,
        text: str,
        keep_last: int = 6,
    ) -> dict[str, Any]:
        session = await self.get_session(channel, chat_id)
        session.setdefault("recent_messages", [])
        session["recent_messages"].append({"role": role, "text": text})
        session["recent_messages"] = session["recent_messages"][-keep_last:]
        await self.save_session(channel, chat_id, session)
        return session

    async def set_pending(
        self,
        channel: str,
        chat_id: str | int,
        question: str,
        missing_fields: list[str],
        pending_intent: Optional[str] = None,
    ) -> dict[str, Any]:
        session = await self.get_session(channel, chat_id)
        session["pending_question"] = question
        session["pending_intent"] = pending_intent
        session["missing_fields"] = missing_fields
        await self.save_session(channel, chat_id, session)
        return session

    async def clear_pending(self, channel: str, chat_id: str | int) -> dict[str, Any]:
        session = await self.get_session(channel, chat_id)
        session["pending_question"] = None
        session["pending_intent"] = None
        session["missing_fields"] = []
        await self.save_session(channel, chat_id, session)
        return session

    async def set_last_complete_question(
        self,
        channel: str,
        chat_id: str | int,
        question: str,
    ) -> dict[str, Any]:
        session = await self.get_session(channel, chat_id)
        session["last_complete_question"] = question
        await self.save_session(channel, chat_id, session)
        return session

    def is_followup_to_pending(self, text: str, session: dict[str, Any]) -> bool:
        pending_question = session.get("pending_question")
        if not pending_question:
            return False

        normalized = text.strip()
        if not normalized:
            return False

        followup_keywords = {
            "الفصل الأول",
            "الفصل الثاني",
            "الصيفي",
            "1447",
            "1448",
            "انتظام",
            "انتساب",
            "بكالوريوس",
            "دبلوم",
            "طلاب",
            "طالبات",
        }

        # رسالة قصيرة جدًا
        if len(normalized) <= 20:
            return True

        # سنة فقط أو أرقام فقط
        if normalized.isdigit():
            return True

        # كلمة/عبارة معروفة كتكميل
        if normalized in followup_keywords:
            return True

        return False

    def merge_pending_with_followup(self, session: dict[str, Any], user_text: str) -> str:
        pending_question = (session.get("pending_question") or "").strip()
        user_text = user_text.strip()

        if not pending_question:
            return user_text

        return f"{pending_question} {user_text}".strip()