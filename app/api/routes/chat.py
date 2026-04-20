from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()


class ChatRequest(BaseModel):
    chat_id: str
    text: str
    channel: str = "telegram"


class ChatResponse(BaseModel):
    status: str
    reply_text: str


def detect_missing_fields(question: str) -> list[str]:
    question = question.strip()

    if "متى يبدأ التسجيل" in question:
        missing = []
        if "الفصل" not in question:
            missing.append("term")
        if "144" not in question:
            missing.append("year")
        return missing

    return []


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest):
    session_manager = request.app.state.session_manager

    session = await session_manager.get_session(payload.channel, payload.chat_id)

    current_question = payload.text.strip()

    if session_manager.is_followup_to_pending(current_question, session):
        current_question = session_manager.merge_pending_with_followup(session, current_question)

    missing_fields = detect_missing_fields(current_question)

    await session_manager.append_recent_message(payload.channel, payload.chat_id, "user", payload.text)

    if missing_fields:
        await session_manager.set_pending(
            channel=payload.channel,
            chat_id=payload.chat_id,
            question=current_question,
            missing_fields=missing_fields,
            pending_intent="ask_registration_date",
        )

        reply_text = "سؤالك ناقص. حدّد الفصل والسنة."
        await session_manager.append_recent_message(payload.channel, payload.chat_id, "assistant", reply_text)

        return ChatResponse(status="needs_clarification", reply_text=reply_text)

    # هنا تضع استدعاء الوكيل الحقيقي/البحث في الملفات
    reply_text = f"تم فهم سؤالك النهائي: {current_question}"

    await session_manager.clear_pending(payload.channel, payload.chat_id)
    await session_manager.set_last_complete_question(payload.channel, payload.chat_id, current_question)
    await session_manager.append_recent_message(payload.channel, payload.chat_id, "assistant", reply_text)

    return ChatResponse(status="answered", reply_text=reply_text)