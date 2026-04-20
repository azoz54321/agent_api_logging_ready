import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from redis.asyncio import Redis

from .openai_client import ask_agent
from .schemas import AskRequest, AskResponse
from .settings import settings
from .supabase_client import log_unanswered_question


AGENT_VERSION = "admissions-agent-v4.1"
VALID_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_REASON_CODES = {
    "official_answer_found",
    "question_ambiguous",
    "missing_required_detail",
    "not_in_official_files",
    "outside_scope",
    "requires_human_action",
    "system_error",
}

REDIS_URL = os.getenv("REDIS_URL", "").strip()
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1200"))

FOLLOWUP_KEYWORDS = {
    "الفصل الأول",
    "الفصل الدراسي الأول",
    "الفصل الثاني",
    "الفصل الدراسي الثاني",
    "الفصل الثالث",
    "الفصل الدراسي الثالث",
    "الأول",
    "الثاني",
    "الثالث",
    "الصيفي",
    "انتظام",
    "انتساب",
    "بكالوريوس",
    "دبلوم",
    "طلاب",
    "طالبات",
    "1445",
    "1446",
    "1447",
    "1448",
    "1449",
    "2024",
    "2025",
    "2026",
    "2027",
}

NEW_QUESTION_PREFIXES = (
    "كيف",
    "متى",
    "هل",
    "كم",
    "وش",
    "ما",
    "ماذا",
    "وين",
    "أين",
    "ابي",
    "أبي",
    "ابغى",
    "أبغى",
    "اريد",
    "أريد",
    "لو سمحت",
)


def create_redis_client() -> Redis:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is missing in environment variables.")

    return Redis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = create_redis_client()
    await redis_client.ping()

    app.state.redis = redis_client
    app.state.session_ttl_seconds = SESSION_TTL_SECONDS

    yield

    await redis_client.aclose()


app = FastAPI(
    title="Bisha Admissions Agent API",
    version="0.5.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    redis_ok = False
    try:
        await app.state.redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {"ok": True, "redis_ok": redis_ok}


def generate_case_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:6].upper()
    return f"ESC-{timestamp}-{suffix}"


def normalize_confidence(value: str | None, default: str) -> str:
    candidate = (value or "").strip().lower()
    return candidate if candidate in VALID_CONFIDENCE else default


def has_year_marker(text: str) -> bool:
    return bool(re.search(r"(14\d{2}|20\d{2})", text))


def detect_missing_field(question_text: str) -> str | None:
    q = question_text.strip()

    semester_keywords = [
        "الحذف والإضافة",
        "الاعتذار",
        "الاختبارات النهائية",
        "إجازة منتصف الفصل",
        "الفصل القادم",
        "التحويل",
        "الزائر",
        "فترة الزائر",
        "التسجيل",
        "متى يبدأ التسجيل",
        "موعد التسجيل",
    ]
    semester_markers = [
        "الفصل الأول",
        "الفصل الدراسي الأول",
        "الفصل الثاني",
        "الفصل الدراسي الثاني",
        "الفصل الثالث",
        "الفصل الدراسي الثالث",
        "الصيفي",
    ]

    if any(keyword in q for keyword in semester_keywords) and not any(
        marker in q for marker in semester_markers
    ):
        return "semester"

    if any(keyword in q for keyword in semester_keywords) and not has_year_marker(q):
        return "year"

    return None


def normalize_reason_code(
    *,
    question_text: str,
    raw_reason: str | None,
    response_status: str,
) -> tuple[str, str | None, str | None]:
    reason = (raw_reason or "").strip().lower()
    missing_field = detect_missing_field(question_text)

    if reason in ALLOWED_REASON_CODES:
        return reason, None, missing_field

    if response_status == "not_found_officially":
        return (
            "not_in_official_files",
            "No explicit official answer was found in the approved files.",
            None,
        )

    out_of_scope_markers = [
        "مواقف",
        "موقف",
        "مبنى",
        "المبنى",
        "مكتب",
        "مقصف",
        "كافتيريا",
        "سكن",
        "بوابة",
        "الأمن الجامعي",
        "موقع المبنى",
        "وين المبنى",
    ]
    if any(marker in question_text for marker in out_of_scope_markers):
        return (
            "outside_scope",
            "Question appears outside the official admissions and registration scope.",
            None,
        )

    if response_status == "escalate":
        return (
            "requires_human_action",
            f"Mapped from raw agent reason: {raw_reason}",
            None,
        )

    return (
        "system_error",
        f"Unexpected response status: {response_status}; raw reason: {raw_reason}",
        None,
    )


def build_session_identity(req: AskRequest) -> tuple[str, str]:
    channel = str(req.channel or "unknown").strip().lower()
    user_id = str(req.user_id or "anonymous").strip()
    return channel, user_id


def session_key(channel: str, user_id: str) -> str:
    return f"session:{channel}:{user_id}"


def default_session(channel: str, user_id: str) -> dict:
    return {
        "channel": channel,
        "user_id": user_id,
        "recent_messages": [],
        "pending_question": None,
        "pending_intent": None,
        "missing_fields": [],
        "last_complete_question": None,
        "updated_at": None,
    }


async def get_session(channel: str, user_id: str) -> dict:
    raw = await app.state.redis.get(session_key(channel, user_id))
    if not raw:
        return default_session(channel, user_id)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default_session(channel, user_id)


async def save_session(channel: str, user_id: str, session: dict) -> None:
    session["updated_at"] = datetime.utcnow().isoformat()
    await app.state.redis.set(
        session_key(channel, user_id),
        json.dumps(session, ensure_ascii=False),
        ex=app.state.session_ttl_seconds,
    )


async def append_recent_message(
    channel: str,
    user_id: str,
    role: str,
    text: str,
    keep_last: int = 6,
) -> dict:
    session = await get_session(channel, user_id)
    session.setdefault("recent_messages", [])
    session["recent_messages"].append({"role": role, "text": text})
    session["recent_messages"] = session["recent_messages"][-keep_last:]
    await save_session(channel, user_id, session)
    return session


async def set_pending(
    channel: str,
    user_id: str,
    question: str,
    missing_fields: list[str] | None = None,
    pending_intent: str | None = None,
) -> dict:
    session = await get_session(channel, user_id)
    session["pending_question"] = question
    session["pending_intent"] = pending_intent
    session["missing_fields"] = missing_fields or []
    await save_session(channel, user_id, session)
    return session


async def clear_pending(channel: str, user_id: str) -> dict:
    session = await get_session(channel, user_id)
    session["pending_question"] = None
    session["pending_intent"] = None
    session["missing_fields"] = []
    await save_session(channel, user_id, session)
    return session


async def set_last_complete_question(
    channel: str,
    user_id: str,
    question: str,
) -> dict:
    session = await get_session(channel, user_id)
    session["last_complete_question"] = question
    await save_session(channel, user_id, session)
    return session


def is_followup_to_pending(text: str, session: dict) -> bool:
    pending_question = session.get("pending_question")
    if not pending_question:
        return False

    normalized = text.strip()
    if not normalized:
        return False

    if "?" in normalized or "؟" in normalized:
        return False

    if normalized.startswith(NEW_QUESTION_PREFIXES):
        return False

    if normalized.isdigit():
        return True

    if has_year_marker(normalized):
        return True

    if any(keyword in normalized for keyword in FOLLOWUP_KEYWORDS):
        return True

    if len(normalized.split()) <= 4 and len(normalized) <= 40:
        return True

    return False


def merge_pending_with_followup(session: dict, user_text: str) -> str:
    pending_question = (session.get("pending_question") or "").strip().rstrip("؟?.، ")
    followup_text = user_text.strip()
    if not pending_question:
        return followup_text
    return f"{pending_question} {followup_text}".strip()


def log_non_answered_case(
    *,
    req: AskRequest,
    question_text: str,
    result: dict,
    response_status: str,
    response_category: str,
    raw_reason: str | None,
    force_case_id: str | None = None,
) -> tuple[str, str]:
    case_id = force_case_id or generate_case_id()
    reason_code, reason_detail, missing_field = normalize_reason_code(
        question_text=question_text,
        raw_reason=raw_reason,
        response_status=response_status,
    )

    try:
        log_unanswered_question(
            question_text=question_text,
            channel=req.channel,
            user_id=req.user_id,
            case_id=case_id,
            status=response_status,
            reason=reason_code,
            reason_code=reason_code,
            reason_detail=reason_detail,
            missing_field=missing_field,
            category=response_category,
            matched_docs_count=result.get("matched_docs_count"),
            top_score=result.get("top_score"),
            agent_version=AGENT_VERSION,
        )
    except Exception as exc:
        print(f"[WARN] Failed to log non-answered question: {exc}")

    return case_id, reason_code


@app.post("/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    if settings.ask_api_key and x_api_key != settings.ask_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    raw_user_text = req.message_text.strip()
    channel, user_id = build_session_identity(req)

    session = await get_session(channel, user_id)
    current_question = raw_user_text

    if is_followup_to_pending(raw_user_text, session):
        current_question = merge_pending_with_followup(session, raw_user_text)

    await append_recent_message(channel, user_id, "user", raw_user_text)

    try:
        result = await run_in_threadpool(ask_agent, current_question)
    except Exception as exc:
        case_id = generate_case_id()
        reason_code = "system_error"
        reason_detail = str(exc)

        try:
            await run_in_threadpool(
                log_unanswered_question,
                question_text=current_question,
                channel=req.channel,
                user_id=req.user_id,
                case_id=case_id,
                status="system_error",
                reason=reason_code,
                reason_code=reason_code,
                reason_detail=reason_detail,
                missing_field=None,
                category="system",
                matched_docs_count=None,
                top_score=None,
                agent_version=AGENT_VERSION,
            )
        except Exception as log_exc:
            print(f"[WARN] Failed to log system error case: {log_exc}")

        reply_text = "تعذر إكمال الطلب الآن، وتم تحويله للمراجعة."
        await append_recent_message(channel, user_id, "assistant", reply_text)

        return AskResponse(
            status="escalate",
            reply_text=reply_text,
            category="system",
            official_source_used=False,
            source_name=None,
            reason=reason_code,
            confidence="low",
            case_id=case_id,
        )

    response_status = result.get("status", "escalate")
    if response_status not in {
        "answered",
        "needs_clarification",
        "not_found_officially",
        "escalate",
    }:
        response_status = "escalate"

    response_text = result.get("reply_text") or "تم تحويل استفسارك إلى الموظف المختص."
    response_category = result.get("category") or "other"
    raw_reason = result.get("reason")
    official_source_used = bool(result.get("official_source_used", False))
    source_name = result.get("source_name")
    confidence = result.get("confidence")

    if response_status == "answered":
        await clear_pending(channel, user_id)
        await set_last_complete_question(channel, user_id, current_question)
        await append_recent_message(channel, user_id, "assistant", response_text)

        return AskResponse(
            status="answered",
            reply_text=response_text,
            category=response_category,
            official_source_used=official_source_used,
            source_name=source_name,
            reason="official_answer_found",
            confidence=normalize_confidence(confidence, "high"),
            case_id=None,
        )

    if response_status == "needs_clarification":
        reason_code = (raw_reason or "").strip().lower()
        missing_field = detect_missing_field(current_question)

        if reason_code not in ALLOWED_REASON_CODES:
            reason_code = "missing_required_detail" if missing_field else "question_ambiguous"

        missing_fields = [missing_field] if missing_field else []

        await set_pending(
            channel=channel,
            user_id=user_id,
            question=current_question,
            missing_fields=missing_fields,
            pending_intent=reason_code,
        )
        await append_recent_message(channel, user_id, "assistant", response_text)

        return AskResponse(
            status="needs_clarification",
            reply_text=response_text,
            category=response_category,
            official_source_used=False,
            source_name=None,
            reason=reason_code,
            confidence=normalize_confidence(confidence, "medium"),
            case_id=None,
        )

    if response_status == "not_found_officially":
        await clear_pending(channel, user_id)
        _, reason_code = await run_in_threadpool(
            log_non_answered_case,
            req=req,
            question_text=current_question,
            result=result,
            response_status=response_status,
            response_category=response_category,
            raw_reason=raw_reason,
        )
        await append_recent_message(channel, user_id, "assistant", response_text)

        return AskResponse(
            status="not_found_officially",
            reply_text=response_text,
            category=response_category,
            official_source_used=False,
            source_name=None,
            reason=reason_code,
            confidence=normalize_confidence(confidence, "low"),
            case_id=None,
        )

    await clear_pending(channel, user_id)
    case_id, reason_code = await run_in_threadpool(
        log_non_answered_case,
        req=req,
        question_text=current_question,
        result=result,
        response_status="escalate",
        response_category=response_category,
        raw_reason=raw_reason,
    )
    await append_recent_message(channel, user_id, "assistant", response_text)

    return AskResponse(
        status="escalate",
        reply_text=response_text,
        category=response_category,
        official_source_used=False,
        source_name=None,
        reason=reason_code,
        confidence=normalize_confidence(confidence, "low"),
        case_id=case_id,
    )