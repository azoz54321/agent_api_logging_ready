from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException

from .openai_client import ask_agent
from .schemas import AskRequest, AskResponse
from .settings import settings
from .supabase_client import log_unanswered_question

app = FastAPI(title="Bisha Admissions Agent API", version="0.4.1")

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


@app.get("/health")
def health():
    return {"ok": True}


def generate_case_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:6].upper()
    return f"ESC-{timestamp}-{suffix}"


def normalize_confidence(value: str | None, default: str) -> str:
    candidate = (value or "").strip().lower()
    return candidate if candidate in VALID_CONFIDENCE else default


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
    ]
    semester_markers = [
        "الفصل الأول",
        "الفصل الدراسي الأول",
        "الفصل الثاني",
        "الفصل الدراسي الثاني",
        "الفصل الثالث",
        "الفصل الدراسي الثالث",
    ]

    if any(keyword in q for keyword in semester_keywords) and not any(
        marker in q for marker in semester_markers
    ):
        return "semester"

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


def log_non_answered_case(
    *,
    req: AskRequest,
    result: dict,
    response_status: str,
    response_category: str,
    raw_reason: str | None,
    force_case_id: str | None = None,
) -> tuple[str, str]:
    case_id = force_case_id or generate_case_id()
    reason_code, reason_detail, missing_field = normalize_reason_code(
        question_text=req.message_text,
        raw_reason=raw_reason,
        response_status=response_status,
    )

    try:
        log_unanswered_question(
            question_text=req.message_text,
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
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    if settings.ask_api_key and x_api_key != settings.ask_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        result = ask_agent(req.message_text)
    except Exception as exc:
        case_id = generate_case_id()
        reason_code = "system_error"
        reason_detail = str(exc)

        try:
            log_unanswered_question(
                question_text=req.message_text,
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

        return AskResponse(
            status="escalate",
            reply_text="تعذر إكمال الطلب الآن، وتم تحويله للمراجعة.",
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
        if reason_code not in ALLOWED_REASON_CODES:
            missing_field = detect_missing_field(req.message_text)
            reason_code = "missing_required_detail" if missing_field else "question_ambiguous"

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
        _, reason_code = log_non_answered_case(
            req=req,
            result=result,
            response_status=response_status,
            response_category=response_category,
            raw_reason=raw_reason,
        )
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

    case_id, reason_code = log_non_answered_case(
        req=req,
        result=result,
        response_status="escalate",
        response_category=response_category,
        raw_reason=raw_reason,
    )

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