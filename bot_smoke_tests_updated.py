"""
Updated smoke tests for the Bisha Admissions Agent API.

Expected policy:
- Time questions without explicit year/academic year => needs_clarification
- Time questions with unsupported explicit year => not_found_officially
- Clear manual/human cases => escalate
- Out-of-scope building/facility questions => escalate

Usage:
    1) Optional environment variables:
       AGENT_BASE_URL   default: http://127.0.0.1:8000
       AGENT_API_KEY    optional

    2) Run:
py bot_smoke_tests_updated.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import requests


BASE_URL = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("AGENT_API_KEY", "").strip()
ASK_ENDPOINT = f"{BASE_URL}/v1/ask"
TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class TestCase:
    name: str
    question: str
    expected_status: str
    accepted_reasons: tuple[str, ...] = ()
    expected_reply_contains: tuple[str, ...] = ()


TEST_CASES: list[TestCase] = [
    TestCase(
        name="clarify_registration_start_missing_year",
        question="متى يبدأ تسجيل الفصل الأول؟",
        expected_status="needs_clarification",
        accepted_reasons=("missing_required_detail", "question_ambiguous"),
        expected_reply_contains=("السنة", "العام الجامعي"),
    ),
    TestCase(
        name="clarify_drop_add_sem1_missing_year",
        question="متى ينتهي الحذف والإضافة في الفصل الدراسي الأول؟",
        expected_status="needs_clarification",
        accepted_reasons=("missing_required_detail", "question_ambiguous"),
        expected_reply_contains=("السنة", "العام الجامعي"),
    ),
    TestCase(
        name="answered_registration_start_1447",
        question="متى يبدأ تسجيل الفصل الأول لعام 1447؟",
        expected_status="answered",
        accepted_reasons=("official_answer_found",),
    ),
    TestCase(
        name="answered_source_request_1447",
        question="متى يبدأ تسجيل الفصل الأول لعام 1447؟ اذكر المصدر الرسمي فقط",
        expected_status="answered",
        accepted_reasons=("official_answer_found",),
        expected_reply_contains=("التقويم الأكاديمي",),
    ),
    TestCase(
        name="clarify_withdrawal_missing_year_and_semester",
        question="متى آخر يوم للاعتذار؟",
        expected_status="needs_clarification",
        accepted_reasons=("missing_required_detail", "question_ambiguous"),
        expected_reply_contains=("السنة", "الفصل"),
    ),
    TestCase(
        name="clarify_finals_missing_year_and_semester",
        question="متى تبدأ الاختبارات النهائية؟",
        expected_status="needs_clarification",
        accepted_reasons=("missing_required_detail", "question_ambiguous"),
        expected_reply_contains=("السنة", "الفصل"),
    ),
    TestCase(
        name="clarify_visitor_missing_year_and_semester",
        question="متى فترة الزائر؟",
        expected_status="needs_clarification",
        accepted_reasons=("missing_required_detail", "question_ambiguous"),
        expected_reply_contains=("السنة", "الفصل"),
    ),
    TestCase(
        name="not_found_registration_start_1448",
        question="متى يبدأ تسجيل الفصل الأول لعام 1448؟",
        expected_status="not_found_officially",
        accepted_reasons=("not_in_official_files",),
        expected_reply_contains=("لا يوجد لدي موعد معتمد", "التقويم الأكاديمي المتاح"),
    ),
    TestCase(
        name="escalate_parking_count_out_of_scope",
        question="كم عدد مواقف السيارات خلف العمادة؟",
        expected_status="escalate",
        accepted_reasons=("outside_scope",),
        expected_reply_contains=("خارج نطاق اختصاص",),
    ),
    TestCase(
        name="not_found_fixed_acceptance_rate",
        question="كم نسبة القبول الثابتة في الأمن السيبراني هذا العام؟",
        expected_status="not_found_officially",
        accepted_reasons=("not_in_official_files",),
    ),
    TestCase(
        name="not_found_actual_last_year_cutoff",
        question="كم أقل نسبة تم قبولها فعليًا العام الماضي في تخصص الطب؟",
        expected_status="not_found_officially",
        accepted_reasons=("not_in_official_files",),
    ),
    TestCase(
        name="escalate_individual_case",
        question="تم رفض طلبي وأريد إعادة النظر في حالتي الفردية.",
        expected_status="escalate",
        accepted_reasons=("requires_human_action",),
    ),
    TestCase(
        name="escalate_manual_record_change",
        question="أبغى تعديل بياناتي يدويًا في النظام.",
        expected_status="escalate",
        accepted_reasons=("requires_human_action",),
    ),
    TestCase(
        name="escalate_out_of_scope_building",
        question="وين مبنى الأمن الجامعي؟",
        expected_status="escalate",
        accepted_reasons=("outside_scope",),
        expected_reply_contains=("خارج نطاق اختصاص",),
    ),
]


def build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    return headers


def post_question(question: str) -> requests.Response:
    payload = {
        "message_text": question,
        "channel": "manual_test",
        "user_id": "smoke-test-user",
    }
    return requests.post(
        ASK_ENDPOINT,
        json=payload,
        headers=build_headers(),
        timeout=TIMEOUT_SECONDS,
    )


def text_contains_any(text: str, snippets: Iterable[str]) -> bool:
    if not snippets:
        return True
    return any(snippet in text for snippet in snippets)


def run_test(case: TestCase) -> tuple[bool, str]:
    try:
        response = post_question(case.question)
    except Exception as exc:
        return False, f"request_error={exc}"

    if response.status_code != 200:
        return False, f"http_status={response.status_code}, body={response.text}"

    try:
        data = response.json()
    except Exception:
        return False, f"invalid_json_body={response.text}"

    actual_status = data.get("status")
    actual_reason = data.get("reason")
    reply_text = data.get("reply_text", "")
    category = data.get("category")
    case_id = data.get("case_id")

    errors: list[str] = []

    if actual_status != case.expected_status:
        errors.append(f"expected_status={case.expected_status}, actual_status={actual_status}")

    if case.accepted_reasons and actual_reason not in case.accepted_reasons:
        errors.append(
            f"expected_reason_in={case.accepted_reasons}, actual_reason={actual_reason}"
        )

    if case.expected_reply_contains and not text_contains_any(reply_text, case.expected_reply_contains):
        errors.append(
            f"reply_text_missing_expected_snippet={case.expected_reply_contains}"
        )

    if not isinstance(reply_text, str) or not reply_text.strip():
        errors.append("reply_text_empty")

    if actual_status == "escalate" and not case_id:
        errors.append("case_id_missing_for_escalate")

    if actual_status != "escalate" and case_id is not None:
        errors.append("case_id_should_be_null_for_non_escalate")

    detail = (
        f"status={actual_status}, reason={actual_reason}, category={category}, "
        f"case_id={case_id}, reply={reply_text}"
    )

    if errors:
        return False, " | ".join(errors) + " | " + detail

    return True, detail


def main() -> int:
    print("=" * 80)
    print(f"Testing endpoint: {ASK_ENDPOINT}")
    print(f"API key sent: {'yes' if API_KEY else 'no'}")
    print("=" * 80)

    passed = 0
    failed = 0

    for idx, case in enumerate(TEST_CASES, start=1):
        ok, detail = run_test(case)
        status_word = "PASS" if ok else "FAIL"
        print(f"[{idx:02d}] {status_word} - {case.name}")
        print(f"     Q: {case.question}")
        print(f"     {detail}")
        print("-" * 80)

        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 80)
    print(f"TOTAL={len(TEST_CASES)} | PASSED={passed} | FAILED={failed}")
    print("=" * 80)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
