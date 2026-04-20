from pydantic import BaseModel, Field
from typing import Literal


class AskRequest(BaseModel):
    message_text: str = Field(..., min_length=1)
    channel: str = "manual_test"
    user_id: str | None = None


class AskResponse(BaseModel):
    status: Literal[
        "answered",
        "needs_clarification",
        "not_found_officially",
        "escalate",
    ]
    reply_text: str
    category: str = "other"
    official_source_used: bool = False
    source_name: str | None = None
    reason: str | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    case_id: str | None = None


OUTPUT_SCHEMA = {
    "type": "json_schema",
    "name": "admission_registration_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": [
                    "answered",
                    "needs_clarification",
                    "not_found_officially",
                    "escalate",
                ],
            },
            "reply_text": {"type": "string"},
            "category": {"type": "string"},
            "official_source_used": {"type": "boolean"},
            "source_name": {"type": ["string", "null"]},
            "reason": {"type": ["string", "null"]},
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
        },
        "required": [
            "status",
            "reply_text",
            "category",
            "official_source_used",
            "source_name",
            "reason",
            "confidence",
        ],
    },
}