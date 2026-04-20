from __future__ import annotations

import json
from pathlib import Path
from openai import OpenAI

from .settings import settings
from .schemas import OUTPUT_SCHEMA

client = OpenAI(api_key=settings.openai_api_key)

BASE_DIR = Path(__file__).resolve().parent.parent
INSTRUCTIONS_PATH = BASE_DIR / "instructions.md"
DECISION_POLICY_PATH = BASE_DIR / "decision_policy.md"

SYSTEM_INSTRUCTIONS = INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
DECISION_POLICY = DECISION_POLICY_PATH.read_text(encoding="utf-8").strip()


def ask_agent(message_text: str) -> dict:
    tools = []
    if settings.openai_vector_store_id:
        tools.append(
            {
                "type": "file_search",
                "vector_store_ids": [settings.openai_vector_store_id],
            }
        )

    response = client.responses.create(
        model=settings.openai_model,
        instructions=SYSTEM_INSTRUCTIONS + "\n\n" + DECISION_POLICY,
        input=message_text,
        tools=tools or None,
        text={"format": OUTPUT_SCHEMA},
    )

    text = getattr(response, "output_text", None)
    if not text:
        raise ValueError("Empty model output")

    data = json.loads(text)

    return {
        "status": data["status"],
        "reply_text": data["reply_text"],
        "category": data["category"],
        "official_source_used": data["official_source_used"],
        "source_name": data["source_name"],
        "reason": data["reason"],
        "confidence": data["confidence"],
    }