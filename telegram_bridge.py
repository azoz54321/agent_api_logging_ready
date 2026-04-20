from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import requests
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================
# Configuration
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "").strip()

ASK_ENDPOINT = f"{AGENT_BASE_URL}/v1/ask"
REQUEST_TIMEOUT_SECONDS = 45

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_bridge")


@dataclass
class AgentReply:
    status: str
    reply_text: str
    reason: str | None = None
    category: str | None = None
    case_id: str | None = None


def build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if AGENT_API_KEY:
        headers["x-api-key"] = AGENT_API_KEY
    return headers


def call_agent(message_text: str, user_id: str) -> AgentReply:
    payload = {
        "message_text": message_text,
        "channel": "telegram",
        "user_id": user_id,
    }

    response = requests.post(
        ASK_ENDPOINT,
        json=payload,
        headers=build_headers(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()

    return AgentReply(
        status=data.get("status", "escalate"),
        reply_text=data.get("reply_text", "تعذر إنشاء رد صحيح."),
        reason=data.get("reason"),
        category=data.get("category"),
        case_id=data.get("case_id"),
    )


async def ask_agent_async(message_text: str, user_id: str) -> AgentReply:
    return await asyncio.to_thread(call_agent, message_text, user_id)


def format_reply(agent_reply: AgentReply) -> str:
    # هنا نرسل للمستخدم فقط نص الرد.
    # إذا أردت لاحقًا إظهار case_id في حالات التصعيد، فعّل الجزء المعلق أدناه.
    text = agent_reply.reply_text.strip()

    # مثال اختياري:
    # if agent_reply.status == "escalate" and agent_reply.case_id:
    #     text += f"\n\nرقم الحالة: {agent_reply.case_id}"

    return text


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    await update.message.reply_text(
        "مرحبًا بك.\n"
        "أرسل سؤالك مباشرة، وسيتم الرد عليك وفق الملفات الرسمية المعتمدة."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    await update.message.reply_text(
        "طريقة الاستخدام:\n"
        "1) أرسل سؤالك مباشرة.\n"
        "2) إذا كان السؤال يحتاج توضيحًا، سيطلب منك النظام التفاصيل الناقصة.\n"
        "3) إذا كانت الحالة تحتاج مراجعة بشرية، سيظهر لك ذلك في الرد."
    )


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat:
        return

    message_text = (update.message.text or "").strip()
    if not message_text:
        await update.message.reply_text("أرسل نص السؤال بشكل واضح.")
        return

    user_id = str(update.effective_chat.id)

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        agent_reply = await ask_agent_async(message_text=message_text, user_id=user_id)
        reply_text = format_reply(agent_reply)

        await update.message.reply_text(reply_text)

        logger.info(
            "Handled telegram message | user_id=%s | status=%s | reason=%s | category=%s | case_id=%s",
            user_id,
            agent_reply.status,
            agent_reply.reason,
            agent_reply.category,
            agent_reply.case_id,
        )

    except requests.HTTPError as exc:
        logger.exception("Agent HTTP error")
        body = ""
        try:
            body = exc.response.text
        except Exception:
            pass

        await update.message.reply_text(
            "تعذر معالجة الطلب حاليًا بسبب خطأ في الاتصال بالنظام."
        )
        logger.error("HTTP error body: %s", body)

    except requests.RequestException:
        logger.exception("Agent request failed")
        await update.message.reply_text(
            "تعذر الاتصال بالنظام حاليًا. حاول مرة أخرى بعد قليل."
        )

    except Exception:
        logger.exception("Unexpected telegram bridge error")
        await update.message.reply_text(
            "حدث خطأ غير متوقع أثناء معالجة الطلب."
        )


async def unsupported_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    await update.message.reply_text(
        "حاليًا يتم دعم الرسائل النصية فقط."
    )


def validate_config() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment variables.")

    if not AGENT_BASE_URL:
        raise RuntimeError("Missing AGENT_BASE_URL in environment variables.")


def main() -> None:
    validate_config()

    logger.info("Starting Telegram bridge...")
    logger.info("Agent endpoint: %s", ASK_ENDPOINT)
    logger.info("Agent API key set: %s", "yes" if AGENT_API_KEY else "no")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
    app.add_handler(MessageHandler(~filters.TEXT, unsupported_message_handler))

    # Polling مناسب الآن للتجربة السريعة
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )

if __name__ == "__main__":
    main()