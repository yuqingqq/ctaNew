"""Tiny Telegram client. Reads bot token + chat ID from env vars.

Env vars:
  TELEGRAM_BOT_TOKEN   bot API token (from @BotFather)
  TELEGRAM_CHAT_ID     chat or channel ID

If either env var is missing, notify_telegram() is a no-op (returns False).
This makes Telegram strictly opt-in — the rest of the bot works without it.
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger("telegram")


def notify_telegram(text: str, parse_mode: str = "HTML",
                     disable_preview: bool = True) -> bool:
    """Send a message to the configured Telegram chat. Returns True on success."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.debug("telegram: TELEGRAM_BOT_TOKEN/CHAT_ID not set; skipping")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram message limit is 4096 chars; truncate gracefully
    if len(text) > 4000:
        text = text[:3950] + "\n…(truncated)"
    try:
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview,
        }, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning("telegram: send failed: %s", e)
        return False
