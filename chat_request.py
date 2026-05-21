"""Parse chat requests (JSON or multipart) into model messages."""

from __future__ import annotations

import json
from typing import Any

from flask import Request
from werkzeug.datastructures import FileStorage

from model_service import build_conversation_with_history, save_upload


def parse_history_field(raw: str | None) -> list[dict]:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid history JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("history must be a JSON array")
    history = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})
    return history


def _save_optional_upload(
    files: dict[str, FileStorage], key: str
) -> str | None:
    file_storage = files.get(key)
    if file_storage and file_storage.filename:
        return save_upload(file_storage)
    return None


def build_messages_from_request(req: Request) -> tuple[list[dict], dict[str, Any]]:
    """
    Build model messages from JSON or multipart request.
    Returns (messages, options dict with max_tokens, temperature, etc.).
    """
    opts: dict[str, Any] = {}

    if req.content_type and "multipart/form-data" in req.content_type:
        history = parse_history_field(req.form.get("history"))
        message = (req.form.get("message") or "").strip()
        if not message and not any(
            req.files.get(k) and req.files[k].filename for k in ("audio", "image", "video")
        ):
            raise ValueError("message or a media file is required")

        image_path = _save_optional_upload(req.files, "image")
        audio_path = _save_optional_upload(req.files, "audio")
        video_path = _save_optional_upload(req.files, "video")

        messages = build_conversation_with_history(
            history,
            message,
            (req.form.get("system_prompt") or "").strip() or None,
            image_path=image_path,
            audio_path=audio_path,
            video_path=video_path,
        )
        if req.form.get("max_tokens"):
            opts["max_new_tokens"] = int(req.form["max_tokens"])
        if req.form.get("temperature"):
            opts["temperature"] = float(req.form["temperature"])
        if req.form.get("top_p"):
            opts["top_p"] = float(req.form["top_p"])
        return messages, opts

    body = req.get_json(force=True, silent=True) or {}
    messages = body.get("messages", [])
    if not messages:
        raise ValueError("messages is required")
    if body.get("max_tokens") is not None:
        opts["max_new_tokens"] = body.get("max_tokens")
    if body.get("temperature") is not None:
        opts["temperature"] = body.get("temperature")
    if body.get("top_p") is not None:
        opts["top_p"] = body.get("top_p")
    if body.get("return_audio") is not None:
        opts["return_audio"] = body.get("return_audio")
    if body.get("speaker"):
        opts["speaker"] = body.get("speaker")
    if body.get("use_audio_in_video") is not None:
        opts["use_audio_in_video"] = body.get("use_audio_in_video")
    return messages, opts
