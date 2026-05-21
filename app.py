"""Flask server for Qwen3-Omni with REST API and web UI."""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from api_docs import build_openapi_spec
from auth import require_api_key, require_ui_session, validate_api_key
from chat_request import build_messages_from_request
from config import DEFAULT_MODELS_DIR, Config
from model_service import (
    audio_to_base64_data_url,
    generate,
    is_loaded,
    is_loading,
    load_error,
    load_model,
    normalize_messages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)

    if not Config.API_KEY:
        generated = secrets.token_urlsafe(32)
        Config.API_KEY = generated
        Config.SECRET_KEY = generated
        app.config["SECRET_KEY"] = generated
        logger.warning(
            "API_KEY not set; generated ephemeral key (set API_KEY on EC2): %s",
            generated,
        )

    @app.route("/api/docs")
    @app.route("/swagger")
    def api_docs():
        return render_template(
            "swagger.html",
            openapi_url=url_for("api_openapi", _external=True),
        )

    @app.route("/api/openapi.json")
    def api_openapi():
        base = request.url_root.rstrip("/")
        return jsonify(build_openapi_spec(base))

    @app.route("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model": Config.MODEL_ID,
                "model_path": Config.MODEL_PATH,
                "backend": Config.INFERENCE_BACKEND,
                "tensor_parallel_size": Config.resolve_tensor_parallel_size(),
                "model_loaded": is_loaded(),
                "model_loading": is_loading(),
                "mock": Config.MOCK_INFERENCE,
                "load_error": load_error(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    @app.route("/")
    def index():
        if session.get("authenticated"):
            return redirect(url_for("chat"))
        return render_template("login.html")

    @app.route("/auth", methods=["POST"])
    def auth():
        api_key = request.form.get("api_key", "").strip()
        if not validate_api_key(api_key):
            return render_template("login.html", error="Invalid API key"), 401
        session["authenticated"] = True
        session.permanent = True
        return redirect(url_for("chat"))

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("index"))

    @app.route("/chat")
    def chat():
        if not session.get("authenticated"):
            return redirect(url_for("index"))
        return render_template(
            "chat.html",
            model_id=Config.MODEL_ID,
            return_audio=Config.RETURN_AUDIO,
        )

    # --- Public API (API key required) ---

    @app.route("/api/v1/models")
    @require_api_key
    def api_models():
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": Config.MODEL_ID,
                        "object": "model",
                        "owned_by": "qwen",
                        "backend": Config.INFERENCE_BACKEND,
                    }
                ],
            }
        )

    @app.route("/api/v1/chat/completions", methods=["POST"])
    @require_api_key
    def api_chat_completions():
        try:
            messages, opts = build_messages_from_request(request)
            messages = normalize_messages(messages)
        except ValueError as exc:
            return jsonify({"error": {"message": str(exc)}}), 400

        result = generate(messages, **opts)

        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": Config.MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": result.get("usage", {}),
        }
        if result.get("audio_path"):
            response["audio"] = {
                "path": result["audio_path"],
                "data_url": audio_to_base64_data_url(result["audio_path"]),
            }
        return jsonify(response)

    @app.route("/api/v1/generate", methods=["POST"])
    @require_api_key
    def api_generate():
        body: dict = {}
        if request.is_json:
            body = request.get_json(force=True) or {}
            messages = body.get("messages")
            if not messages and body.get("prompt"):
                messages = [{"role": "user", "content": body["prompt"]}]
        else:
            prompt = request.form.get("prompt", "")
            messages = [{"role": "user", "content": prompt}] if prompt else []

        if not messages:
            return jsonify({"error": {"message": "messages or prompt required"}}), 400
        result = generate(
            messages,
            max_new_tokens=body.get("max_tokens"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
        )
        return jsonify(
            {
                "model": Config.MODEL_ID,
                "text": result["text"],
                "audio_path": result.get("audio_path"),
                "usage": result.get("usage", {}),
            }
        )

    # --- UI proxy (session auth; uses same inference path) ---

    def _ui_chat_response(result: dict) -> Any:
        payload = {"text": result["text"], "usage": result.get("usage", {})}
        if result.get("audio_path"):
            payload["audio_url"] = audio_to_base64_data_url(result["audio_path"])
        return jsonify(payload)

    @app.route("/ui/api/chat", methods=["POST"])
    @require_ui_session
    def ui_chat():
        try:
            messages, opts = build_messages_from_request(request)
            messages = normalize_messages(messages)
            result = generate(messages, **opts)
        except ValueError as exc:
            return jsonify({"error": {"message": str(exc)}}), 400
        except Exception as exc:
            logger.exception("UI inference failed")
            return jsonify({"error": {"message": str(exc)}}), 500
        return _ui_chat_response(result)

    return app


app = create_app()


def preload_model_at_startup() -> None:
    """Load model before the Flask server starts accepting requests."""
    if not Config.LOAD_MODEL_ON_STARTUP or Config.MOCK_INFERENCE:
        return
    if is_loaded():
        return
    logger.info("Preloading model at server startup (path=%s)...", Config.MODEL_PATH)
    load_model()
    logger.info("Model ready for inference")


if __name__ == "__main__":
    preload_model_at_startup()
    app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)
