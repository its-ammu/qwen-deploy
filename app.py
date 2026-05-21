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
import threading
import time
from datetime import datetime, timezone

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
from config import DEFAULT_MODELS_DIR, Config
from model_service import (
    audio_to_base64_data_url,
    build_conversation_from_form,
    generate,
    is_loaded,
    load_error,
    load_model,
    normalize_messages,
    save_upload,
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
        body = request.get_json(force=True, silent=True) or {}
        messages = body.get("messages", [])
        if not messages:
            return jsonify({"error": {"message": "messages is required"}}), 400

        result = generate(
            messages,
            max_new_tokens=body.get("max_tokens"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            return_audio=body.get("return_audio"),
            speaker=body.get("speaker"),
            use_audio_in_video=body.get("use_audio_in_video"),
        )

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

    @app.route("/ui/api/chat", methods=["POST"])
    @require_ui_session
    def ui_chat():
        system_prompt = request.form.get("system_prompt", "").strip() or None
        message = request.form.get("message", "").strip()
        if not message:
            return jsonify({"error": {"message": "message is required"}}), 400

        image_path = audio_path = video_path = None
        if "image" in request.files and request.files["image"].filename:
            image_path = save_upload(request.files["image"])
        if "audio" in request.files and request.files["audio"].filename:
            audio_path = save_upload(request.files["audio"])
        if "video" in request.files and request.files["video"].filename:
            video_path = save_upload(request.files["video"])

        conversation = build_conversation_from_form(
            message,
            system_prompt=system_prompt,
            image_path=image_path,
            audio_path=audio_path,
            video_path=video_path,
        )

        try:
            max_tokens = request.form.get("max_tokens")
            result = generate(
                conversation,
                max_new_tokens=int(max_tokens) if max_tokens else None,
                temperature=float(request.form["temperature"])
                if request.form.get("temperature")
                else None,
            )
        except Exception as exc:
            logger.exception("UI inference failed")
            return jsonify({"error": {"message": str(exc)}}), 500

        payload = {
            "text": result["text"],
            "usage": result.get("usage", {}),
        }
        if result.get("audio_path"):
            payload["audio_url"] = audio_to_base64_data_url(result["audio_path"])
        return jsonify(payload)

    @app.route("/ui/api/chat/json", methods=["POST"])
    @require_ui_session
    def ui_chat_json():
        """JSON chat for UI when using prior messages history."""
        body = request.get_json(force=True) or {}
        messages = body.get("messages", [])
        if not messages:
            return jsonify({"error": {"message": "messages is required"}}), 400
        messages = normalize_messages(messages)
        try:
            result = generate(
                messages,
                max_new_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
            )
        except Exception as exc:
            logger.exception("UI JSON inference failed")
            return jsonify({"error": {"message": str(exc)}}), 500

        payload = {"text": result["text"], "usage": result.get("usage", {})}
        if result.get("audio_path"):
            payload["audio_url"] = audio_to_base64_data_url(result["audio_path"])
        return jsonify(payload)

    return app


app = create_app()


@app.before_request
def _ensure_model_on_first_request():
    if (
        not is_loaded()
        and Config.LOAD_MODEL_ON_STARTUP
        and not Config.MOCK_INFERENCE
        and request.path.startswith(("/api/", "/ui/"))
    ):
        try:
            load_model()
        except Exception:
            pass


def _startup_load():
    if Config.LOAD_MODEL_ON_STARTUP and not Config.MOCK_INFERENCE:
        try:
            load_model()
        except Exception:
            logger.error("Startup model load failed; will retry on first request")


if __name__ == "__main__":
    if Config.LOAD_MODEL_ON_STARTUP:
        threading.Thread(target=_startup_load, daemon=True).start()
    app.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)
