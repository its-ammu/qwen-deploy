from functools import wraps

from flask import jsonify, request, session

from config import Config


def _extract_api_key() -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.headers.get("X-API-Key") or request.args.get("api_key")


def _is_valid_key(key: str | None) -> bool:
    if not Config.API_KEY:
        return False
    return bool(key) and key == Config.API_KEY


def require_api_key(f):
    """Protect API routes with a shared API key."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if not Config.API_KEY:
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Server API_KEY is not configured",
                            "type": "configuration_error",
                        }
                    }
                ),
                503,
            )
        if not _is_valid_key(_extract_api_key()):
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Invalid or missing API key",
                            "type": "authentication_error",
                        }
                    }
                ),
                401,
            )
        return f(*args, **kwargs)

    return decorated


def require_ui_session(f):
    """Protect UI proxy routes with a Flask session flag."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Not authenticated. Submit API key on the login form.",
                            "type": "authentication_error",
                        }
                    }
                ),
                401,
            )
        return f(*args, **kwargs)

    return decorated


def validate_api_key(key: str | None) -> bool:
    return _is_valid_key(key)
