"""OpenAPI spec and API documentation helpers."""

from __future__ import annotations

from config import Config


def build_openapi_spec(base_url: str) -> dict:
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Qwen3-Omni API",
            "description": "REST API for Qwen3-Omni inference. Use the same API_KEY for "
            "X-API-Key / Bearer auth and for the web UI login.",
            "version": "1.0.0",
        },
        "servers": [{"url": base_url}],
        "components": {
            "securitySchemes": {
                "ApiKeyHeader": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                },
            },
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "model": {"type": "string"},
                        "model_path": {"type": "string"},
                        "backend": {"type": "string"},
                        "model_loaded": {"type": "boolean"},
                        "mock": {"type": "boolean"},
                        "load_error": {"type": "string", "nullable": True},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
                "ChatCompletionRequest": {
                    "type": "object",
                    "required": ["messages"],
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ChatMessage"},
                        },
                        "max_tokens": {"type": "integer"},
                        "temperature": {"type": "number"},
                        "top_p": {"type": "number"},
                        "return_audio": {"type": "boolean"},
                        "speaker": {"type": "string"},
                        "use_audio_in_video": {"type": "boolean"},
                    },
                },
                "ChatMessage": {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["system", "user", "assistant"],
                        },
                        "content": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {
                                        "$ref": "#/components/schemas/ContentBlock"
                                    },
                                },
                            ],
                        },
                    },
                },
                "ContentBlock": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "text",
                                "image_url",
                                "audio_url",
                                "video_url",
                            ],
                        },
                        "text": {"type": "string"},
                        "image_url": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                        },
                        "audio_url": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                        },
                        "video_url": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                        },
                    },
                },
                "GenerateRequest": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ChatMessage"},
                        },
                        "prompt": {"type": "string"},
                        "max_tokens": {"type": "integer"},
                        "temperature": {"type": "number"},
                        "top_p": {"type": "number"},
                    },
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"},
                                "type": {"type": "string"},
                            },
                        }
                    },
                },
            },
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "Service status",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthResponse"
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/v1/models": {
                "get": {
                    "summary": "List models",
                    "tags": ["Models"],
                    "security": [
                        {"ApiKeyHeader": []},
                        {"BearerAuth": []},
                    ],
                    "responses": {"200": {"description": "Model list"}},
                }
            },
            "/api/v1/chat/completions": {
                "post": {
                    "summary": "Chat completions (OpenAI-compatible)",
                    "tags": ["Chat"],
                    "security": [
                        {"ApiKeyHeader": []},
                        {"BearerAuth": []},
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ChatCompletionRequest"
                                },
                                "example": {
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": "Hello, what can you do?",
                                        }
                                    ],
                                    "max_tokens": 512,
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Completion"},
                        "401": {
                            "description": "Unauthorized",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/api/v1/generate": {
                "post": {
                    "summary": "Simple text generation",
                    "tags": ["Chat"],
                    "security": [
                        {"ApiKeyHeader": []},
                        {"BearerAuth": []},
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/GenerateRequest"
                                },
                                "example": {
                                    "prompt": "Summarize Qwen3-Omni in one sentence.",
                                    "max_tokens": 256,
                                },
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Generated text"},
                        "401": {
                            "description": "Unauthorized",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/ui/api/chat": {
                "post": {
                    "summary": "UI chat (multipart, session cookie)",
                    "tags": ["UI"],
                    "description": "Requires Flask session after UI login. "
                    "Fields: message, system_prompt, max_tokens, image, audio, video.",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "system_prompt": {"type": "string"},
                                        "max_tokens": {"type": "integer"},
                                        "image": {"type": "string", "format": "binary"},
                                        "audio": {"type": "string", "format": "binary"},
                                        "video": {"type": "string", "format": "binary"},
                                    },
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Assistant reply"}},
                }
            },
        },
        "tags": [
            {"name": "System"},
            {"name": "Models"},
            {"name": "Chat"},
            {"name": "UI"},
        ],
        "x-model": Config.MODEL_ID,
    }
