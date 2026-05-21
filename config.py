import os
import secrets

# Project root (directory containing config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "models")


def default_model_path(model_id: str) -> str:
    models_dir = os.environ.get("MODELS_DIR", DEFAULT_MODELS_DIR)
    if not os.path.isabs(models_dir):
        models_dir = os.path.abspath(models_dir)
    return os.path.join(models_dir, model_id.split("/")[-1])


def _resolve_shared_key() -> str:
    """Single key for API auth and Flask session signing."""
    return os.environ.get("API_KEY") or os.environ.get("SECRET_KEY") or ""


class Config:
    """Application configuration loaded from environment variables."""

    API_KEY = _resolve_shared_key()
    SECRET_KEY = API_KEY or secrets.token_hex(32)

    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", "7860"))

    MODEL_ID = os.environ.get(
        "MODEL_ID", "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    MODEL_PATH = os.environ.get("MODEL_PATH") or default_model_path(MODEL_ID)

    # transformers | vllm
    INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "transformers")
    # 0 = use all visible GPUs (g5.12xlarge → 4); set 4 explicitly on 4-GPU nodes
    TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "0"))
    MAX_MEMORY_PER_GPU = os.environ.get("MAX_MEMORY_PER_GPU", "22GiB")
    DISABLE_CPU_OFFLOAD = os.environ.get("DISABLE_CPU_OFFLOAD", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    VLLM_GPU_MEMORY_UTILIZATION = float(
        os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
    )
    VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))

    LOAD_MODEL_ON_STARTUP = os.environ.get("LOAD_MODEL_ON_STARTUP", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    MOCK_INFERENCE = os.environ.get("MOCK_INFERENCE", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    RETURN_AUDIO = os.environ.get("RETURN_AUDIO", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    USE_AUDIO_IN_VIDEO = os.environ.get("USE_AUDIO_IN_VIDEO", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "Ethan")

    MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
    TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
    TOP_P = float(os.environ.get("TOP_P", "0.95"))

    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/tmp/qwen-uploads")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", str(100 * 1024 * 1024)))

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    @staticmethod
    def resolve_tensor_parallel_size() -> int:
        if Config.TENSOR_PARALLEL_SIZE > 0:
            return Config.TENSOR_PARALLEL_SIZE
        try:
            import torch

            return max(1, torch.cuda.device_count())
        except ImportError:
            return 1
