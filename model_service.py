"""Qwen3-Omni model loading and inference."""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from config import Config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None
_processor = None
_vllm = None
_loaded = False
_load_error: str | None = None


def _ensure_upload_dir() -> Path:
    path = Path(Config.UPLOAD_FOLDER)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_loaded() -> bool:
    return _loaded


def load_error() -> str | None:
    return _load_error


def load_model() -> None:
    global _model, _processor, _vllm, _loaded, _load_error

    if _loaded or Config.MOCK_INFERENCE:
        _loaded = True
        return

    with _lock:
        if _loaded:
            return
        try:
            logger.info(
                "Loading model %s with backend %s",
                Config.MODEL_PATH,
                Config.INFERENCE_BACKEND,
            )
            if Config.INFERENCE_BACKEND == "vllm":
                _load_vllm()
            else:
                _load_transformers()
            _loaded = True
            logger.info("Model loaded successfully")
        except Exception as exc:
            _load_error = str(exc)
            logger.exception("Failed to load model: %s", exc)
            raise


def _load_transformers() -> None:
    global _model, _processor
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    attn = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn = "sdpa"
        logger.warning("flash-attn not found; using sdpa attention")

    _processor = Qwen3OmniMoeProcessor.from_pretrained(Config.MODEL_PATH)
    load_kwargs: dict[str, Any] = {
        "dtype": "auto",
        "device_map": "auto",
        "attn_implementation": attn,
    }
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        max_memory = {i: Config.MAX_MEMORY_PER_GPU for i in range(gpu_count)}
        if Config.DISABLE_CPU_OFFLOAD:
            max_memory["cpu"] = "0GiB"
            max_memory["disk"] = "0GiB"
        load_kwargs["max_memory"] = max_memory
        logger.info(
            "Multi-GPU load: %s GPUs, max_memory=%s",
            gpu_count,
            Config.MAX_MEMORY_PER_GPU,
        )
    _model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        Config.MODEL_PATH,
        **load_kwargs,
    )
    if not Config.RETURN_AUDIO:
        _model.disable_talker()


def _load_vllm() -> None:
    global _vllm, _processor
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM

    os.environ["VLLM_USE_V1"] = "0"
    _processor = Qwen3OmniMoeProcessor.from_pretrained(Config.MODEL_PATH)
    tp = Config.resolve_tensor_parallel_size()
    logger.info("vLLM tensor_parallel_size=%s", tp)
    _vllm = LLM(
        model=Config.MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=Config.VLLM_GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=tp,
        limit_mm_per_prompt={"image": 3, "video": 3, "audio": 3},
        max_num_seqs=4,
        max_model_len=Config.VLLM_MAX_MODEL_LEN,
    )


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style content blocks to Qwen Omni format."""
    normalized = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        blocks = []
        for block in content:
            if isinstance(block, str):
                blocks.append({"type": "text", "text": block})
                continue
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "text")
            if block_type == "text":
                blocks.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image_url":
                url = block.get("image_url", {}).get("url", "")
                blocks.append({"type": "image", "image": url})
            elif block_type == "input_image":
                blocks.append({"type": "image", "image": block.get("image_url", "")})
            elif block_type == "image":
                blocks.append({"type": "image", "image": block.get("image", "")})
            elif block_type == "audio_url":
                url = block.get("audio_url", {}).get("url", "")
                blocks.append({"type": "audio", "audio": url})
            elif block_type == "audio":
                blocks.append({"type": "audio", "audio": block.get("audio", "")})
            elif block_type == "video_url":
                url = block.get("video_url", {}).get("url", "")
                blocks.append({"type": "video", "video": url})
            elif block_type == "video":
                blocks.append({"type": "video", "video": block.get("video", "")})
            else:
                text = block.get("text")
                if text:
                    blocks.append({"type": "text", "text": text})

        if len(blocks) == 1 and blocks[0]["type"] == "text":
            normalized.append({"role": role, "content": blocks[0]["text"]})
        else:
            normalized.append({"role": role, "content": blocks})
    return normalized


def save_upload(file_storage) -> str:
    """Save an uploaded file and return its absolute path."""
    upload_dir = _ensure_upload_dir()
    ext = Path(file_storage.filename or "bin").suffix or ".bin"
    dest = upload_dir / f"{uuid.uuid4().hex}{ext}"
    file_storage.save(dest)
    return str(dest.resolve())


def build_conversation_from_form(
    message: str,
    system_prompt: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
    video_path: str | None = None,
) -> list[dict]:
    conversation: list[dict] = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})

    user_content: list[dict] = []
    if image_path:
        user_content.append({"type": "image", "image": image_path})
    if audio_path:
        user_content.append({"type": "audio", "audio": audio_path})
    if video_path:
        user_content.append({"type": "video", "video": video_path})
    user_content.append({"type": "text", "text": message})
    conversation.append({"role": "user", "content": user_content})
    return conversation


def generate(
    messages: list[dict],
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    return_audio: bool | None = None,
    speaker: str | None = None,
    use_audio_in_video: bool | None = None,
) -> dict[str, Any]:
    if Config.MOCK_INFERENCE:
        return _mock_generate(messages)

    load_model()
    if not _loaded:
        raise RuntimeError(_load_error or "Model is not loaded")

    messages = normalize_messages(messages)
    max_new_tokens = max_new_tokens or Config.MAX_NEW_TOKENS
    temperature = temperature if temperature is not None else Config.TEMPERATURE
    top_p = top_p if top_p is not None else Config.TOP_P
    return_audio = Config.RETURN_AUDIO if return_audio is None else return_audio
    speaker = speaker or Config.DEFAULT_SPEAKER
    use_audio_in_video = (
        Config.USE_AUDIO_IN_VIDEO
        if use_audio_in_video is None
        else use_audio_in_video
    )

    with _lock:
        if Config.INFERENCE_BACKEND == "vllm":
            return _generate_vllm(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        return _generate_transformers(
            messages,
            max_new_tokens=max_new_tokens,
            return_audio=return_audio,
            speaker=speaker,
            use_audio_in_video=use_audio_in_video,
        )


def _mock_generate(messages: list[dict]) -> dict[str, Any]:
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user = content
            elif isinstance(content, list):
                texts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                last_user = " ".join(texts)
            break
    return {
        "text": f"[mock] Echo: {last_user[:500]}",
        "audio_path": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _generate_transformers(
    messages: list[dict],
    *,
    max_new_tokens: int,
    return_audio: bool,
    speaker: str,
    use_audio_in_video: bool,
) -> dict[str, Any]:
    from qwen_omni_utils import process_mm_info

    text = _processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        messages, use_audio_in_video=use_audio_in_video
    )
    inputs = _processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(_model.device).to(_model.dtype)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "use_audio_in_video": use_audio_in_video,
        "thinker_return_dict_in_generate": True,
    }
    if return_audio:
        gen_kwargs["speaker"] = speaker
    else:
        gen_kwargs["return_audio"] = False

    text_ids, audio_out = _model.generate(**inputs, **gen_kwargs)
    decoded = _processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response_text = decoded[0] if decoded else ""

    audio_path = None
    if audio_out is not None:
        import soundfile as sf

        upload_dir = _ensure_upload_dir()
        audio_path = str(upload_dir / f"response_{uuid.uuid4().hex}.wav")
        sf.write(
            audio_path,
            audio_out.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )

    prompt_len = inputs["input_ids"].shape[1]
    completion_len = text_ids.sequences.shape[1] - prompt_len
    return {
        "text": response_text,
        "audio_path": audio_path,
        "usage": {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(completion_len),
            "total_tokens": int(prompt_len + completion_len),
        },
    }


def _generate_vllm(
    messages: list[dict],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    from qwen_omni_utils import process_mm_info
    from vllm import SamplingParams

    use_audio_in_video = Config.USE_AUDIO_IN_VIDEO
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(
        messages, use_audio_in_video=use_audio_in_video
    )

    vllm_input: dict[str, Any] = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
    }
    if images is not None:
        vllm_input["multi_modal_data"]["image"] = images
    if videos is not None:
        vllm_input["multi_modal_data"]["video"] = videos
    if audios is not None:
        vllm_input["multi_modal_data"]["audio"] = audios

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    outputs = _vllm.generate([vllm_input], sampling_params=sampling_params)
    response_text = outputs[0].outputs[0].text
    return {
        "text": response_text,
        "audio_path": None,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def audio_to_base64_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"
