# Qwen3-Omni Flask Server

Flask app that hosts **Qwen3-Omni** from Hugging Face with:

- REST API protected by API key (`X-API-Key` or `Authorization: Bearer`)
- Web UI on port **7860** with API key login (stored in Flask session)
- Multimodal input: text, image, audio, video

Default model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`

## Requirements (EC2 GPU)

- NVIDIA GPU with **~79GB+ VRAM** for Instruct (BF16, short video) per [model card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- CUDA drivers, Python 3.10+
- `ffmpeg` for media processing

### GPU sizing (important)

**22GB VRAM is not enough** for `Qwen3-Omni-30B-A3B-Instruct`. With `device_map="auto"`, Transformers spills weights to CPU/disk; this MoE model then fails with the `offload_folder` / safetensors error you saw.

| Model | Min VRAM (BF16, ~15s video) | Notes |
|-------|-----------------------------|--------|
| `Qwen3-Omni-30B-A3B-Instruct` | **~79 GB** | Thinker + talker |
| Same + `disable_talker()` | **~69 GB** | Text-only output (app sets this when `RETURN_AUDIO=false`) |
| `Qwen3-Omni-30B-A3B-Thinking` | **~69 GB** | Text only, no talker |

**AWS instances that fit (examples):**

| Instance | GPUs × VRAM | Total | Fit? |
|----------|-------------|-------|------|
| `g5.2xlarge` | 1 × 24 GB | 24 GB | No |
| `g5.12xlarge` | 4 × 24 GB | 96 GB | Yes — use 4-way tensor parallel (`INFERENCE_BACKEND=vllm`, `-tp 4`) or multi-GPU `device_map` |
| `g5.48xlarge` | 8 × 24 GB | 192 GB | Yes |
| `p5.2xlarge` | 1 × 80 GB (H100) | 80 GB | Yes (minimum headroom) |
| `p4de.24xlarge` | 8 × 80 GB (A100) | 640 GB | Yes |

**Recommendation:** move from a **24GB** box (`g5.2xlarge` / similar) to at least **`p5.2xlarge` (80GB)** for a single GPU, or **`g5.12xlarge` (4×24GB)** with multi-GPU inference.

Disk/CPU offload is not a practical fix for this MoE model at 22GB (very slow, and often breaks as above).

### g5.12xlarge (4× A10G 24GB) — recommended `.env`

After resizing the instance, confirm 4 GPUs:

```bash
nvidia-smi
python3 -c "import torch; print('GPUs:', torch.cuda.device_count())"
```

Copy `.env.example` to `.env` and use (already tuned for **96 GB** total):

```env
INFERENCE_BACKEND=transformers
TENSOR_PARALLEL_SIZE=4
MAX_MEMORY_PER_GPU=22GiB
DISABLE_CPU_OFFLOAD=true
RETURN_AUDIO=false
```

Then deploy as usual (`./run.sh`). The app spreads the model across 4 GPUs and **blocks CPU/disk offload** (avoids the MoE `offload_folder` error).

**Tips on g5.12xlarge:**

- Keep **`RETURN_AUDIO=false`** unless you need speech output (~10 GB extra VRAM).
- First model load can take **5–15 minutes**.
- If OOM: lower `MAX_NEW_TOKENS`, avoid long video inputs, or set `VLLM_MAX_MODEL_LEN=16384` when using vLLM.
- Run a **single** server process (`python app.py` / `./run.sh`) — do not start multiple copies.
- For vLLM (optional, faster): `INFERENCE_BACKEND=vllm`, `TENSOR_PARALLEL_SIZE=4`, plus the [Qwen3-Omni vLLM build](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct#vllm-usage).

## Quick start (development / mock)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install python-dotenv

cp .env.example .env
# Edit API_KEY in .env

export MOCK_INFERENCE=true
export API_KEY=dev-secret-key
python app.py
```

Open http://localhost:7860 — sign in with `dev-secret-key`.

## Production setup on EC2

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv python3-pip

python3 -m venv .venv
. .venv/bin/activate   # use bash, or: source .venv/bin/activate

# GPU: PyTorch has no cu131 index — use cu130 for CUDA toolkit 13.1 (pin versions; unpinned often fails)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
python3 -c "import torch; print(torch.__version__, 'cuda', torch.version.cuda)"
pip install -r requirements.txt

# flash-attn (optional; matches cu130 + CUDA 13.1 toolkit better than cu126)
pip install -U flash-attn --no-build-isolation
# If flash-attn still fails, skip it — the app falls back to sdpa attention

cp .env.example .env
# Set API_KEY, optionally MODEL_PATH to pre-downloaded weights
# export HF_TOKEN=... if needed

chmod +x run.sh
./run.sh
```

The model **loads when the server starts** (before accepting traffic). First boot can take several minutes on g5.12xlarge; watch logs until `Model ready for inference`. `GET /health` reports `model_loaded` / `model_loading`.

Security group: allow inbound **TCP 7860**.

### Pre-download weights (recommended)

Weights default to `./models/` in the project directory (no `/opt` permissions needed).

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p ./models
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir ./models/Qwen3-Omni-30B-A3B-Instruct
```

Optional `.env` overrides:

```
MODELS_DIR=./models
MODEL_PATH=./models/Qwen3-Omni-30B-A3B-Instruct
```

## API documentation

| Endpoint | Description |
|----------|-------------|
| `GET /api/docs` or `GET /swagger` | Swagger UI (interactive) |
| `GET /api/openapi.json` | OpenAPI 3 specification (JSON) |

Use **Authorize** in Swagger UI with your `API_KEY` (`X-API-Key` or Bearer) before trying protected endpoints.

## API

All `/api/v1/*` routes require:

```
X-API-Key: <your-api-key>
```

or

```
Authorization: Bearer <your-api-key>
```

### Health

```bash
curl http://localhost:7860/health
```

### Chat completions (OpenAI-style)

```bash
curl -X POST http://localhost:7860/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, what can you do?"}
    ],
    "max_tokens": 512
  }'
```

### Multimodal example

```bash
curl -X POST http://localhost:7860/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }]
  }'
```

### Simple generate

```bash
curl -X POST http://localhost:7860/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"prompt": "Summarize Qwen3-Omni in one sentence."}'
```

## Web UI

1. Open `http://<ec2-host>:7860`
2. Enter the same `API_KEY` configured on the server
3. Chat; optional image/audio/video attachments

The UI uses session cookies (`/ui/api/chat`) — no need to paste the key on every message.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | (generated) | Shared secret for API auth, UI login, and Flask sessions |
| `PORT` | `7860` | Listen port |
| `MODEL_ID` / `MODEL_PATH` | Instruct HF id | Model to load |
| `INFERENCE_BACKEND` | `transformers` | `transformers` or `vllm` |
| `TENSOR_PARALLEL_SIZE` | `0` (all GPUs) | vLLM tensor parallel; set `4` on g5.12xlarge |
| `MAX_MEMORY_PER_GPU` | `22GiB` | Per-GPU cap for Transformers multi-GPU load |
| `DISABLE_CPU_OFFLOAD` | `true` | Avoid CPU/disk offload (MoE errors on small GPUs) |
| `MOCK_INFERENCE` | `false` | Skip GPU model for testing |
| `RETURN_AUDIO` | `false` | Enable talker audio output (~10 GB extra VRAM) |
| `LOAD_MODEL_ON_STARTUP` | `true` | Warm model at boot |

## vLLM backend (optional)

For higher throughput, set `INFERENCE_BACKEND=vllm` and install vLLM from the Qwen3-Omni branch per the [model card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct#vllm-usage).

## Systemd example

```ini
[Unit]
Description=Qwen3-Omni Flask
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/qwen-deploy
EnvironmentFile=/opt/qwen-deploy/.env
ExecStart=/opt/qwen-deploy/run.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
