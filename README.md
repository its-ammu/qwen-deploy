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
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers
pip install -U flash-attn --no-build-isolation  # recommended on GPU

cp .env.example .env
# Set API_KEY, optionally MODEL_PATH to pre-downloaded weights
# export HF_TOKEN=... if needed

chmod +x run.sh
./run.sh
```

Security group: allow inbound **TCP 7860**.

### Pre-download weights (recommended)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /opt/models/Qwen3-Omni-30B-A3B-Instruct
```

Set in `.env`:

```
MODEL_PATH=/opt/models/Qwen3-Omni-30B-A3B-Instruct
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
| `MOCK_INFERENCE` | `false` | Skip GPU model for testing |
| `RETURN_AUDIO` | `false` | Enable talker audio output |
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
