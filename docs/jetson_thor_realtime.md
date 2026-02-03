# Jetson Thor Realtime Deployment Guide

This guide provides detailed instructions for deploying Unmute's realtime voice conversation system on NVIDIA Jetson Thor hardware, optimized for low-latency robot applications.

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [System Architecture](#system-architecture)
4. [Prerequisites](#prerequisites)
5. [Deployment Options](#deployment-options)
6. [Process Layout & Resource Allocation](#process-layout--resource-allocation)
7. [GPU/CPU Pinning Strategy](#gpucpu-pinning-strategy)
8. [Service Configuration](#service-configuration)
9. [Health Probes & Monitoring](#health-probes--monitoring)
10. [Performance Tuning](#performance-tuning)
11. [Logging Configuration](#logging-configuration)
12. [Failover Procedures](#failover-procedures)
13. [Troubleshooting](#troubleshooting)
14. [Known Limitations](#known-limitations)

## Overview

Unmute on Jetson Thor enables real-time voice conversations for robotic applications with:
- **Sub-300ms STT flush latency**: Critical for responsive robot interactions
- **Low-jitter audio output**: Smooth, natural-sounding speech
- **Fast tool-call turnaround**: Quick actuator responses (<2000ms RTT)
- **Graceful degradation**: Continues operating when services hiccup

The system consists of four core services:
1. **Backend** (Python/FastAPI): WebSocket orchestration and session management
2. **STT** (Rust/CUDA): Speech-to-text transcription (Kyutai STT-1B)
3. **TTS** (Rust/CUDA + Python): Text-to-speech synthesis (Kyutai TTS-1.6B)
4. **LLM** (vLLM): Language model for conversation (configurable model)

Optional services:
- **Frontend** (Next.js): Web UI for testing and monitoring
- **Traefik**: Reverse proxy and load balancer
- **Prometheus**: Metrics collection and monitoring

## Hardware Requirements

### Minimum Specifications

**For Jetson Thor (or equivalent):**
- **GPU**: NVIDIA GPU with CUDA support
  - Minimum 16GB VRAM for basic operation
  - 24GB+ VRAM recommended for production with larger models
- **CPU**: 8+ cores recommended for parallel service execution
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: 50GB+ for models and logs
- **Network**: Gigabit Ethernet or WiFi 6 for low-latency WebSocket connections

**Note**: The official README states that x86_64 architecture is required and aarch64 (ARM64, used by Jetson) is not officially supported. However, this guide documents the deployment process for those working to enable Jetson support or using compatible hardware.

### GPU Memory Budget (Typical)

Based on docker-compose.yml and README specifications:

| Service | VRAM Usage | Notes |
|---------|-----------|-------|
| STT | ~2.5GB | Kyutai STT-1B model |
| TTS | ~5.3GB | Kyutai TTS-1.6B model |
| LLM (Llama-3.2-1B) | ~6.1GB | Depends on model size |
| LLM (Mistral-Small-24B) | ~14GB | Larger model option |
| **Total (small config)** | **~14GB** | Minimal setup with Llama-3.2-1B |
| **Total (production)** | **~22GB** | Production with larger LLM |

**Multi-GPU Strategy**: On unmute.sh production, STT, TTS, and LLM run on separate GPUs, reducing TTS latency from ~750ms (single GPU) to ~450ms (multi-GPU).

## System Architecture

### Service Communication Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        Jetson Thor                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │   Frontend   │────────▶│   Traefik    │                  │
│  │  (Optional)  │         │  (Optional)  │                  │
│  └──────────────┘         └──────┬───────┘                  │
│                                   │                          │
│                           ┌───────▼────────┐                 │
│  Robot ──────────────────▶│    Backend     │                 │
│  Client                   │  (FastAPI)     │                 │
│  (WebSocket)              │  Port: 8000    │                 │
│                           └───────┬────────┘                 │
│                                   │                          │
│           ┌───────────────────────┼──────────────────┐       │
│           │                       │                  │       │
│      ┌────▼─────┐          ┌─────▼─────┐      ┌────▼────┐  │
│      │   STT    │          │    TTS    │      │   LLM   │  │
│      │(Rust+GPU)│          │(Rust+GPU) │      │ (vLLM)  │  │
│      │Port: 8080│          │Port: 8080 │      │Port:8000│  │
│      │VRAM: 2.5G│          │VRAM: 5.3G │      │VRAM: 6GB│  │
│      └──────────┘          └───────────┘      └─────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Audio Input**: Robot client sends Opus-encoded audio via WebSocket (`input_audio_buffer.append`)
2. **STT Processing**: Backend streams audio to STT service, receives transcription deltas
3. **LLM Generation**: On speech pause, backend sends transcript to LLM for response generation
4. **TTS Synthesis**: LLM response streamed word-by-word to TTS for audio synthesis
5. **Audio Output**: TTS audio streamed back to robot client via WebSocket (`response.audio.delta`)

## Prerequisites

### Software Dependencies

**Option 1: Docker Deployment (Recommended)**
- Docker 24.0+ with Docker Compose
- NVIDIA Container Toolkit for GPU access
- CUDA drivers compatible with your GPU

**Option 2: Native Deployment (Advanced)**
- Python 3.11+ with `uv` package manager
- Rust toolchain (latest stable)
- Cargo build system
- `pnpm` for frontend (optional)
- CUDA 12.1+ (required for Rust STT/TTS workers)
- libpython (for TTS Python component)

### CUDA Version Requirements

The STT and TTS services require **CUDA 12.1 or later**. This is needed because:
- The Rust moshi-server binary is compiled with CUDA support
- GPU inference for both models requires CUDA runtime

**Verify CUDA installation:**
```bash
nvidia-smi  # Should show driver version and CUDA version
nvcc --version  # Should show CUDA compiler version
```

**For Docker deployment:**
```bash
# Verify NVIDIA Container Toolkit
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### Hugging Face Authentication

Models are downloaded from Hugging Face Hub. Set up authentication:

```bash
# Create Hugging Face account and token at https://huggingface.co/settings/tokens
# Token needs "Read access to contents of all public gated repos you can access"

# Add to ~/.bashrc or equivalent
export HUGGING_FACE_HUB_TOKEN=hf_your_token_here

# For production: DO NOT use tokens with write access
```

**Accept model licenses:**
- [Kyutai STT-1B](https://huggingface.co/kyutai/stt-1b-en_fr-candle)
- [Kyutai TTS-1.6B](https://huggingface.co/kyutai/tts-1.6b-en_fr)
- [Default LLM Model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (or your chosen alternative)

## Deployment Options

### Option 1: Docker Compose (Recommended for Testing)

**Advantages**: Reproducible environment, easy service management, isolated dependencies

**Limitations**: Single-machine deployment, requires sufficient VRAM on one GPU

**Steps:**

1. Clone the repository:
```bash
git clone https://github.com/kyutai-labs/unmute.git
cd unmute
```

2. Configure environment:
```bash
# Ensure HF token is set
echo $HUGGING_FACE_HUB_TOKEN

# (Optional) Set other environment variables
export NEWSAPI_API_KEY=your_key_here  # For news demo character
```

3. Review and adjust `docker-compose.yml`:
```yaml
# Key configuration points (see NOTE comments in file):

# LLM model selection (line 142):
services:
  llm:
    command:
      - "--model=meta-llama/Llama-3.2-1B-Instruct"  # 6GB VRAM
      # - "--model=mistralai/Mistral-Small-3.2-24B-Instruct-2506"  # 14GB VRAM
      - "--max-model-len=6144"  # Context window
      - "--gpu-memory-utilization=0.7"  # Adjust based on VRAM

# CPU limits (for CPU pinning, see lines 99, 125):
services:
  tts:
    deploy:
      resources:
        limits:
          cpus: 6  # Adjust based on available cores
  stt:
    deploy:
      resources:
        limits:
          cpus: 2  # Adjust based on available cores
```

4. Start services:
```bash
docker compose up --build
```

5. Verify deployment:
```bash
# Check all services are running
docker compose ps

# Check GPU allocation
docker exec unmute-tts-1 nvidia-smi

# Test health endpoint
curl http://localhost/api/v1/health
```

6. Access UI (if using frontend):
```bash
# Local: http://localhost
# Remote: Set up SSH tunnel (see README)
```

### Option 2: Native Deployment (Production)

**Advantages**: Fine-grained control, better performance, easier debugging

**Disadvantages**: Manual dependency management, more complex setup

**Directory Structure:**
```
unmute/
├── dockerless/          # Startup scripts
│   ├── start_backend.sh
│   ├── start_stt.sh
│   ├── start_tts.sh
│   ├── start_llm.sh
│   └── start_frontend.sh  (optional)
├── services/
│   └── moshi-server/
│       └── configs/      # Service configurations
│           ├── stt.toml
│           ├── tts.toml
│           ├── stt-prod.toml
│           └── tts-prod.toml
└── unmute/              # Backend Python code
```

**Steps:**

1. Install system dependencies:
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust and Cargo
curl https://sh.rustup.rs -sSf | sh

# Install pnpm (optional, for frontend)
curl -fsSL https://get.pnpm.io/install.sh | sh -

# Verify CUDA 12.1+ is installed
nvcc --version
```

2. Clone repository:
```bash
git clone https://github.com/kyutai-labs/unmute.git
cd unmute
```

3. Configure service ports and URLs:

Edit service configurations as needed:
- STT service: `services/moshi-server/configs/stt.toml` (default port: 8080)
- TTS service: `services/moshi-server/configs/tts.toml` (default port: 8080)
- Backend: Expects `KYUTAI_STT_URL=ws://localhost:8080`, `KYUTAI_TTS_URL=ws://localhost:8080`, `KYUTAI_LLM_URL=http://localhost:8000`

4. Start services in separate terminals or tmux sessions:

**Terminal 1: STT Service**
```bash
cd dockerless
./start_stt.sh
# Listens on: ws://localhost:8090
```

**Terminal 2: TTS Service**
```bash
cd dockerless
./start_tts.sh
# Listens on: ws://localhost:8089
```

**Terminal 3: LLM Service**
```bash
cd dockerless
./start_llm.sh
# Listens on: http://localhost:11434
# Note: Adjust model in script based on VRAM
```

**Terminal 4: Backend**
```bash
cd dockerless
# Set environment variables
export KYUTAI_STT_URL=ws://localhost:8090
export KYUTAI_TTS_URL=ws://localhost:8089
export KYUTAI_LLM_URL=http://localhost:11434
export KYUTAI_LLM_MODEL=llama3.2:1b  # Adjust based on LLM server

./start_backend.sh
# Listens on: http://localhost:8000
```

**Terminal 5: Frontend (Optional)**
```bash
cd dockerless
./start_frontend.sh
# Listens on: http://localhost:3000
```

5. Verify deployment:
```bash
# Check STT
curl http://localhost:8090/health || echo "STT running (WebSocket only)"

# Check TTS
curl http://localhost:8089/health || echo "TTS running (WebSocket only)"

# Check LLM
curl http://localhost:11434/v1/models

# Check Backend
curl http://localhost:8000/v1/health

# Check all processes
ps aux | grep -E "(moshi-server|uvicorn|vllm|next)"
```

### Option 3: Systemd Services (Production Deployment)

For production on Jetson Thor, use systemd for automatic startup and management.

**Create systemd service files:**

**/etc/systemd/system/unmute-stt.service**
```ini
[Unit]
Description=Unmute STT Service
After=network.target

[Service]
Type=simple
User=unmute
WorkingDirectory=/opt/unmute
Environment="HUGGING_FACE_HUB_TOKEN=hf_your_token"
Environment="LD_LIBRARY_PATH=/opt/unmute/.venv/lib"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/unmute/.cargo/bin/moshi-server worker --config /opt/unmute/services/moshi-server/configs/stt-prod.toml --port 8090
Restart=on-failure
RestartSec=10
CPUAffinity=0-3
Nice=-10

[Install]
WantedBy=multi-user.target
```

**/etc/systemd/system/unmute-tts.service**
```ini
[Unit]
Description=Unmute TTS Service
After=network.target

[Service]
Type=simple
User=unmute
WorkingDirectory=/opt/unmute
Environment="HUGGING_FACE_HUB_TOKEN=hf_your_token"
Environment="LD_LIBRARY_PATH=/opt/unmute/.venv/lib"
Environment="CUDA_VISIBLE_DEVICES=1"
ExecStart=/opt/unmute/.cargo/bin/moshi-server worker --config /opt/unmute/services/moshi-server/configs/tts-prod.toml --port 8089
Restart=on-failure
RestartSec=10
CPUAffinity=4-9
Nice=-10

[Install]
WantedBy=multi-user.target
```

**/etc/systemd/system/unmute-llm.service**
```ini
[Unit]
Description=Unmute LLM Service (vLLM)
After=network.target

[Service]
Type=simple
User=unmute
WorkingDirectory=/opt/unmute
Environment="HUGGING_FACE_HUB_TOKEN=hf_your_token"
Environment="CUDA_VISIBLE_DEVICES=2"
ExecStart=/opt/unmute/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max-model-len 6144 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.7 \
    --port 11434
Restart=on-failure
RestartSec=15
CPUAffinity=10-15

[Install]
WantedBy=multi-user.target
```

**/etc/systemd/system/unmute-backend.service**
```ini
[Unit]
Description=Unmute Backend Service
After=network.target unmute-stt.service unmute-tts.service unmute-llm.service
Requires=unmute-stt.service unmute-tts.service unmute-llm.service

[Service]
Type=simple
User=unmute
WorkingDirectory=/opt/unmute
Environment="KYUTAI_STT_URL=ws://localhost:8090"
Environment="KYUTAI_TTS_URL=ws://localhost:8089"
Environment="KYUTAI_LLM_URL=http://localhost:11434"
Environment="KYUTAI_LLM_MODEL=meta-llama/Llama-3.2-1B-Instruct"
ExecStart=/opt/unmute/.venv/bin/uvicorn unmute.main_websocket:app \
    --host 0.0.0.0 \
    --port 8000 \
    --ws-per-message-deflate=false
Restart=on-failure
RestartSec=5
CPUAffinity=16-19

[Install]
WantedBy=multi-user.target
```

**Enable and start services:**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable unmute-stt unmute-tts unmute-llm unmute-backend

# Start services
sudo systemctl start unmute-stt
sudo systemctl start unmute-tts
sudo systemctl start unmute-llm
sudo systemctl start unmute-backend

# Check status
sudo systemctl status unmute-stt
sudo systemctl status unmute-tts
sudo systemctl status unmute-llm
sudo systemctl status unmute-backend

# View logs
sudo journalctl -u unmute-backend -f
```

## Process Layout & Resource Allocation

### Recommended Process Topology for Jetson Thor

**Goal**: Minimize interference between services while maximizing GPU utilization.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Jetson Thor                              │
├─────────────────────────────────────────────────────────────────┤
│  CPU Cores: 0-19 (20 cores assumed)                             │
│  GPUs: 0-2 (3 GPUs assumed, adjust for your hardware)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  STT Service                                            │   │
│  │  - CPU Affinity: Cores 0-3 (4 cores)                   │   │
│  │  - GPU: GPU 0                                            │   │
│  │  - VRAM: 2.5GB                                           │   │
│  │  - Priority: Nice -10 (high)                            │   │
│  │  - Port: 8090                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TTS Service                                            │   │
│  │  - CPU Affinity: Cores 4-9 (6 cores)                   │   │
│  │  - GPU: GPU 1                                            │   │
│  │  - VRAM: 5.3GB                                           │   │
│  │  - Priority: Nice -10 (high)                            │   │
│  │  - Port: 8089                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LLM Service (vLLM)                                     │   │
│  │  - CPU Affinity: Cores 10-15 (6 cores)                 │   │
│  │  - GPU: GPU 2                                            │   │
│  │  - VRAM: 6-14GB (depends on model)                      │   │
│  │  - Priority: Nice 0 (normal)                            │   │
│  │  - Port: 11434                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Backend Service (FastAPI)                              │   │
│  │  - CPU Affinity: Cores 16-19 (4 cores)                 │   │
│  │  - No GPU                                                │   │
│  │  - Priority: Nice -5 (high, but lower than STT/TTS)    │   │
│  │  - Port: 8000                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Rationale:**
- **STT gets priority**: Fast transcription is critical for perceived responsiveness
- **TTS has more cores**: Audio synthesis benefits from parallel processing
- **LLM is isolated**: Token generation is compute-heavy but can tolerate slight delays
- **Backend has dedicated cores**: Prevents WebSocket/orchestration delays

### Single-GPU Configuration

If only one GPU is available:

```
┌─────────────────────────────────────────────────────────────────┐
│  GPU 0: Shared across STT, TTS, LLM                             │
│  - Total VRAM: 16GB minimum, 24GB+ recommended                  │
│  - Memory sharing controlled by CUDA context                    │
│  - Services load sequentially to avoid OOM during startup       │
└──────────────────────────────────────────────────────────────────┘
```

**Single-GPU startup order:**
1. Start STT (smallest model, fast startup)
2. Start TTS (medium model)
3. Start LLM (largest model, may need adjusted max-model-len)
4. Start Backend

**Expected latency increase**: TTS latency may increase by ~300-500ms due to GPU contention.

## GPU/CPU Pinning Strategy

### CPU Affinity with `taskset`

Pin processes to specific cores to reduce context switching:

```bash
# Start STT on cores 0-3
taskset -c 0-3 moshi-server worker --config stt-prod.toml --port 8090

# Start TTS on cores 4-9
taskset -c 4-9 moshi-server worker --config tts-prod.toml --port 8089

# Start LLM on cores 10-15
taskset -c 10-15 python -m vllm.entrypoints.openai.api_server ...

# Start Backend on cores 16-19
taskset -c 16-19 uvicorn unmute.main_websocket:app ...
```

**Verification:**
```bash
# Check affinity of running process
taskset -cp $(pgrep -f moshi-server | head -1)
```

### GPU Assignment with `CUDA_VISIBLE_DEVICES`

Control which GPU each service uses:

```bash
# STT on GPU 0
CUDA_VISIBLE_DEVICES=0 moshi-server worker --config stt-prod.toml

# TTS on GPU 1
CUDA_VISIBLE_DEVICES=1 moshi-server worker --config tts-prod.toml

# LLM on GPU 2
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server ...

# Backend (no GPU)
CUDA_VISIBLE_DEVICES="" uvicorn unmute.main_websocket:app ...
```

**Verification:**
```bash
# Check GPU utilization per process
nvidia-smi pmon -i 0,1,2 -c 1
```

### Docker Resource Constraints

For Docker Compose deployments, add resource limits:

```yaml
services:
  stt:
    deploy:
      resources:
        limits:
          cpus: '4'  # 4 cores
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # GPU 0 only
              capabilities: [gpu]

  tts:
    deploy:
      resources:
        limits:
          cpus: '6'  # 6 cores
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']  # GPU 1 only
              capabilities: [gpu]
```

## Service Configuration

### STT Configuration (services/moshi-server/configs/stt-prod.toml)

Key parameters:

```toml
static_dir = "./static/"
log_dir = "/tmp/unmute_logs"
instance_name = "stt"
authorized_ids = ["public_token"]

[modules.asr]
path = "/api/asr-streaming"
type = "BatchedAsr"
lm_model_file = "hf://kyutai/stt-1b-en_fr-candle/model.safetensors"
text_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model"
audio_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors"

# Tuning parameters
asr_delay_in_tokens = 6  # Lower = faster but less accurate
batch_size = 1           # Higher = more concurrent users but higher latency
temperature = 0.25       # Lower = more deterministic
```

**Production tuning:**
- `asr_delay_in_tokens`: Keep at 6 for optimal accuracy/latency balance
- `batch_size`: Increase to 2-4 if serving multiple concurrent robot sessions
- For single-robot use: Keep `batch_size = 1` for minimum latency

### TTS Configuration (services/moshi-server/configs/tts-prod.toml)

Key parameters:

```toml
static_dir = "./static/"
log_dir = "/tmp/unmute_logs"
instance_name = "tts"
authorized_ids = ["public_token"]

[modules.tts_py]
type = "Py"
path = "/api/tts_streaming"
text_tokenizer_file = "hf://kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model"

# Tuning parameters
batch_size = 2             # Higher = more concurrent users but higher latency
cfg_coef = 2.0            # Classifier-free guidance coefficient
padding_between = 1       # Padding between audio chunks
n_q = 24                  # Number of quantization levels

[modules.tts_py.py]
log_folder = "/tmp/unmute_logs"
voice_folder = "hf-snapshot://kyutai/tts-voices/**/*.safetensors"
default_voice = "unmute-prod-website/default_voice.wav"
```

**Production tuning:**
- `batch_size`: Set to 2 for dual-session support, or 1 for single-robot minimum latency
- Voice selection: Customize `default_voice` to match robot character

### Backend Environment Variables

```bash
# Required
export KYUTAI_STT_URL=ws://localhost:8090     # STT service WebSocket URL
export KYUTAI_TTS_URL=ws://localhost:8089     # TTS service WebSocket URL
export KYUTAI_LLM_URL=http://localhost:11434  # LLM HTTP API URL
export KYUTAI_LLM_MODEL=meta-llama/Llama-3.2-1B-Instruct  # Model identifier

# Optional
export KYUTAI_LLM_API_KEY=your_api_key        # If using external LLM API
export NEWSAPI_API_KEY=your_key               # For news demo character
export LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR
```

### LLM Configuration

**For vLLM (self-hosted):**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max-model-len 6144 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.7 \
    --port 11434
```

**Key parameters:**
- `--max-model-len`: Context window size
  - 1536: Minimal, for very short conversations
  - 6144: Recommended for moderate conversation history
  - Higher: Requires more VRAM, increases latency
- `--gpu-memory-utilization`: Fraction of GPU memory to use
  - 0.7: Recommended, leaves headroom for other processes
  - 0.9: Maximum, only if GPU is dedicated to LLM
- `--dtype`: Data type for inference
  - `bfloat16`: Best balance of speed and quality
  - `float16`: Slightly faster, may reduce quality

**Model selection:**
- **Small (6GB VRAM)**: meta-llama/Llama-3.2-1B-Instruct
- **Medium (14GB VRAM)**: mistralai/Mistral-Small-3.2-24B-Instruct-2506
- **Large (22GB+ VRAM)**: google/gemma-3-12b-it

**For external LLM (OpenAI, Ollama):**

Backend environment:
```bash
# OpenAI
export KYUTAI_LLM_URL=https://api.openai.com/v1
export KYUTAI_LLM_MODEL=gpt-4o-mini
export KYUTAI_LLM_API_KEY=sk-your-key

# Ollama (local)
export KYUTAI_LLM_URL=http://localhost:11434
export KYUTAI_LLM_MODEL=llama3.2:1b
export KYUTAI_LLM_API_KEY=ollama
```

## Health Probes & Monitoring

### Health Endpoints

**Backend:**
```bash
curl http://localhost:8000/v1/health
# Response: {"status": "ok"}
```

**Prometheus Metrics:**
```bash
curl http://localhost:8000/metrics
# Returns Prometheus-formatted metrics
```

### Key Metrics to Monitor

Based on `docs/profiling.md`, monitor these Prometheus metrics:

**Latency Histograms:**
```promql
# Audio ingestion (Opus decode)
unmute_audio_ingestion_duration_ms

# STT flush pipeline
unmute_stt_flush_duration_ms

# LLM word generation
unmute_llm_word_generation_ms

# TTS processing
unmute_tts_word_processing_ms

# WebSocket send
unmute_websocket_send_duration_ms
```

**Time To First Token (TTFT):**
```promql
# STT first word
worker_stt_ttft

# LLM first token
worker_vllm_ttft

# TTS first audio
worker_tts_ttft
```

**Queue Depths (Backpressure Indicators):**
```promql
# Output queue size (STT → Backend)
worker_output_queue_size

# Emit queue size (Backend → WebSocket)
worker_emit_queue_size
```

**Error Counters:**
```promql
# Audio buffer overflows
worker_buffer_overflow_errors

# Backpressure errors
worker_backpressure_errors

# Silence timeout errors
worker_silence_timeout_errors
```

### Performance Targets for Robot Applications

Based on conformance harness thresholds:

| Metric | Target | Critical | Notes |
|--------|--------|----------|-------|
| Overall TTFT | <1000ms | >2000ms | Response start to first audio |
| STT TTFT | <100ms | >200ms | First word after speech starts |
| STT Flush | <300ms | >500ms | Critical for robot responsiveness |
| LLM TTFT | <300ms | >500ms | First token generation |
| TTS TTFT | <500ms | >1000ms | First audio chunk |
| Opus Decode | <10ms | >20ms | Per-frame latency |
| WebSocket Send | <5ms | >10ms | Event serialization + send |
| Queue Wait | <10ms | >50ms | Queue operation latency |

**Alert thresholds:**
```promql
# Alert if 95th percentile STT flush exceeds 500ms
histogram_quantile(0.95, rate(unmute_stt_flush_duration_ms_bucket[5m])) > 500

# Alert if output queue depth exceeds 50
worker_output_queue_size > 50

# Alert if any buffer overflow errors
rate(worker_buffer_overflow_errors[1m]) > 0
```

### Implementing Health Probes

**Simple HTTP health check:**
```bash
#!/bin/bash
# /opt/unmute/scripts/health_check.sh

BACKEND_URL="http://localhost:8000/v1/health"
TIMEOUT=5

if ! response=$(curl -s --max-time $TIMEOUT "$BACKEND_URL"); then
    echo "FAIL: Backend not responding"
    exit 1
fi

if ! echo "$response" | grep -q '"status":"ok"'; then
    echo "FAIL: Backend unhealthy: $response"
    exit 1
fi

echo "OK: Backend healthy"
exit 0
```

**Advanced health check with latency monitoring:**
```python
#!/usr/bin/env python3
# /opt/unmute/scripts/advanced_health_check.py

import requests
import sys

PROMETHEUS_URL = "http://localhost:8000/metrics"
THRESHOLDS = {
    "worker_vllm_ttft": 500,  # ms
    "unmute_stt_flush_duration_ms": 500,  # ms
    "worker_output_queue_size": 50,
}

def check_metrics():
    try:
        response = requests.get(PROMETHEUS_URL, timeout=5)
        response.raise_for_status()
        lines = response.text.split('\n')

        # Parse metrics (simplified)
        for metric_name, threshold in THRESHOLDS.items():
            # Extract metric values and check against thresholds
            # (Full implementation would parse Prometheus text format)
            pass

        print("OK: All metrics within thresholds")
        return 0
    except Exception as e:
        print(f"FAIL: Health check error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(check_metrics())
```

**Kubernetes/systemd health probes:**

For systemd, add to service file:
```ini
[Service]
ExecStartPost=/bin/sleep 10
ExecStartPost=/opt/unmute/scripts/health_check.sh
```

For Kubernetes (if deploying in k8s):
```yaml
livenessProbe:
  httpGet:
    path: /v1/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /v1/health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
```

## Performance Tuning

### Optimization Checklist

Based on `docs/performance_optimization_guide.md`:

- [ ] **Offload CPU-bound work**: Opus decoding runs in thread pool (✅ already implemented)
- [ ] **Bounded queues**: Consider adding queue size limits to prevent unbounded growth
- [ ] **Optimize serialization**: Use orjson instead of stdlib json for 5-10x speedup
- [ ] **Monitor GIL contention**: Profile with py-spy if seeing poor concurrent scaling
- [ ] **Reduce STT delay**: Tune `asr_delay_in_tokens` if transcription accuracy allows
- [ ] **LLM context management**: Limit conversation history to last N messages
- [ ] **TTS batching**: Increase `batch_size` for multi-user scenarios

### Common Bottlenecks

**1. High STT Flush Latency (>300ms)**

**Symptoms:**
```promql
histogram_quantile(0.95, rate(unmute_stt_flush_duration_ms_bucket[5m])) > 300
```

**Causes:**
- Network latency to STT service
- STT service overload
- GPU contention (single-GPU setup)

**Mitigations:**
- Check network: `ping localhost` (should be <1ms for local)
- Check GPU utilization: `nvidia-smi` (should not be at 100% constantly)
- Reduce `asr_delay_in_tokens` if acceptable
- Ensure STT has dedicated GPU in multi-GPU setup

**2. High Queue Depths (>50)**

**Symptoms:**
```promql
worker_output_queue_size > 50 OR worker_emit_queue_size > 50
```

**Causes:**
- Downstream consumer too slow (TTS, WebSocket)
- Backpressure building up

**Mitigations:**
- Check WebSocket send latency: Should be <5ms
- Profile emit loop: `python scripts/profile_pipeline.py`
- Consider bounded queues with backpressure handling

**3. Audio Buffer Overflows**

**Symptoms:**
```promql
rate(worker_buffer_overflow_errors[1m]) > 0
```

**Causes:**
- Client sending too fast
- Server processing too slow
- Opus decoding bottleneck

**Mitigations:**
- Check `unmute_audio_ingestion_duration_ms`: Should be <10ms
- Verify Opus decoding is running in thread pool
- Increase buffer size in `realtime_buffer.py` if needed

**4. LLM/TTS Slow First Token (>500ms)**

**Symptoms:**
```promql
histogram_quantile(0.95, rate(worker_vllm_ttft_bucket[5m])) > 500
```

**Causes:**
- LLM service cold start
- Large context window
- Network latency

**Mitigations:**
- Reduce `--max-model-len` for LLM
- Keep LLM warm with periodic health checks
- Use smaller model if quality allows
- Check LLM GPU is not shared with STT/TTS

### Profiling Tools

**1. Run performance profile:**
```bash
cd /opt/unmute
python scripts/profile_pipeline.py --output performance_report.md
```

**Outputs:**
- `performance_report.md`: Comprehensive metrics analysis
- `profile.html`: Interactive pyinstrument flamegraph
- `metrics_snapshot.json`: Raw Prometheus metrics

**2. View live metrics:**
```bash
# Start Prometheus scraper (optional)
# Or query directly:
curl http://localhost:8000/metrics | grep -E "(ttft|queue_size|duration_ms)"
```

**3. Check per-span traces:**

Set `PROFILE_ACTIVE = True` in `unmute/main_websocket.py` and access:
```
http://localhost:8000/profile
```

**4. Memory profiling (if needed):**
```bash
# Install memray
uv pip install memray

# Profile backend
memray run unmute/main_websocket.py
memray flamegraph output.bin
```

## Logging Configuration

### Log Locations

**Docker Compose:**
```bash
# Backend logs
docker logs unmute-backend-1 -f

# STT logs
docker logs unmute-stt-1 -f
# Also: ./volumes/stt-logs/

# TTS logs
docker logs unmute-tts-1 -f
# Also: ./volumes/tts-logs/

# LLM logs
docker logs unmute-llm-1 -f
```

**Native Deployment:**
```
/tmp/unmute_logs/       # STT/TTS logs (configured in .toml)
/var/log/syslog         # systemd service logs
journalctl -u unmute-*  # View systemd logs
```

### Log Levels

**Backend (Python):**
```bash
# Set via environment variable
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Or in code (unmute/main_websocket.py):
import logging
logging.basicConfig(level=logging.DEBUG)
```

**STT/TTS (Rust):**

Logging is configured in .toml files:
```toml
log_dir = "/tmp/unmute_logs"
```

Rust log levels are controlled by `RUST_LOG` environment variable:
```bash
export RUST_LOG=info  # trace, debug, info, warn, error
```

### Log Rotation

For production, configure logrotate:

**/etc/logrotate.d/unmute**
```
/tmp/unmute_logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 unmute unmute
}

/var/log/unmute/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 unmute unmute
    postrotate
        systemctl reload unmute-backend || true
    endscript
}
```

Test logrotate:
```bash
sudo logrotate -f /etc/logrotate.d/unmute
```

## Failover Procedures

### Service Recovery Strategies

**1. Automatic Restart (systemd)**

All systemd services should have:
```ini
[Service]
Restart=on-failure
RestartSec=10
```

This automatically restarts services on crash.

**2. Backend Graceful Degradation**

The backend is designed to tolerate temporary service outages:
- WebSocket connections remain open during STT/TTS/LLM failures
- Clients receive error events but connection persists
- Services automatically reconnect when available

**3. Health-Based Rejection**

TODO: Implement readiness probes that reject new connections when:
- STT backlog > threshold
- LLM queue depth > threshold
- Recent error rate > threshold

This prevents accepting sessions that cannot meet latency SLAs.

**4. Manual Failover Procedure**

If a service is unresponsive:

```bash
# Check service status
sudo systemctl status unmute-stt
sudo systemctl status unmute-tts
sudo systemctl status unmute-llm
sudo systemctl status unmute-backend

# View recent logs
sudo journalctl -u unmute-stt -n 100 --no-pager

# Restart failed service
sudo systemctl restart unmute-stt

# Verify restart
sudo systemctl status unmute-stt
curl http://localhost:8000/v1/health
```

**5. Rollback Procedure**

If new deployment fails:

```bash
# For Docker Compose
cd /opt/unmute
git checkout <previous-commit>
docker compose down
docker compose up --build -d

# For native deployment
cd /opt/unmute
git checkout <previous-commit>
sudo systemctl restart unmute-*
```

**6. Database/State Recovery**

Unmute is stateless except for:
- LLM conversation history (in-memory, per-session)
- Logs
- Downloaded model files

No database backup/recovery needed. Sessions do not persist across backend restarts.

## Troubleshooting

### Common Issues

#### 1. "Backend not responding"

**Check:**
```bash
# Is backend running?
ps aux | grep uvicorn
sudo systemctl status unmute-backend

# Check logs
sudo journalctl -u unmute-backend -n 50

# Check port
sudo netstat -tulpn | grep 8000
```

**Fix:**
```bash
# Restart backend
sudo systemctl restart unmute-backend

# Check dependencies are running
sudo systemctl status unmute-stt unmute-tts unmute-llm
```

#### 2. "STT/TTS service unavailable"

**Check:**
```bash
# Is service running?
ps aux | grep moshi-server

# Check CUDA availability
nvidia-smi

# Check logs
tail -f /tmp/unmute_logs/*.log
```

**Fix:**
```bash
# Restart service
sudo systemctl restart unmute-stt
sudo systemctl restart unmute-tts

# Check CUDA environment
echo $CUDA_VISIBLE_DEVICES
echo $LD_LIBRARY_PATH
```

#### 3. "Out of memory on GPU"

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Check:**
```bash
# GPU memory usage
nvidia-smi

# Which processes are using GPU?
nvidia-smi pmon
```

**Fix:**
- Reduce LLM `--max-model-len`
- Reduce LLM `--gpu-memory-utilization`
- Use smaller LLM model
- Kill other GPU processes
- Add more GPUs or use multi-GPU setup

#### 4. "High latency (>1s responses)"

**Check:**
```bash
# Query metrics
curl http://localhost:8000/metrics | grep -E "(ttft|flush|queue)"

# Run profiler
cd /opt/unmute
python scripts/profile_pipeline.py

# Check CPU/GPU utilization
htop
nvidia-smi
```

**Fix:**
- See "Performance Tuning" section above
- Check for CPU/GPU bottlenecks
- Verify services have CPU affinity set
- Ensure services are on dedicated GPUs (multi-GPU)
- Profile and optimize hot paths

#### 5. "Model download fails"

**Symptoms:**
```
Could not download model from hf://...
Authentication error
```

**Check:**
```bash
# Is token set?
echo $HUGGING_FACE_HUB_TOKEN

# Test token
huggingface-cli whoami
```

**Fix:**
```bash
# Set token
export HUGGING_FACE_HUB_TOKEN=hf_your_token

# Add to ~/.bashrc
echo 'export HUGGING_FACE_HUB_TOKEN=hf_your_token' >> ~/.bashrc

# Manually download models (if needed)
huggingface-cli download kyutai/stt-1b-en_fr-candle
huggingface-cli download kyutai/tts-1.6b-en_fr
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
```

#### 6. "WebSocket connection fails"

**Symptoms:**
- Client cannot connect
- Connection immediately closes

**Check:**
```bash
# Backend accessible?
curl http://localhost:8000/v1/health

# WebSocket endpoint exists?
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: test" \
  http://localhost:8000/v1/realtime
```

**Fix:**
- Verify backend is running
- Check firewall rules
- For remote access, use SSH tunnel (see README)
- Check `--ws-per-message-deflate=false` flag is set

#### 7. "Rust build fails"

**Symptoms:**
```
error: could not compile `moshi-server`
```

**Check:**
```bash
# CUDA installed?
nvcc --version

# Rust installed?
cargo --version

# Python environment?
echo $LD_LIBRARY_PATH
```

**Fix:**
```bash
# Ensure CUDA 12.1+ installed
# Ensure LD_LIBRARY_PATH set:
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# Fix for GCC 15 (if applicable):
export CXXFLAGS="-include cstdint"

# Clean and rebuild
cargo clean
cargo install --features cuda moshi-server@0.6.4
```

### Debug Mode

Enable debug logging:

```bash
# Backend
export LOG_LEVEL=DEBUG
sudo systemctl restart unmute-backend

# STT/TTS (Rust)
export RUST_LOG=debug
sudo systemctl restart unmute-stt unmute-tts

# View logs
sudo journalctl -u unmute-backend -f
tail -f /tmp/unmute_logs/*.log
```

Enable development UI features:

In `frontend/src/hooks/useKeyboardShortcuts.ts`:
```typescript
const ALLOW_DEV_MODE = true;  // Change from false
```

Then press:
- `S`: Show subtitles
- `D`: Show debug view with metrics

### Performance Debugging

See `docs/profiling.md` for comprehensive profiling guide.

Quick commands:
```bash
# Generate performance report
python scripts/profile_pipeline.py

# View flamegraph
# Open profile.html in browser

# Monitor real-time metrics
watch -n 1 'curl -s http://localhost:8000/metrics | grep -E "(queue_size|ttft)"'

# Check trace spans
# Set PROFILE_ACTIVE=True in unmute/main_websocket.py
# Then visit http://localhost:8000/profile
```

## Known Limitations

### Architecture Compatibility

- **ARM64 (aarch64) not officially supported**: The README states x86_64 is required. Jetson Thor uses ARM architecture, so this deployment may require:
  - Building Rust components from source for ARM64
  - Ensuring CUDA compatibility with ARM64 (Jetson ships with ARM-compatible CUDA)
  - Testing and validating all components on ARM hardware

**Status**: This guide assumes work is underway to enable ARM64 support or that compatible hardware is being used for testing.

### Model Loading Time

- STT/TTS services take 30-60 seconds to start (model download + CUDA initialization)
- LLM can take 1-2 minutes to start with large models
- First inference is slower than subsequent ones (CUDA warmup)

**Mitigation**: Use systemd for automatic startup on boot, or keep services running continuously.

### GPU Memory Constraints

- Running all services on a single 16GB GPU is tight
- Model sizes are fixed (cannot be quantized further without retraining)
- VRAM usage increases with batch size and context length

**Mitigation**: Use multi-GPU setup or external LLM API to offload compute.

### Network Latency Sensitivity

- WebSocket is sensitive to network jitter
- High packet loss or latency >50ms degrades experience
- No built-in network resilience (no automatic reconnection)

**Mitigation**: Use wired Ethernet, ensure low-latency network, implement client-side reconnection logic.

### Stateless Sessions

- Conversation history does not persist across backend restarts
- No long-term memory or user profiles
- Each WebSocket session is independent

**Mitigation**: Implement external conversation logging if persistence is needed.

## Validation Checklist

Use this checklist to verify successful deployment:

### Initial Setup
- [ ] CUDA 12.1+ installed and `nvidia-smi` works
- [ ] Hugging Face token set and models downloadable
- [ ] Sufficient GPU VRAM available (16GB minimum)
- [ ] All dependencies installed (Docker or native)

### Service Startup
- [ ] STT service starts and listens on port 8090
- [ ] TTS service starts and listens on port 8089
- [ ] LLM service starts and listens on port 11434
- [ ] Backend service starts and listens on port 8000
- [ ] Health endpoint returns `{"status": "ok"}`

### Performance Validation
- [ ] STT TTFT < 100ms (check metrics)
- [ ] STT flush latency < 300ms
- [ ] LLM TTFT < 300ms
- [ ] TTS TTFT < 500ms
- [ ] Overall response TTFT < 1000ms
- [ ] No buffer overflow errors
- [ ] Queue depths stay < 50

### End-to-End Testing
- [ ] WebSocket connection successful from client
- [ ] Audio streaming works (send Opus frames)
- [ ] STT transcription received in real-time
- [ ] LLM generates response after speech pause
- [ ] TTS audio received and playable
- [ ] Interruption handling works (stop mid-sentence)
- [ ] Session cleanup on disconnect

### Production Readiness
- [ ] Systemd services configured and enabled
- [ ] CPU/GPU affinity set correctly
- [ ] Log rotation configured
- [ ] Monitoring/alerting set up
- [ ] Health probes integrated
- [ ] Failover procedures documented
- [ ] Backup/rollback process tested

## Additional Resources

- **Profiling Guide**: `docs/profiling.md` - Comprehensive performance instrumentation
- **Performance Optimization**: `docs/performance_optimization_guide.md` - Bottleneck analysis and mitigations
- **Protocol Documentation**: `docs/browser_backend_communication.md` - WebSocket protocol details
- **Main README**: `README.md` - General setup and usage
- **OpenAI Realtime API**: https://platform.openai.com/docs/guides/realtime - Protocol reference
- **Kyutai Models**: https://github.com/kyutai-labs/delayed-streams-modeling - STT/TTS details

## Support

For issues specific to Jetson Thor deployment:
1. Check this guide's troubleshooting section
2. Review profiling and performance optimization docs
3. Open an issue: https://github.com/kyutai-labs/unmute/issues
4. Include:
   - Hardware specs (CPU, GPU, RAM)
   - Deployment method (Docker/native/systemd)
   - Logs from all services
   - Performance metrics (if available)
   - Steps to reproduce the issue

## TODO: Future Work

Items to complete for full Jetson Thor production readiness:

- [ ] **Validate ARM64 compatibility**: Test all components on actual Jetson Thor hardware
- [ ] **Implement readiness probes**: Add health checks that gate new sessions based on realtime capacity (STT backlog, LLM queue depth, actuator availability)
- [ ] **Add automatic reconnection**: Client-side logic to handle backend restarts gracefully
- [ ] **Benchmark on target hardware**: Run conformance harness on Jetson Thor, capture latency metrics
- [ ] **Optimize for single-GPU**: Further tune for memory-constrained deployments
- [ ] **Create deployment automation**: Ansible playbook or deployment script for one-command setup
- [ ] **Add monitoring dashboard**: Grafana dashboard for real-time metrics visualization
- [ ] **Document tool calling**: Add guide for robot actuator integration (move_arm, capture_frame, etc.)
- [ ] **Implement bounded queues**: Add backpressure handling to prevent unbounded queue growth
- [ ] **Add graceful shutdown**: Ensure in-flight requests complete before service stops

---

**Document Version**: 1.0
**Last Updated**: 2026-01-27
**Status**: Initial draft based on profiling findings and architecture documentation
