<p align="center">
  <img src="assets/header.png" alt="polytts — text to speech" width="100%">
</p>

<h1 align="center">polytts</h1>

<p align="center">
  <b>A self-hosted text-to-speech server with voice cloning.</b><br>
  Give it a few seconds of reference audio, then turn any text into speech in that voice.
</p>

---

**polytts** is a small HTTP server that turns text into natural speech and can
**clone a voice from a short sample**. Upload a few seconds of reference audio
plus its transcript to mint a `voice_id`, then synthesize unlimited speech in
that voice. It's **engine-agnostic** — the same API drives two interchangeable
TTS models today, [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
and [VoxCPM](https://huggingface.co/openbmb/VoxCPM2), and is built so more can be
added behind it. It's the text-to-speech half of the `poly*` family, alongside
its sibling [polyasr](https://github.com/zigzag-tech/polyasr) (speech-to-text).

### What you can do with it

- 🗣️ **Clone a voice** — upload a short reference clip + transcript, get a stable
  `voice_id` you can reuse forever. (`POST /voices`)
- 🔊 **Synthesize speech** — send text + a `voice_id`, get back a WAV in that
  voice, with control over language, temperature, and prosody. (`POST /tts`)
- 🎚️ **Consistent tone** — VoxCPM voices can lock a tone seed so prosody stays
  steady across a whole video instead of drifting sentence to sentence.

### Why it exists

- **One API, many engines & GPUs.** The same contract runs on Apple Silicon
  (MLX, quantized & fast) and on NVIDIA / Apple via PyTorch (CUDA or MPS) —
  switch engines per-request without touching the client.
- **Shares the GPU politely.** On the multi-engine path it keeps **at most one
  model in VRAM**, loading lazily and **idle-evicting** after a timeout, so
  polytts co-exists with other GPU workloads (such as
  [polyasr](https://github.com/zigzag-tech/polyasr)) on a single card.
- **Built for narration.** Streaming PCM output, per-request generation
  controls, and voice/tone consistency aimed at long-form Chinese narration.

## Installation

### Prerequisites

- **Python 3.13+**
- **16 GB RAM** minimum (32 GB recommended for 1.7B model)
- **~8 GB disk** for models + dependencies
- **Backend-specific:**
  - **MLX** (recommended on Mac): Apple Silicon (M1/M2/M3/M4)
  - **PyTorch + MPS**: Apple Silicon, macOS
  - **PyTorch + CUDA**: Linux or Windows with an NVIDIA GPU

### Quick start

PolyTTS is a standalone Python server. It is also vendored into Voxlert as a git submodule at `cli/polytts`, but you can clone and run it on its own:

```bash
git clone https://github.com/zigzag-tech/polytts.git
cd polytts
```

**macOS / Linux:** Run the setup script, then start the server:

```bash
# 1. Run first-time setup (venv, deps, model download)
./setup.sh

# 2. Start the server (MLX backend by default on Mac; see Backends below)
./run.sh

# Or run it from a uv-managed environment
uv run ./run.sh

# 3. Point Voxlert at it
voxlert config set tts_backend qwen
```

**Windows:** Use the native PowerShell scripts (no WSL or Git Bash needed). Windows
runs the PyTorch backend — CUDA if you have an NVIDIA GPU, otherwise CPU.

```powershell
# 1. First-time setup (venv, deps, model download)
.\setup.ps1

# 2. Start the server
.\run.ps1
```

For an NVIDIA GPU, install a CUDA-enabled `torch` build (the default pip install is
CPU-only) — see the note at the top of `setup.ps1`. The bash scripts also work under
**WSL** or **Git Bash** if you prefer.

Generate speech directly:

```bash
VOICE_ID="$(
  curl -sS -X POST http://localhost:8100/voices \
    -F audio=@reference.wav \
    -F ref_text='大家好，欢迎来到课程。' \
    -F x_vector_only_mode=true |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["voice_id"])'
)"

curl -X POST http://localhost:8100/tts \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"你好，这是一段用于测试的示例文本。\", \"voice_id\": \"$VOICE_ID\", \"language\": \"Chinese\"}" \
  --output hello.wav
```

## Backends

| Backend | Best for | Runtime flag | Models |
|---------|----------|--------------|--------|
| **MLX** | Apple Silicon Macs (quantized, fast) | `POLYTTS_RUNTIME=mlx` (default on Mac) | Different 8-bit model; **downloaded automatically** when the server starts with MLX |
| **PyTorch + MPS** | Apple Silicon Macs (full precision) | `POLYTTS_RUNTIME=pytorch` on macOS | Same as CUDA — see below |
| **PyTorch + CUDA** | Linux/Windows with NVIDIA GPU | `POLYTTS_RUNTIME=pytorch` when CUDA is available | **Same** HuggingFace models as MPS; `./setup.sh` downloads them |

**PyTorch (MPS and CUDA)** use the same model checkpoints (`Qwen/Qwen3-TTS-12Hz-1.7B-Base` and optionally `0.6B`). No separate download for CUDA — run `./setup.sh` once; it downloads the PyTorch models and works on both Apple (MPS) and Linux/Windows (CUDA). **MLX** uses a different, quantized model and fetches it on first run.

The server chooses PyTorch device automatically: CUDA if available, else MPS (Apple), else CPU.

Example — run with PyTorch (MPS on Mac, or CUDA on Linux/Windows):

```bash
POLYTTS_RUNTIME=pytorch POLYTTS_MODEL=0.6B ./run.sh
```

### Multi-engine manager (non-MLX runtimes)

On any non-MLX runtime the server runs a **multi-engine manager** that keeps **at
most one model in VRAM at a time** — built for GPUs shared with other workloads.

- **Engines:** `qwen` (Qwen3-TTS) and `voxcpm` (VoxCPM2). Models load **lazily** on
  first use; switching engines unloads the previous one; the resident model is
  also **evicted after `POLYTTS_IDLE_EVICT_SECONDS`** of inactivity, returning
  VRAM to the driver. `GET /health` reports the resident engine and live VRAM.
- **Choosing an engine:** a voice is registered under an engine (`engine=` form
  field on `POST /voices`, default `qwen`); requests route to that engine, or
  override per-request with `"engine"` in the `/tts` body.
- **VoxCPM voices & tone:** a VoxCPM voice is a reference clip (timbre) plus an
  optional **tone seed** (`seed_audio` + `seed_text` on `POST /voices`). Every
  generation continues that locked tone, so prosody stays consistent across a
  long video instead of drifting per sentence. Output is 48 kHz.
- **Requires Python <3.13** (VoxCPM constraint). The MLX path is unaffected.

```bash
# CUDA box: multi-engine (Qwen + VoxCPM), evict after 2 min idle
POLYTTS_RUNTIME=pytorch POLYTTS_IDLE_EVICT_SECONDS=120 ./run.sh
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYTTS_RUNTIME` | `mlx` | Backend: `mlx` or `pytorch` |
| `POLYTTS_MLX_MODEL` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | HuggingFace model ID for MLX |
| `POLYTTS_MODEL` | `1.7B` | PyTorch model size: `1.7B` or `0.6B` |
| `POLYTTS_TIMEOUT` | `600` | Per-request generation timeout in seconds |
| `POLYTTS_IDLE_EVICT_SECONDS` | `120` | Manager path: evict the resident model after this many idle seconds |
| `POLYTTS_DEFAULT_ENGINE` | `qwen` | Manager path: engine for voices/requests that don't specify one |
| `VOXCPM_MODEL_ID` | `openbmb/VoxCPM2` | HuggingFace model ID for the VoxCPM engine |
| `VOXCPM_CFG_VALUE` | `3.3` | VoxCPM guidance scale |
| `VOXCPM_TIMESTEPS` | `10` | VoxCPM diffusion inference steps |

## API endpoints

### `POST /voices`

Register a cloned voice from a reference WAV and transcript.

**Request:** multipart form data

| Field | Required | Notes |
|-------|----------|-------|
| `audio` | yes | Reference WAV/audio file |
| `ref_text` | yes | Transcript matching the reference audio |
| `x_vector_only_mode` | no | `true` uses speaker embedding only; `false` uses ICL/reference-code cloning |

For synthetic reference voices, prefer `x_vector_only_mode=true`. It keeps the
speaker color while avoiding the machine-generated cadence in the reference
clip.

**Response:**

```json
{"voice_id": "43b93e137986c16b"}
```

```bash
curl -X POST http://localhost:8100/voices \
  -F audio=@reference.wav \
  -F ref_text='大家好，欢迎来到课程。' \
  -F x_vector_only_mode=true
```

### `POST /tts`

Generate speech from text using a registered `voice_id`.

**Request:**

```json
{
  "text": "你好，这是一段用于测试的示例文本。",
  "voice_id": "43b93e137986c16b",
  "language": "Chinese",
  "temperature": 0.95,
  "subtalker_temperature": 0.95,
  "top_p": 1.0
}
```

**Response:** `audio/wav` (PCM 16-bit)

**Errors:** `404` if `voice_id` is not found, `504` if generation exceeds `POLYTTS_TIMEOUT_SECONDS` (default 600 s).

```bash
curl -X POST http://localhost:8100/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "你好，这是一段用于测试的示例文本。", "voice_id": "43b93e137986c16b", "language": "Chinese"}' \
  --output speech.wav
```

Supported PyTorch generation fields include `language`, `temperature`,
`top_k`, `top_p`, `repetition_penalty`, `subtalker_temperature`,
`subtalker_top_k`, `subtalker_top_p`, `max_new_tokens`, and
`non_streaming_mode`. The server defaults `language` to `Chinese`; callers
should still send it explicitly for production Chinese narration.

### `GET /health`

Returns server status, loaded model, runtime, and available voices.

```bash
curl http://localhost:8100/health | python3 -m json.tool
```

```json
{
    "model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "runtime": "mlx",
    "device": "apple-silicon-mlx",
    "voices": ["32230314c32ab3e5", "43b93e137986c16b", "..."]
}
```

`device` can be `apple-silicon-mlx`, `mps`, `cuda`, or `cpu`.

## Scripts reference

| Script | Purpose |
|--------|---------|
| `server.py` | FastAPI TTS server (the main application) |
| `run.sh` | Starts the server using `venv/bin/python`, `python`, or `python3` |
| `setup.sh` | First-time setup: creates or repairs `venv`, installs deps, downloads models |

### The MLX path needs `livestack-node`, and nothing installs it for you

On Apple Silicon (`POLYTTS_RUNTIME=mlx`) `server.py` hard-imports
`livestack_node`, so polytts can publish its Metal footprint and let a host
broker evict the heavy voxcpm engine under pressure. It is a private package and
is deliberately absent from `requirements-mlx.txt` (see the note there), so an
MLX host needs it installed by hand into the venv `run.sh` will resolve:

```bash
./setup.sh                                   # creates venv/
venv/bin/pip install -e ../../livestack/node-py
```

`run.sh` prefers `venv/bin/python`, and only falls back to a bare `python3` +
`uv` bootstrap when no venv exists. That fallback **cannot** supply
`livestack-node` — uv installs from the requirements files, and this package is
not in them — so an MLX host with no venv will import-fail on every start. If
you see `ModuleNotFoundError: No module named 'livestack_node'`, the venv is
missing, not the dependency.

## Voices

Uploaded voices live in `polytts/voices/`. Each voice directory contains:

- **`meta.json`** — metadata including `ref_text` and `x_vector_only_mode`
- **`voice.wav`** — a short reference audio clip of the target voice

The server reads all voices at startup and caches them. Only voices that have
both `voice.wav` and a non-empty `ref_text` in `meta.json` are loaded.

## Auto-start on boot

### macOS (LaunchAgent)

Create `~/Library/LaunchAgents/com.voxlert.polytts.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.voxlert.polytts</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/FULL/PATH/TO/cli/polytts/run.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/YOU/Library/Logs/polytts.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOU/Library/Logs/polytts.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

Replace `/FULL/PATH/TO/` and `/Users/YOU/` with real paths.

`KeepAlive`/`SuccessfulExit=false` retries on any non-zero exit. Without it a
single failed start is permanent: launchd never retries and the server is
silently down until someone notices. A clean `exit(0)` still stops the job.

> **If polytts lives on an external volume (macOS):** the agent will fail with a
> bare `/bin/bash: …/run.sh: Operation not permitted` — EPERM on a 755 file,
> same uid. That is macOS TCC (`kTCCServiceSystemPolicyRemovableVolumes`), not
> POSIX. launchd agents have no responsible app, so they get no grant; your
> terminal works because *it* holds Full Disk Access and children inherit the
> responsible process's grant — so testing from a shell will not reproduce it.
> Fix: grant Full Disk Access to the binary launchd execs (`/bin/bash` here) in
> System Settings → Privacy & Security, or keep the code+venv on the internal
> disk. Worse, a process that is *already running* parks uninterruptibly in
> `__WAITING_ON_APPROVAL_FROM_SANDBOXD__` on its first cold file open — at 0%
> CPU, while `/health` keeps returning 200.

Then load it:

```bash
# Load (starts immediately and on every future login)
launchctl load ~/Library/LaunchAgents/com.voxlert.polytts.plist

# Unload
launchctl unload ~/Library/LaunchAgents/com.voxlert.polytts.plist

# Check status
launchctl list | grep polytts

# View logs
tail -f ~/Library/Logs/polytts.log
```

**Note:** `run.sh` restarts the server up to 10 times, 3 s apart, before giving
up — but that budget only bounds crashes *of the server*. It does not bound
relaunches of the script, and every supervisor config documented here
(`SuccessfulExit=false`, `Restart=on-failure`, `Restart=always`) relaunches
`run.sh` when it exits non-zero. So a failure that is fatal at *startup* —
a missing dependency, an unreadable model path — is not a bounded 10-crash
budget; it is an unbounded hot loop, and it writes to `StandardOutPath` forever.
Watch the log size after any change to requirements or model config, and rotate
that file. (A missing `requests` did exactly this on one host: 44,728 relaunches
and a 60 MB log before anyone looked.)

### Linux (systemd user service)

Create `~/.config/systemd/user/polytts.service`:

```ini
[Unit]
Description=PolyTTS server (Voxlert)
After=network.target

[Service]
Type=simple
ExecStart=/bin/bash /FULL/PATH/TO/cli/polytts/run.sh
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Replace `/FULL/PATH/TO/` with the real path. Then enable it:

```bash
# Reload, enable (auto-start on login), and start now
systemctl --user daemon-reload
systemctl --user enable --now polytts

# Check status
systemctl --user status polytts

# View logs
journalctl --user -u polytts -f

# Stop / disable
systemctl --user disable --now polytts
```

**Note:** For the service to run without an active login session, enable lingering: `loginctl enable-linger $USER`.

### Linux (systemd system service)

For a headless GPU box that must serve without any login (e.g. a shared CUDA
host running the pytorch/manager backend), use the system-wide unit shipped at
[`deploy/polytts.service`](deploy/polytts.service) instead of
the user unit above. Edit its `User`, `WorkingDirectory`, and `ExecStart` path,
then:

```bash
sudo cp deploy/polytts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now polytts   # boots on startup, Restart=always
journalctl -u polytts -f
```


## Troubleshooting

**Segfault or crash under concurrent requests**  
MLX and PyTorch MPS/CUDA are not fully thread-safe. The server serializes all inference behind a lock, but sending many requests in rapid succession can still cause memory pressure. Stick to one request at a time.

**Model not found (PyTorch backend)**  
The PyTorch backend looks for models in `models/Qwen3-TTS-12Hz-{size}-Base`. Run `./setup.sh` to download them, or manually:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='models/Qwen3-TTS-12Hz-1.7B-Base')
"
```

**MPS not available**  
Ensure you're on Apple Silicon with a recent macOS. Check with:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

**CUDA not used on Linux/Windows**  
Ensure PyTorch is installed with CUDA support and a GPU is available:

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**MLX model download fails**  
The MLX backend auto-downloads from HuggingFace on first run. If you're behind a proxy, set `HF_HUB_OFFLINE=0` and ensure `huggingface_hub` can reach the internet.

**Voice not showing in /health**
The voice needs both `voice.wav` and a non-empty `ref_text` field in
`voices/<voice_id>/meta.json`. Prefer registering voices through `POST /voices`
so the server creates the correct metadata and model-specific prompt cache.
