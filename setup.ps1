<#
.SYNOPSIS
    First-time setup for the PolyTTS server on Windows (native PowerShell).

.DESCRIPTION
    Creates a venv, installs the PyTorch backend dependencies, and downloads the
    HuggingFace models. Windows uses the PyTorch backend only (MLX is Apple
    Silicon only). Mirrors setup.sh.

    For an NVIDIA GPU, install a CUDA-enabled torch build BEFORE running this
    (or re-install torch afterwards) from the official index, e.g.:
        venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    Otherwise pip installs the CPU build and the server runs on CPU.
#>
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Info($m) { Write-Host "==> $m" -ForegroundColor Green }
function Warn($m) { Write-Host "==> $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "==> $m" -ForegroundColor Red; exit 1 }

# ── Pre-flight: Python 3.10–3.12 (VoxCPM caps <3.13) ──────────────────────────
Info 'Checking prerequisites…'
$pyCmd = $null
foreach ($c in 'python', 'py') {
    if (Get-Command $c -ErrorAction SilentlyContinue) { $pyCmd = $c; break }
}
if (-not $pyCmd) { Fail 'Python not found. Install Python 3.10–3.12 first.' }

$pyVersion = & $pyCmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
$parts = $pyVersion.Split('.')
$maj = [int]$parts[0]; $min = [int]$parts[1]
if ($maj -ne 3 -or $min -lt 10 -or $min -gt 12) {
    Fail "Python 3.10–3.12 required (VoxCPM caps <3.13) — detected $pyVersion."
}
Write-Host "  Python: $pyVersion"

# ── Virtual environment ───────────────────────────────────────────────────────
$venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
if (Test-Path 'venv') {
    if (-not (Test-Path $venvPython)) {
        Warn 'Existing virtual environment looks stale — recreating it.'
        Remove-Item -Recurse -Force 'venv'
    } else {
        Info 'Virtual environment already exists — skipping creation.'
    }
}
if (-not (Test-Path 'venv')) {
    Info 'Creating virtual environment…'
    & $pyCmd -m venv venv
}
if (-not (Test-Path $venvPython)) { Fail 'venv created but no python.exe found in venv\Scripts\.' }

# ── Install dependencies (PyTorch backend) ────────────────────────────────────
Info 'Installing Python dependencies (PyTorch backend)…'
& $venvPython -m pip install --upgrade pip -q
& $venvPython -m pip install -r requirements.txt -q
Write-Host '  Done.'
Warn 'For NVIDIA CUDA, re-install torch from the CUDA index (see this script header).'

# ── Download models ───────────────────────────────────────────────────────────
Info 'Downloading PyTorch models (used for both CUDA and CPU)…'
New-Item -ItemType Directory -Force -Path 'models' | Out-Null

function Download-Model($repo) {
    $dest = "models/" + ($repo.Split('/')[-1])
    if (Test-Path $dest) { Write-Host "  $dest already exists — skipping."; return }
    Write-Host "  Downloading $repo …"
    & $venvPython -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', local_dir='$dest')"
    Write-Host "  Saved to $dest"
}

Download-Model 'Qwen/Qwen3-TTS-12Hz-1.7B-Base'

$choice = Read-Host 'Download the smaller 0.6B model too? [y/N]'
if ($choice -eq 'y' -or $choice -eq 'Y') {
    Download-Model 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
}

# ── Verify imports ────────────────────────────────────────────────────────────
Info 'Verifying key imports…'
& $venvPython -c "import torch; print(f'  torch {torch.__version__} (CUDA: {torch.cuda.is_available()})'); from qwen_tts import Qwen3TTSModel; print('  qwen_tts OK'); import soundfile; print('  soundfile OK'); import fastapi; print('  fastapi OK')"

Write-Host ''
Info 'Setup complete!'
Write-Host ''
Write-Host '  Start the server (PyTorch backend; uses CUDA if available):'
Write-Host '    .\run.ps1' -ForegroundColor Cyan
