<#
.SYNOPSIS
    Start the PolyTTS server on Windows (native PowerShell — no WSL/Git Bash needed).

.DESCRIPTION
    Windows always uses the PyTorch backend (MLX is Apple Silicon only): CUDA if an
    NVIDIA GPU + CUDA-enabled torch is present, otherwise CPU. Mirrors run.sh:
    resolves a Python interpreter, then restarts the server up to 10 times on crash.

    Run setup.ps1 first to create the venv and download models.
#>
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Windows is never the MLX runtime; default to pytorch but honour an override.
if (-not $env:POLYTTS_RUNTIME) { $env:POLYTTS_RUNTIME = 'pytorch' }

function Resolve-Python {
    $venvPy = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
    if (Test-Path $venvPy) { return $venvPy }
    foreach ($cmd in 'python', 'py') {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) { return $found.Source }
    }
    return $null
}

$python = Resolve-Python
if (-not $python) {
    Write-Host 'Python environment not found.' -ForegroundColor Red
    Write-Host 'Run .\setup.ps1 to create the local venv.'
    exit 1
}

# Verify core deps are importable; point at setup.ps1 if not.
& $python -c 'import fastapi, soundfile' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python dependencies are missing for $python." -ForegroundColor Red
    Write-Host 'Run .\setup.ps1 to install them.'
    exit 1
}

$maxRestarts = 10
$cooldown = 3
$restarts = 0

while ($true) {
    Write-Host "==> Starting PolyTTS server (restart #$restarts)…" -ForegroundColor Green
    & $python server.py
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host '==> Server exited cleanly.'
        break
    }

    $restarts++
    if ($restarts -ge $maxRestarts) {
        Write-Host "==> Crashed $restarts times — giving up." -ForegroundColor Red
        exit 1
    }

    Write-Host "==> Server crashed (exit $exitCode). Restarting in ${cooldown}s… ($restarts/$maxRestarts)" -ForegroundColor Yellow
    Start-Sleep -Seconds $cooldown
}
