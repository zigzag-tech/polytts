#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default: MLX on Apple Silicon, PyTorch elsewhere (CUDA/MPS/CPU)
if [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
    export POLYTTS_RUNTIME=${POLYTTS_RUNTIME:-mlx}
else
    export POLYTTS_RUNTIME=${POLYTTS_RUNTIME:-pytorch}
fi

resolve_python() {
    if [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
        printf '%s\n' "$SCRIPT_DIR/venv/bin/python"
        return 0
    fi
    if [[ -x "$SCRIPT_DIR/venv/bin/python3" ]]; then
        printf '%s\n' "$SCRIPT_DIR/venv/bin/python3"
        return 0
    fi
    # A venv/ that exists but holds no usable interpreter is a BROKEN venv, not
    # an absent one, and the two must not share a fallback. Falling through here
    # picks a system python that has none of the venv's packages — so the deps
    # check below "fails", uv bootstraps an environment that cannot contain the
    # private livestack-node, and the server import-crashes forever. The venv is
    # right there with 1.4 GB of packages in it; the symlink is what died.
    # Seen on xc-mac-studio: venv/bin/python -> python3.13 -> a uv-managed
    # CPython that uv later removed. Dangling symlink, silent downgrade,
    # 44,728 relaunches.
    # Return 2, distinct from 1: a MISSING environment is what uv bootstrap is
    # for, a BROKEN one is not. Bootstrapping over it hides the breakage.
    if [[ -d "$SCRIPT_DIR/venv" ]]; then
        return 2
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    return 1
}

uv_bootstrap() {
    local uv_args=(run --with-requirements requirements-common.txt)
    if [[ "$POLYTTS_RUNTIME" == "mlx" ]] && [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
        uv_args+=(--with-requirements requirements-mlx.txt --prerelease=allow)
    else
        uv_args+=(--with-requirements requirements-pytorch.txt)
    fi
    exec env POLYTTS_UV_BOOTSTRAPPED=1 uv "${uv_args[@]}" ./run.sh
}

set +e
PYTHON_BIN="$(resolve_python)"
resolve_rc=$?
set -e

if (( resolve_rc == 2 )); then
    echo "venv/ exists but holds no usable interpreter — it is BROKEN, not missing." >&2
    echo "  Check with: ls -l venv/bin/python   (a dangling symlink is the usual cause)" >&2
    echo "  Recreate with ./setup.sh. Refusing to fall back to a system python or to" >&2
    echo "  uv-bootstrap over it: both hide the breakage and neither can supply the" >&2
    echo "  packages already sitting in venv/." >&2
    exit 1
fi

if (( resolve_rc != 0 )); then
    if command -v uv >/dev/null 2>&1 && [[ "${POLYTTS_UV_BOOTSTRAPPED:-0}" != "1" ]]; then
        echo "==> No Python interpreter found. Bootstrapping with uv…"
        uv_bootstrap
    fi
    echo "Python environment not found."
    echo "Run ./setup.sh to create the local venv, or start with uv run ./run.sh."
    exit 1
fi

if ! "$PYTHON_BIN" -c "import fastapi, soundfile, requests" >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1 && [[ "${POLYTTS_UV_BOOTSTRAPPED:-0}" != "1" ]]; then
        echo "==> Python dependencies are missing for $PYTHON_BIN. Bootstrapping with uv…"
        uv_bootstrap
    fi
    echo "Python dependencies are missing for $PYTHON_BIN."
    echo "Run ./setup.sh, or use uv with requirements."
    exit 1
fi

MAX_RESTARTS=10
COOLDOWN=3
restarts=0

while true; do
    echo "==> Starting PolyTTS server (restart #$restarts)…"
    # `set -e` (line 2) aborts the whole script the instant server.py exits
    # non-zero, so without this the crash budget below is unreachable code and
    # every restart is really the supervisor's — uncounted, uncooled. Observed
    # on xc-mac-studio: 44,728 relaunches, a 60 MB log, and neither "Restarting
    # in 3s" nor "giving up" ever printed once.
    set +e
    POLYTTS_RUNTIME=${POLYTTS_RUNTIME:-mlx} "$PYTHON_BIN" server.py
    exit_code=$?
    set -e

    if [[ $exit_code -eq 0 ]]; then
        echo "==> Server exited cleanly."
        break
    fi

    restarts=$((restarts + 1))
    if (( restarts >= MAX_RESTARTS )); then
        echo "==> Crashed $restarts times — giving up."
        exit 1
    fi

    echo "==> Server crashed (exit $exit_code). Restarting in ${COOLDOWN}s… ($restarts/$MAX_RESTARTS)"
    sleep "$COOLDOWN"
done
