#!/usr/bin/env python3
"""CosyVoice 3 sidecar — runs in the isolated `cosyvoice` conda env and exposes
a small HTTP API on 127.0.0.1:8101 so PolyTTS's CosyvoiceEngine can drive it
without importing CosyVoice's conflicting deps (CosyVoice pins torch 2.3.1; the
main PolyTTS venv runs torch 2.12 for qwen/voxcpm).

Run:  conda run -n cosyvoice python cosyvoice_worker.py
PolyTTS orchestrates VRAM: POST /load loads the model, /unload frees it.
"""
import os
import sys
import io
import gc
import threading

COSYVOICE_REPO = os.environ.get("COSYVOICE_REPO", "/home/ubuntu/CosyVoice")
MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR",
                           f"{COSYVOICE_REPO}/pretrained_models/Fun-CosyVoice3-0.5B")
PORT = int(os.environ.get("COSYVOICE_PORT", "8101"))

# CosyVoice package + its Matcha-TTS submodule must be importable
for p in (COSYVOICE_REPO, f"{COSYVOICE_REPO}/third_party/Matcha-TTS"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from cosyvoice.cli.cosyvoice import AutoModel

SYS_PREFIX = "You are a helpful assistant"
app = FastAPI(title="CosyVoice sidecar")
_model = None
_lock = threading.Lock()


def _ensure_loaded():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                print(f"[cosyvoice] loading model from {MODEL_DIR} ...", flush=True)
                _model = AutoModel(model_dir=MODEL_DIR)
                print(f"[cosyvoice] loaded. sr={_model.sample_rate}", flush=True)
    return _model


class TTSReq(BaseModel):
    text: str
    voice_wav_path: str
    ref_text: str = ""
    instruct: str | None = None        # emotion/style instruction, e.g. "请用温暖的语气"
    speed: float | None = None          # 0.5..2.0 (passed as part of instruct if set)


def _synth(req: TTSReq):
    m = _ensure_loaded()
    inst = req.instruct or ""
    if req.speed:
        inst = (inst + " " if inst else "") + f"语速设为{req.speed}"
    with _lock:  # serialize GPU access
        if inst:
            prompt = f"{SYS_PREFIX}. {inst}<|endofprompt|>"
            gen = m.inference_instruct2(req.text, prompt, req.voice_wav_path, stream=False)
        else:
            prompt = f"{SYS_PREFIX}<|endofprompt|>{req.ref_text}"
            gen = m.inference_zero_shot(req.text, prompt, req.voice_wav_path, stream=False)
        pieces = []
        for j in gen:
            t = j["tts_speech"]
            if t.dim() > 1:
                t = t.squeeze(0)
            pieces.append(t.cpu().float())
    audio = torch.cat(pieces).numpy() if pieces else np.zeros(0, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32), m.sample_rate


@app.post("/load")
def load():
    _ensure_loaded()
    return {"loaded": True, "sample_rate": _model.sample_rate}


@app.post("/unload")
def unload():
    global _model
    with _lock:
        _model = None
    gc.collect()
    torch.cuda.empty_cache()
    return {"loaded": False}


@app.get("/health")
def health():
    return {"loaded": _model is not None, "model_dir": MODEL_DIR}


@app.post("/tts")
def tts(req: TTSReq):
    try:
        audio, sr = _synth(req)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Sample-Rate": str(sr)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
