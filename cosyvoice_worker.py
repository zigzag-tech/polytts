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
    # Group N sentences per generation so prosody flows within a chunk, then join
    # chunks with a controlled pause. 1 = per-sentence (choppy); large = whole-text
    # (CosyVoice decides boundaries, can cut mid-sentence / under-pause).
    sentences_per_chunk: int | None = None
    interchunk_silence: float | None = None
    # Insert a CosyVoice control token (e.g. "[breath]" / "[quick_breath]") at
    # every within-chunk sentence boundary so the model breathes there naturally
    # (fixes under-paused sentence ends inside multi-sentence chunks).
    breath_token: str | None = None


# Generate sentence-by-sentence and join with a guaranteed inter-sentence
# silence. CosyVoice's whole-text mode groups multiple sentences per chunk and
# hard-concat ran some sentence ends into the next with insufficient pause.
INTERSENTENCE_SILENCE = float(os.environ.get("COSYVOICE_INTERSENTENCE_SILENCE", "0.35"))
_SENT_ENDERS = "。！？!?\n"


def _split_sentences(text: str):
    out, buf = [], ""
    for ch in text:
        buf += ch
        if ch in _SENT_ENDERS:
            s = buf.strip()
            if s:
                out.append(s)
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out or [text]


def _gen_one(m, text, inst, ref_text, voice_wav_path):
    if inst:
        prompt = f"{SYS_PREFIX}. {inst}<|endofprompt|>"
        return m.inference_instruct2(text, prompt, voice_wav_path, stream=False)
    prompt = f"{SYS_PREFIX}<|endofprompt|>{ref_text}"
    return m.inference_zero_shot(text, prompt, voice_wav_path, stream=False)


def _synth(req: TTSReq):
    m = _ensure_loaded()
    inst = req.instruct or ""
    if req.speed:
        inst = (inst + " " if inst else "") + f"语速设为{req.speed}"
    sents = _split_sentences(req.text)
    spc = req.sentences_per_chunk or int(os.environ.get("COSYVOICE_SENTENCES_PER_CHUNK", "2"))
    gap_s = (req.interchunk_silence if req.interchunk_silence is not None
             else float(os.environ.get("COSYVOICE_INTERCHUNK_SILENCE", "0.5")))
    breath = req.breath_token or os.environ.get("COSYVOICE_BREATH_TOKEN", "")
    # Pack sentences into sentence-aligned chunks (no mid-sentence cuts); insert
    # a breath token between sentences within a chunk so every boundary breathes.
    joiner = f"{breath}" if breath else ""
    chunks_text = [joiner.join(sents[i:i + spc]) for i in range(0, len(sents), spc)]
    gap = torch.zeros(int(m.sample_rate * gap_s)) if len(chunks_text) > 1 else None
    with _lock:  # serialize GPU access across the whole multi-chunk job
        pieces = []
        for ct in chunks_text:
            seg = []
            for j in _gen_one(m, ct, inst, req.ref_text, req.voice_wav_path):
                t = j["tts_speech"]
                if t.dim() > 1:
                    t = t.squeeze(0)
                seg.append(t.cpu().float())
            if seg:
                pieces.append(torch.cat(seg))
    print(f"[cosyvoice] {len(sents)} sentences -> {len(chunks_text)} chunks "
          f"(<= {spc}/chunk), gap={gap_s}s, breath={breath or 'none'}", flush=True)
    if not pieces:
        return np.zeros(0, dtype=np.float32), m.sample_rate
    audio = pieces[0]
    for p in pieces[1:]:
        audio = torch.cat([audio, gap, p])
    return np.asarray(audio.numpy(), dtype=np.float32), m.sample_rate


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
