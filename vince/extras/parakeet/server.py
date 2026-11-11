"""Parakeet speech-to-text behind an OpenAI-compatible endpoint, for a voice chat beside the GLM server.

POST /v1/audio/transcriptions   multipart: file=<audio in any container PyAV reads>, response_format=json|text|
                                verbose_json|srt|vtt  ->  NVIDIA Parakeet TDT 0.6B v3 (transformers, on the GPU, fp32)
GET  /health, GET /v1/models    (never need a key)

Settings (environment): PARAKEET_DIR (the model folder), PARAKEET_CHUNK_S (seconds a forward takes at most, 120),
PARAKEET_DTYPE (float32), PARAKEET_API_KEY (unset: no key is asked; set: `Authorization: Bearer <key>` is required).

Run:  venv/bin/uvicorn server:app --host 127.0.0.1 --port 8189
"""

from __future__ import annotations

import asyncio
import hmac
import io
import logging
import os
import threading
import time

import av
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

log = logging.getLogger("parakeet-stt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR = os.environ.get("PARAKEET_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                                                        "parakeet-tdt-0.6b-v3"))
NAME = "nvidia/parakeet-tdt-0.6b-v3"
CHUNK_S = float(os.environ.get("PARAKEET_CHUNK_S", "120"))
API_KEY = os.environ.get("PARAKEET_API_KEY", "")
RATE = 16000

app = FastAPI(title="Parakeet speech-to-text")
_stt = {"model": None, "processor": None, "lock": threading.Lock()}


def decode_audio(data: bytes, rate: int = RATE) -> np.ndarray:
    """Any container/codec PyAV reads -> mono float32 at ``rate``."""

    try:
        container = av.open(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"cannot read the audio file: {e}") from e
    resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
    parts = []
    with container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise HTTPException(400, "the file has no audio stream")
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            parts.append(out.to_ndarray().reshape(-1))
    if not parts:
        raise HTTPException(400, "the audio file is empty")
    return np.concatenate(parts).astype(np.float32)


def _load():
    if _stt["model"] is None:
        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        t0 = time.time()
        _stt["processor"] = AutoProcessor.from_pretrained(MODEL_DIR)
        dtype = getattr(torch, os.environ.get("PARAKEET_DTYPE", "float32"))
        model = AutoModelForTDT.from_pretrained(MODEL_DIR, dtype=dtype)          # on the CPU first
        # The weights come memory-mapped from the file, and on DGX OS driver 615 a copy from file-backed memory to
        # the GPU crawls at a few MB/s (a 6-minute load). Cloned into ordinary memory first it takes a second or two;
        # one tensor at a time, so the copies never add up (all at once left 2 GiB more in the process).
        with torch.no_grad():
            for t in list(model.parameters()) + list(model.buffers()):
                t.data = t.data.clone().to("cuda")
        _stt["model"] = model.to("cuda").eval()
        log.info("Parakeet loaded in %.1f s", time.time() - t0)
    return _stt["model"], _stt["processor"]


def _transcribe(audio: np.ndarray) -> str:
    import torch

    model, proc = _load()
    step = int(CHUNK_S * RATE)
    texts = []
    with _stt["lock"], torch.inference_mode():
        for i in range(0, audio.size, step):
            chunk = audio[i:i + step]
            if chunk.size < RATE // 10:               # under 0.1 s
                continue
            inputs = proc(chunk, sampling_rate=RATE, return_tensors="pt").to(model.device, dtype=model.dtype)
            out = model.generate(**inputs, return_dict_in_generate=True)
            dec = proc.decode(out.sequences, skip_special_tokens=True)
            texts.append((dec[0] if isinstance(dec, list) else dec).strip())
    return " ".join(t for t in texts if t)


def _authorized(request: Request) -> None:
    if not API_KEY:
        return
    given = request.headers.get("authorization", "")
    if not hmac.compare_digest(given.encode(), ("Bearer " + API_KEY).encode()):
        raise HTTPException(401, "this server asks for its key: Authorization: Bearer <PARAKEET_API_KEY>")


@app.on_event("startup")
async def _startup():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load)
    await loop.run_in_executor(None, _transcribe, np.zeros(RATE, np.float32))      # one second of silence: warm-up


@app.get("/health")
async def health():
    return {"ok": _stt["model"] is not None, "model": NAME, "key_required": bool(API_KEY)}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": NAME, "object": "model", "owned_by": "nvidia"},
                                       {"id": "whisper-1", "object": "model", "owned_by": "alias:" + NAME}]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request, file: UploadFile = File(...), model: str = Form(NAME),
                         language: str | None = Form(None), prompt: str | None = Form(None),
                         response_format: str = Form("json"), temperature: float | None = Form(None)):
    _authorized(request)
    data = await file.read()
    t0 = time.time()
    audio = decode_audio(data)
    duration = audio.size / RATE
    text = await asyncio.get_running_loop().run_in_executor(None, _transcribe, audio)
    log.info("transcribed %.2f s of audio in %.3f s (%d characters)", duration, time.time() - t0, len(text))
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": language, "duration": duration, "text": text,
                "segments": [{"id": 0, "start": 0.0, "end": duration, "text": text}]}
    if response_format in ("srt", "vtt"):
        def ts(s, sep):
            h, r = divmod(s, 3600)
            m, s = divmod(r, 60)
            return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", sep)
        if response_format == "srt":
            return PlainTextResponse(f"1\n{ts(0, ',')} --> {ts(duration, ',')}\n{text}\n")
        return PlainTextResponse(f"WEBVTT\n\n{ts(0, '.')} --> {ts(duration, '.')}\n{text}\n")
    return {"text": text}
