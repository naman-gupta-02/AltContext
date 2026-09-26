"""AltContext local inference service.

Run:
  python -m backend.app                       # binds 127.0.0.1:8765
  ALTCONTEXT_FAKE_MODEL=1 python -m backend.app   # fixed responses, for extension development

Endpoints:
  GET  /health     service/model readiness (no token needed)
  POST /describe   multipart: image=<file>, metadata=<JSON>; header X-AltContext-Token
  GET  /demo/      static demo page (if demo/site exists)

Privacy: images and page text are held in memory only for the request and an
in-memory result cache; nothing is written to disk or logged.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from altcontext.config import load_config  # noqa: E402
from altcontext.text import build_context  # noqa: E402

MAX_IMAGE_BYTES = int(os.environ.get("ALTCONTEXT_MAX_IMAGE_BYTES", 10 * 2**20))
MAX_METADATA_CHARS = 20_000
MAX_PIXELS = 40_000_000
MAX_QUEUE = int(os.environ.get("ALTCONTEXT_MAX_QUEUE", 8))
TIMEOUT_S = float(os.environ.get("ALTCONTEXT_TIMEOUT_S", 90))
CACHE_SIZE = int(os.environ.get("ALTCONTEXT_CACHE_SIZE", 256))
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP"}
TOKEN_FILE = Path(os.environ.get("ALTCONTEXT_TOKEN_FILE", Path.home() / ".altcontext" / "token"))
DEFAULT_ADAPTER = ROOT / "runs" / "D_context" / "best_adapter"

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def load_or_create_token() -> str:
    if tok := os.environ.get("ALTCONTEXT_TOKEN"):
        return tok
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(tok)
    TOKEN_FILE.chmod(0o600)
    return tok


class ModelService:
    """Loads the model once in a background thread and runs one generation at a time."""

    def __init__(self, cfg: dict, adapter: str | None, fake: bool):
        self.cfg = cfg
        self.adapter = adapter
        self.fake = fake
        self.status = "loading"
        self.error = None
        self.lm = None
        self.load_seconds = None
        self.version = "fake-model" if fake else None
        self.lock = asyncio.Lock()
        self.waiting = 0

    def load(self):
        try:
            if self.fake:
                self.status = "ready"
                self.load_seconds = 0.0
                return
            from altcontext.model import load_model

            self.lm = load_model(self.cfg, adapter_path=self.adapter)
            self.version = self.lm.version
            self.load_seconds = round(self.lm.load_seconds, 2)
            self.status = "ready"
        except Exception as e:  # surfaced via /health
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"

    def run(self, image: Image.Image, context: str, mode: str) -> dict:
        if self.fake:
            time.sleep(float(os.environ.get("ALTCONTEXT_FAKE_DELAY_S", 0.3)))
            base = "A placeholder description from the fake model."
            return {"text": base + (" Context was provided." if context else " No context was provided.") + (" (detailed)" if mode == "detailed" else ""), "latency_ms": 0.0, "model_version": "fake-model"}
        from altcontext.model import generate

        # Brief mode uses the fine-tuned adapter. Detailed mode uses the base model with the
        # detailed prompt, because the adapter was trained only on brief Concadia descriptions.
        use_adapter = mode == "brief"
        g = generate(self.lm, image, context, mode, use_adapter=use_adapter)
        version = self.version if (use_adapter or not self.adapter) else self.version.split("+")[0] + " (adapter disabled)"
        return {"text": g["text"], "latency_ms": g["latency_ms"], "model_version": version}


def create_app(cfg: dict | None = None, adapter: str | None = None, fake: bool | None = None, token: str | None = None) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("ALTCONTEXT_CONFIG", "configs/default.json"))
    fake = fake if fake is not None else os.environ.get("ALTCONTEXT_FAKE_MODEL") == "1"
    if adapter is None:
        adapter = os.environ.get("ALTCONTEXT_ADAPTER") or (str(DEFAULT_ADAPTER) if DEFAULT_ADAPTER.exists() else None)
    if adapter in ("", "none"):
        adapter = None
    token = token or load_or_create_token()
    service = ModelService(cfg, adapter, fake)
    cache: OrderedDict[str, dict] = OrderedDict()

    @asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=service.load, daemon=True).start()
        yield

    app = FastAPI(title="AltContext", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.service, app.state.cache, app.state.token = service, cache, token

    @app.middleware("http")
    async def origin_guard(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and not origin.startswith("chrome-extension://") and request.url.path.startswith("/describe"):
            return JSONResponse({"error": "forbidden_origin", "message": "Requests are only accepted from the AltContext extension."}, status_code=403)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {
            "status": service.status,
            "model_version": service.version,
            "adapter": Path(service.adapter).parent.name if service.adapter else None,
            "device": service.lm.device if service.lm else None,
            "load_seconds": service.load_seconds,
            "queue": service.waiting,
            "cache_entries": len(cache),
            "error": service.error,
        }

    @app.post("/describe")
    async def describe(image: UploadFile = File(...), metadata: str = Form("{}"), x_altcontext_token: str | None = Header(None)):
        if not x_altcontext_token or not secrets.compare_digest(x_altcontext_token, token):
            raise HTTPException(401, detail={"error": "bad_token", "message": "Missing or incorrect service token. Copy it from the backend console into the extension settings."})
        if service.status == "loading":
            raise HTTPException(503, detail={"error": "model_loading", "message": "The model is still loading. Try again in a few seconds."})
        if service.status != "ready":
            raise HTTPException(503, detail={"error": "model_unavailable", "message": f"The model failed to load: {service.error}"})
        if len(metadata) > MAX_METADATA_CHARS:
            raise HTTPException(413, detail={"error": "metadata_too_large", "message": "Context metadata is too large."})
        try:
            meta = json.loads(metadata or "{}")
            assert isinstance(meta, dict)
        except Exception:
            raise HTTPException(400, detail={"error": "bad_metadata", "message": "metadata must be a JSON object."})
        mode = meta.get("mode", "brief")
        if mode not in ("brief", "detailed"):
            raise HTTPException(400, detail={"error": "bad_mode", "message": "mode must be 'brief' or 'detailed'."})

        data = await image.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, detail={"error": "image_too_large", "message": f"Image exceeds {MAX_IMAGE_BYTES // 2**20} MB."})
        if not data:
            raise HTTPException(400, detail={"error": "empty_image", "message": "No image bytes received."})
        try:
            img = Image.open(io.BytesIO(data))
            fmt = img.format
            if fmt not in ALLOWED_FORMATS:
                raise HTTPException(415, detail={"error": "unsupported_format", "message": f"Unsupported image format {fmt}. Supported: {', '.join(sorted(ALLOWED_FORMATS))}."})
            img.load()
            if getattr(img, "is_animated", False):
                img.seek(0)
            img = img.convert("RGB")
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
            raise HTTPException(415, detail={"error": "undecodable_image", "message": "The image could not be decoded (SVG and some formats are not supported)."})
        if min(img.size) < 8:
            raise HTTPException(400, detail={"error": "image_too_small", "message": "Image is too small to describe."})

        fields = {k: (str(meta.get(k)) if meta.get(k) is not None else None) for k in ("caption", "heading", "paragraph", "page_title")}
        context = build_context(fields, cfg["context"]) if meta.get("use_context", True) else ""
        key = hashlib.sha256(json.dumps({
            "image": hashlib.sha256(data).hexdigest(), "context": context, "mode": mode, "model": service.version,
            "adapter_in_use": mode == "brief", "processor": cfg["processor"], "generation": cfg["generation"][mode],
        }, sort_keys=True).encode()).hexdigest()

        t0 = time.perf_counter()
        if key in cache:
            cache.move_to_end(key)
            hit = cache[key]
            return {**hit, "cached": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "generation_latency_ms": hit["latency_ms"]}

        if service.waiting >= MAX_QUEUE:
            raise HTTPException(503, detail={"error": "busy", "message": "Too many requests are queued. Try again shortly."})
        service.waiting += 1
        try:
            async with service.lock:
                try:
                    res = await asyncio.wait_for(asyncio.to_thread(service.run, img, context, mode), timeout=TIMEOUT_S)
                except asyncio.TimeoutError:
                    raise HTTPException(504, detail={"error": "timeout", "message": f"Generation took longer than {TIMEOUT_S:.0f} s."})
        finally:
            service.waiting -= 1

        result = {
            "description": res["text"],
            "model_version": res["model_version"],
            "mode": mode,
            "context_used": bool(context),
            "ai_generated": True,
            "latency_ms": res["latency_ms"],
        }
        if not result["description"]:
            raise HTTPException(500, detail={"error": "empty_output", "message": "The model returned an empty description."})
        cache[key] = result
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
        return {**result, "cached": False, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "generation_latency_ms": res["latency_ms"]}

    demo = ROOT / "demo" / "site"
    if demo.exists():
        app.mount("/demo", StaticFiles(directory=demo, html=True), name="demo")
    return app


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("ALTCONTEXT_PORT", 8765)))
    ap.add_argument("--adapter", help="path to a LoRA adapter; 'none' for the original model")
    ap.add_argument("--fake", action="store_true", help="fixed responses without loading a model")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("AltContext binds to loopback only.")
    app = create_app(adapter=args.adapter, fake=args.fake or None)
    print(f"\nAltContext backend on http://{args.host}:{args.port}")
    print(f"Extension token: {app.state.token}\n(stored in {TOKEN_FILE}; paste it into the extension settings)\n")
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
