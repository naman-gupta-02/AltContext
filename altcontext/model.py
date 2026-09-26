"""Model / processor loading and generation shared by baselines, evaluation and the backend."""
from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .text import build_prompt, clean_output

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def pick_device(requested: str | None = None) -> str:
    requested = requested or os.environ.get("ALTCONTEXT_DEVICE")
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    """Synchronize asynchronous devices so wall-clock timing is meaningful."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def peak_memory_mb(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def load_processor(cfg: dict):
    from transformers import AutoProcessor

    m = cfg["model"]
    proc = AutoProcessor.from_pretrained(m["base_model_id"], revision=m["base_model_revision"])
    p = cfg["processor"]
    proc.image_processor.do_image_splitting = p["do_image_splitting"]
    proc.image_processor.size = {"longest_edge": p["longest_edge"]}
    return proc


@dataclass
class LoadedModel:
    model: torch.nn.Module
    processor: object
    device: str
    adapter_path: str | None
    load_seconds: float
    cfg: dict

    @property
    def version(self) -> str:
        base = self.cfg["model"]["base_model_id"].split("/")[-1] + "@" + self.cfg["model"]["base_model_revision"][:8]
        if self.adapter_path:
            return base + "+" + Path(self.adapter_path).parent.name + "/" + Path(self.adapter_path).name
        return base


def load_model(cfg: dict, adapter_path: str | None = None, device: str | None = None, merge: bool = False) -> LoadedModel:
    from transformers import AutoModelForImageTextToText

    t0 = time.perf_counter()
    device = pick_device(device)
    m = cfg["model"]
    processor = load_processor(cfg)
    model = AutoModelForImageTextToText.from_pretrained(
        m["base_model_id"], revision=m["base_model_revision"], dtype=DTYPES[m.get("dtype", "float32")]
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        if merge:
            model = model.merge_and_unload()
    model.to(device).eval()
    sync(device)
    return LoadedModel(model, processor, device, adapter_path, time.perf_counter() - t0, cfg)


def build_chat(processor, prompt: str, target: str | None = None) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    if target is not None:
        text = text + " " + target + "<end_of_utterance>"
    return text


def open_image(path_or_image) -> Image.Image:
    if isinstance(path_or_image, Image.Image):
        img = path_or_image
    else:
        img = Image.open(path_or_image)
    if getattr(img, "is_animated", False):
        img.seek(0)
    return img.convert("RGB")


@torch.inference_mode()
def generate(lm: LoadedModel, image, context: str, mode: str = "brief", use_adapter: bool = True) -> dict:
    """Generate one description. Returns text plus timing (warm, synchronized)."""
    gen_cfg = dict(lm.cfg["generation"][mode])
    prompt = build_prompt(context, mode)
    text = build_chat(lm.processor, prompt)
    inputs = lm.processor(text=text, images=[open_image(image)], return_tensors="pt")
    inputs = {k: v.to(lm.device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(lm.model.dtype)
    adapter_ctx = nullcontext()
    if not use_adapter and hasattr(lm.model, "disable_adapter"):
        adapter_ctx = lm.model.disable_adapter()
    sync(lm.device)
    t0 = time.perf_counter()
    with adapter_ctx:
        out = lm.model.generate(**inputs, **gen_cfg, pad_token_id=lm.processor.tokenizer.pad_token_id)
    sync(lm.device)
    dt = time.perf_counter() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = lm.processor.decode(new_tokens, skip_special_tokens=True)
    return {
        "text": clean_output(raw),
        "latency_ms": round(dt * 1000, 1),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(len(new_tokens)),
    }


def environment_info(device: str) -> dict:
    import platform

    import peft
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": device,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_name": _cpu_name(),
    }


def _cpu_name() -> str:
    import platform
    import subprocess

    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return platform.processor()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
