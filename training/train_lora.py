"""LoRA fine-tuning of SmolVLM on Concadia with answer-only loss.

The vision encoder and connector stay frozen; LoRA adapters are placed only
on language-model attention projections (see configs/default.json).

Stages (from the build guide):
  --inspect-only                       decode one processed example and its supervised span
  --train-subset tiny --epochs 30      tiny fit (debugging, not generalization)
  --train-subset pilot                 pilot run: memory, step time, save/reload
  --train-subset train_5k              main run

Examples:
  python training/train_lora.py --run-name D_context --context-mode with_context
  python training/train_lora.py --run-name C_image_only --context-mode image_only
  python training/train_lora.py --run-name D_context --resume        # continue from last checkpoint
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altcontext.config import ROOT, load_config  # noqa: E402
from altcontext.data import image_path, load_subset, record_context  # noqa: E402
from altcontext.model import DTYPES, build_chat, environment_info, load_processor, open_image, pick_device, sync, write_json  # noqa: E402
from altcontext.text import build_prompt  # noqa: E402

IGNORE = -100
MAX_SEQ_TOKENS = 1024  # 64 image tokens + bounded context + target; guards against runaway inputs


# ---------------------------------------------------------------- examples --
class ExampleBuilder:
    def __init__(self, processor, cfg, context_mode: str, context_dropout: float, seed: int):
        self.processor = processor
        self.cfg = cfg
        self.context_mode = context_mode
        self.context_dropout = context_dropout
        self.seed = seed
        tok = processor.tokenizer
        self.image_token_id = tok.convert_tokens_to_ids("<image>")
        self.pad_id = tok.pad_token_id

    def drops_context(self, record) -> bool:
        if self.context_dropout <= 0:
            return False
        h = hashlib.sha256(f"{self.seed}:{record['id']}".encode()).digest()
        return (int.from_bytes(h[:8], "big") / 2**64) < self.context_dropout

    def prompt_for(self, record, train: bool = True) -> str:
        if self.context_mode == "image_only" or (train and self.drops_context(record)):
            return build_prompt("", "brief")
        return build_prompt(record_context(record, self.cfg), "brief")

    def __call__(self, record, train: bool = True) -> dict:
        img = open_image(image_path(record))
        prefix = build_chat(self.processor, self.prompt_for(record, train))
        full = build_chat(self.processor, self.prompt_for(record, train), record["target_description"])
        p_ids = self.processor(text=prefix, images=[img], return_tensors="pt")["input_ids"][0]
        enc = self.processor(text=full, images=[img], return_tensors="pt")
        ids = enc["input_ids"][0]
        if not torch.equal(ids[: len(p_ids)], p_ids):
            raise RuntimeError(f"prompt tokens are not a prefix of the full sequence for {record['id']}")
        if len(ids) > MAX_SEQ_TOKENS:
            raise ValueError(f"{record['id']}: {len(ids)} tokens exceeds MAX_SEQ_TOKENS; context bounds are not being applied")
        labels = ids.clone()
        labels[: len(p_ids)] = IGNORE
        labels[ids == self.image_token_id] = IGNORE
        if self.pad_id is not None:
            labels[ids == self.pad_id] = IGNORE
        enc["labels"] = labels.unsqueeze(0)
        return enc


def to_device(batch, device, dtype):
    out = {k: v.to(device) for k, v in batch.items()}
    if "pixel_values" in out:
        out["pixel_values"] = out["pixel_values"].to(dtype)
    return out


class ImageFeatureCache:
    """Caches frozen vision-encoder + connector outputs per image.

    The vision encoder and connector are frozen, so their outputs for a given image
    and processor setting never change during training. Feeding them back through
    `image_hidden_states` is numerically the same forward pass and ~40% faster on MPS.
    """

    def __init__(self, cfg, device):
        key = hashlib.sha256(json.dumps([cfg["model"], cfg["processor"]], sort_keys=True).encode()).hexdigest()[:12]
        self.dir = ROOT / "data" / "cache" / f"image_features_{key}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def get(self, base_model, record, enc):
        path = self.dir / f"{record['id']}.pt"
        if path.exists():
            return torch.load(path, map_location=self.device)
        with torch.no_grad():
            pam = enc.get("pixel_attention_mask")
            f = base_model.model.get_image_features(
                pixel_values=enc["pixel_values"].to(self.device),
                pixel_attention_mask=pam.to(self.device) if pam is not None else None,
            )
            f = getattr(f, "pooler_output", f).detach()
        torch.save(f.cpu(), path)
        return f

    def swap(self, base_model, record, inputs):
        feats = self.get(base_model, record, inputs)
        inputs = dict(inputs)
        inputs.pop("pixel_values", None)
        inputs.pop("pixel_attention_mask", None)
        inputs["image_hidden_states"] = feats.to(self.device)
        return inputs


def inspect_example(builder, record):
    enc = builder(record)
    tok = builder.processor.tokenizer
    ids, labels = enc["input_ids"][0], enc["labels"][0]
    sup = ids[labels != IGNORE]
    info = {
        "id": record["id"],
        "input_ids_shape": list(enc["input_ids"].shape),
        "pixel_values_shape": list(enc["pixel_values"].shape),
        "num_image_tokens": int((ids == builder.image_token_id).sum()),
        "num_supervised_tokens": int(len(sup)),
        "decoded_prompt_tail": tok.decode(ids[labels == IGNORE][-40:]),
        "decoded_supervised": tok.decode(sup),
        "target": record["target_description"],
    }
    ok = info["decoded_supervised"].strip().replace("<end_of_utterance>", "").strip() == record["target_description"].strip()
    info["mask_check_passed"] = bool(ok and info["decoded_supervised"].endswith("<end_of_utterance>"))
    return info


# ------------------------------------------------------------------- model --
def build_model(cfg, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    m = cfg["model"]
    model = AutoModelForImageTextToText.from_pretrained(m["base_model_id"], revision=m["base_model_revision"], dtype=DTYPES[m.get("dtype", "float32")])
    for p in model.parameters():
        p.requires_grad_(False)
    lc = cfg["lora"]
    lora = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules_regex"], bias="none")
    model = get_peft_model(model, lora)
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, _ in trainable if "vision_model" in n or "connector" in n or "lora_" not in n]
    if bad:
        raise RuntimeError(f"unexpected trainable parameters: {bad[:5]}")
    total = sum(p.numel() for p in model.parameters())
    n_train = sum(n for _, n in trainable)
    model.to(device)
    return model, {"trainable_params": n_train, "total_params": total, "trainable_pct": round(100 * n_train / total, 4), "lora_modules": len({n.rsplit('.lora_', 1)[0] for n, _ in trainable})}


@torch.no_grad()
def validation_loss(model, builder, records, device, dtype, feats=None):
    model.eval()
    losses = []
    for r in records:
        inputs = to_device(builder(r, train=False), device, dtype)
        if feats:
            inputs = feats.swap(model.base_model.model, r, inputs)
        out = model(**inputs)
        losses.append(out.loss.item())
        del out
        empty_cache(device)
    model.train()
    return float(np.mean(losses))


def empty_cache(device):
    # Variable sequence lengths fragment the MPS allocator; release cached blocks.
    if device == "mps":
        torch.mps.empty_cache()


# ------------------------------------------------------------- checkpoints --
def save_checkpoint(run_dir: Path, model, optimizer, scheduler, state: dict, keep_state: bool = True):
    ck = run_dir / f"checkpoint-{state['global_step']}"
    model.save_pretrained(ck / "adapter")
    if keep_state:
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            **state,
        }, ck / "trainer_state.pt")
    return ck


def latest_checkpoint(run_dir: Path):
    cks = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    cks = [c for c in cks if (c / "trainer_state.pt").exists()]
    return cks[-1] if cks else None


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--train-subset", default="train_5k")
    ap.add_argument("--val-subset", default="val")
    ap.add_argument("--context-mode", choices=["with_context", "image_only"], default="with_context")
    ap.add_argument("--context-dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--max-steps", type=int, help="optimizer steps cap")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--val-loss-n", type=int)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--no-feature-cache", action="store_true", help="run the vision encoder every step")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["training"]
    if args.lr:
        tc["learning_rate"] = args.lr
    if args.grad_accum:
        tc["grad_accum_steps"] = args.grad_accum
    if args.epochs:
        tc["epochs"] = args.epochs
    if args.eval_every:
        tc["eval_every_steps"] = tc["save_every_steps"] = args.eval_every
    if args.val_loss_n:
        tc["val_loss_n"] = args.val_loss_n

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = DTYPES[cfg["model"].get("dtype", "float32")]
    processor = load_processor(cfg)
    builder = ExampleBuilder(processor, cfg, args.context_mode, args.context_dropout, args.seed)
    train = load_subset(args.train_subset)
    val = load_subset(args.val_subset)[: tc["val_loss_n"]]
    run_dir = Path(args.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    inspection = [inspect_example(builder, r) for r in train[:3]]
    write_json(run_dir / "batch_inspection.json", inspection)
    for info in inspection[:1]:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    if not all(i["mask_check_passed"] for i in inspection):
        raise SystemExit("mask check failed: supervised tokens do not decode to the target")
    if args.inspect_only:
        return

    model, param_info = build_model(cfg, device)
    print(f"trainable params: {param_info}")
    feats = None if args.no_feature_cache else ImageFeatureCache(cfg, device)
    if feats:
        model.eval()
        with torch.no_grad():
            enc = to_device(builder(train[0], train=False), device, dtype)
            l_pix = model(**enc).loss.item()
            l_feat = model(**feats.swap(model.base_model.model, train[0], enc)).loss.item()
        print(f"feature-cache check: loss(pixels)={l_pix:.5f} loss(cached features)={l_feat:.5f}")
        if abs(l_pix - l_feat) > 1e-3:
            raise SystemExit("cached image features do not reproduce the pixel forward pass")
        param_info["feature_cache_check"] = {"loss_pixels": l_pix, "loss_cached": l_feat, "dir": str(feats.dir)}
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    accum = tc["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(train) / accum)
    total_steps = args.max_steps or math.ceil(steps_per_epoch * tc["epochs"])
    warmup = max(1, int(total_steps * tc["warmup_ratio"]))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    state = {"global_step": 0, "micro_step": 0, "best_val_loss": None, "best_step": None, "history": []}

    if args.resume and (ck := latest_checkpoint(run_dir)):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(ck / "adapter" / "adapter_model.safetensors")))
        s = torch.load(ck / "trainer_state.pt", weights_only=False)
        optimizer.load_state_dict(s["optimizer"])
        scheduler.load_state_dict(s["scheduler"])
        random.setstate(s["python_rng"])
        np.random.set_state(s["numpy_rng"])
        torch.set_rng_state(s["torch_rng"])
        state = {k: s[k] for k in state}
        print(f"resumed from {ck} at step {state['global_step']}")

    run_config = {
        "args": vars(args),
        "config": cfg,
        "param_info": param_info,
        "train_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in train).encode()).hexdigest(),
        "val_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in val).encode()).hexdigest(),
        "n_train": len(train),
        "n_val_loss": len(val),
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup,
        "environment": environment_info(device),
        "checkpoint_selection": "lowest validation loss on the fixed val_loss_n examples",
    }
    write_json(run_dir / "run_config.json", run_config)
    (run_dir / "train_ids.txt").write_text("\n".join(r["id"] for r in train))

    if state["global_step"] == 0 and not state["history"]:
        v0 = validation_loss(model, builder, val, device, dtype, feats)
        state["history"].append({"step": 0, "val_loss": v0})
        print(f"step 0 val_loss {v0:.4f}")

    log = open(run_dir / "train_log.jsonl", "a")
    # Deterministic example order per epoch; resuming skips consumed micro-steps.
    def order_for_epoch(ep):
        idx = list(range(len(train)))
        random.Random(args.seed * 1000 + ep).shuffle(idx)
        return idx

    total_micro = total_steps * accum
    micro = state["micro_step"]
    running, t_last = [], time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    while state["global_step"] < total_steps:
        ep, pos = divmod(micro, len(train))
        rec = train[order_for_epoch(ep)[pos]]
        inputs = to_device(builder(rec), device, dtype)
        if feats:
            inputs = feats.swap(model.base_model.model, rec, inputs)
        out = model(**inputs)
        (out.loss / accum).backward()
        running.append(out.loss.item())
        del out, inputs
        empty_cache(device)
        micro += 1
        if micro % accum == 0 or micro == total_micro:
            torch.nn.utils.clip_grad_norm_(trainable, tc["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            state["micro_step"] = micro
            sync(device)
            now = time.perf_counter()
            entry = {"step": state["global_step"], "epoch": round(micro / len(train), 3), "loss": float(np.mean(running)), "lr": scheduler.get_last_lr()[0], "sec_per_step": round(now - t_last, 2)}
            if device == "mps":
                entry["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2**20)
            elif device == "cuda":
                entry["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
            running, t_last = [], now
            gs = state["global_step"]
            if gs % tc["eval_every_steps"] == 0 or gs == total_steps:
                entry["val_loss"] = validation_loss(model, builder, val, device, dtype, feats)
                state["history"].append({"step": gs, "val_loss": entry["val_loss"]})
                if state["best_val_loss"] is None or entry["val_loss"] < state["best_val_loss"]:
                    state["best_val_loss"], state["best_step"] = entry["val_loss"], gs
                save_checkpoint(run_dir, model, optimizer, scheduler, state)
                # Only the newest checkpoint keeps optimizer state; older ones keep the adapter.
                for old in run_dir.glob("checkpoint-*"):
                    if old.name != f"checkpoint-{gs}" and (old / "trainer_state.pt").exists():
                        (old / "trainer_state.pt").unlink()
                t_last = time.perf_counter()
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry))

    best = run_dir / f"checkpoint-{state['best_step']}" / "adapter"
    final = run_dir / "best_adapter"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(best, final)
    summary = {"best_step": state["best_step"], "best_val_loss": state["best_val_loss"], "history": state["history"], "total_steps": total_steps}
    write_json(run_dir / "summary.json", summary)
    print(f"done. best step {state['best_step']} val_loss {state['best_val_loss']:.4f} -> {final}")


if __name__ == "__main__":
    main()
