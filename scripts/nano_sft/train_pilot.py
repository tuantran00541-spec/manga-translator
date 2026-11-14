#!/usr/bin/env python3
"""Pilot: LoRA SFT of Qwen3.5-0.8B (uncensored base) for image -> Vietnamese.

One job only: read the text in the image and return it in Vietnamese.
No tools, no chat. Training data: pairs2 jsonl files with en_crop + vi.
Measures seconds/step on this machine and extrapolates to a full run,
so we know BEFORE committing a 6h job whether CPU training is viable.
Usage (GitHub CPU runner or local):
  python3 train_pilot.py --pairs pairs2/ch1.jsonl pairs2/ch2.jsonl \
      --model huihui-ai/Huihui-Qwen3.5-0.8B-abliterated --out pilot-out \
      --epochs 2 --batch 1 --accum 4 --lora-r 16 --lr 1e-4
"""
import argparse
import json
import math
import os
import time

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = "Dịch chữ trong ảnh sang tiếng Việt."


class PairsDataset(Dataset):
    def __init__(self, paths, processor, max_pairs=0):
        self.rows = []
        for p in paths:
            base = os.path.dirname(os.path.abspath(p))
            with open(p) as f:
                for line in f:
                    d = json.loads(line)
                    crop = d.get("en_crop")
                    vi = (d.get("vi") or "").strip()
                    if not crop or not vi:
                        continue
                    if not os.path.exists(crop):
                        rel = crop.split("/", 1)[1] if "/" in crop else crop
                        alt = os.path.join(base, rel)
                        crop = alt if os.path.exists(alt) else crop
                    if not os.path.exists(crop):
                        continue
                    self.rows.append((crop, vi))
        import random
        random.Random(0).shuffle(self.rows)
        if max_pairs:
            self.rows = self.rows[:max_pairs]
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        crop, vi = self.rows[i]
        image = Image.open(crop).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT}]}]
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + vi + self.processor.tokenizer.eos_token
        enc_p = self.processor(text=prompt, images=[image], return_tensors="pt")
        enc_f = self.processor(text=full, images=[image], return_tensors="pt")
        ids = enc_f["input_ids"][0]
        labels = ids.clone()
        labels[: enc_p["input_ids"].shape[1]] = -100
        item = {"input_ids": ids, "labels": labels,
                "pixel_values": enc_f["pixel_values"][0],
                "image_grid_thw": enc_f["image_grid_thw"][0]}
        if "mm_token_type_ids" in enc_f:
            item["mm_token_type_ids"] = enc_f["mm_token_type_ids"][0]
        return item


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(torch.cat([b["input_ids"],
                                    torch.full((n,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"],
                                 torch.full((n,), -100, dtype=torch.long)]))
        attn.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long),
                               torch.zeros(n, dtype=torch.long)]))
    out = {"input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch])}
    if "mm_token_type_ids" in batch[0]:
        out["mm_token_type_ids"] = torch.stack([
            torch.cat([b["mm_token_type_ids"],
                       torch.zeros(maxlen - len(b["input_ids"]), dtype=torch.long)])
            for b in batch])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--model", default="huihui-ai/Huihui-Qwen3.5-0.8B-abliterated")
    ap.add_argument("--out", default="pilot-out")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, device_map="cpu")
    model.config.use_cache = False
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad = False

    ds = PairsDataset(args.pairs, processor, max_pairs=args.max_pairs)
    print(f"dataset: {len(ds)} pairs", flush=True)
    if len(ds) == 0:
        raise SystemExit("no usable pairs")
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = math.ceil(len(ds) / (args.batch * args.accum))
    total_opt_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_opt_steps = min(total_opt_steps, args.max_steps)
    print(f"opt steps planned: {total_opt_steps} "
          f"({steps_per_epoch}/epoch x {args.epochs})", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_opt_steps)
    model.train()
    model.gradient_checkpointing_enable()
    step = 0
    micro = 0
    opt.zero_grad()
    t_step = time.time()
    losses = []
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for batch in loader:
            out = model(**batch)
            (out.loss / args.accum).backward()
            losses.append(out.loss.item())
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                dt = time.time() - t_step
                t_step = time.time()
                print(f"step {step}/{total_opt_steps} loss "
                      f"{sum(losses[-args.accum:]) / args.accum:.4f} "
                      f"{dt:.1f}s/step", flush=True)
                if step >= total_opt_steps:
                    done = True
                    break
    sps = None
    if step > 0 and losses:
        pass
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    total_s = time.time() - t0
    with open(os.path.join(args.out, "pilot-report.json"), "w") as f:
        json.dump({"opt_steps": step, "pairs": len(ds),
                   "epochs": args.epochs, "total_seconds": round(total_s, 1),
                   "avg_loss": round(sum(losses) / len(losses), 4) if losses else None,
                   "final_lr": sched.get_last_lr()[0]}, f, indent=2)
    print(f"PILOT DONE: {step} opt steps in {total_s:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
