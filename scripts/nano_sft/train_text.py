#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import time

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT_TMPL = "Dịch sang tiếng Việt:\n{en}"


def en_garbage(t):
    if not t or len(t.strip()) < 3:
        return True
    letters = sum(1 for c in t if c.isalpha())
    if letters / max(1, len(t)) < 0.6:
        return True
    if re.search(r"\d{2,}", t):
        return True
    if any(c in t for c in "¬`~|<>^_¿¡«»§¶†‡°"):
        return True
    return False


class TextPairsDataset(Dataset):
    def __init__(self, paths, processor, max_pairs=0, holdout=0):
        self.rows = []
        for p in paths:
            with open(p) as f:
                for line in f:
                    d = json.loads(line)
                    en = (d.get("en") or "").strip()
                    vi = (d.get("vi") or "").strip()
                    if not en or not vi:
                        continue
                    if en_garbage(en):
                        continue
                    self.rows.append((en, vi))
        random.Random(0).shuffle(self.rows)
        if max_pairs:
            self.rows = self.rows[:max_pairs]
        self.holdout_rows = self.rows[:holdout] if holdout else []
        self.rows = self.rows[holdout:] if holdout else self.rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        en, vi = self.rows[i]
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT_TMPL.format(en=en)}]}]
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + vi + self.processor.tokenizer.eos_token
        enc_p = self.processor(text=prompt, return_tensors="pt")
        enc_f = self.processor(text=full, return_tensors="pt")
        ids = enc_f["input_ids"][0]
        labels = ids.clone()
        labels[: enc_p["input_ids"].shape[1]] = -100
        return {"input_ids": ids, "labels": labels}


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
    return {"input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn)}


def _norm_text(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def eval_holdout_text(model, processor, holdout_rows, max_new_tokens=96):
    import difflib
    model.eval()
    was_cache = model.config.use_cache
    model.config.use_cache = True
    exact = 0
    sims = []
    samples = []
    for en, vi in holdout_rows:
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT_TMPL.format(en=en)}]}]
        prompt = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = processor(text=prompt, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        out = processor.tokenizer.decode(
            gen[0][enc["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()
        n_out, n_vi = _norm_text(out), _norm_text(vi)
        if n_out == n_vi:
            exact += 1
        sims.append(difflib.SequenceMatcher(None, n_out, n_vi).ratio())
        if len(samples) < 3:
            samples.append({"en": en, "target": vi, "got": out})
    model.config.use_cache = was_cache
    model.train()
    n = max(1, len(holdout_rows))
    return {"exact": round(exact / n, 3),
            "mean_sim": round(sum(sims) / len(sims), 3) if sims else 0.0,
            "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--model", default="huihui-ai/Huihui-Qwen3.5-0.8B-abliterated")
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--out", default="text-out")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--holdout", type=int, default=48)
    ap.add_argument("--time-budget-minutes", type=float, default=330.0)
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

    from peft import LoraConfig, get_peft_model, PeftModel
    if args.init_adapter:
        base = model
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        print(f"continued from adapter {args.init_adapter}", flush=True)
    else:
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                         lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj",
                                         "in_proj_qkv", "in_proj_z", "in_proj_b",
                                         "in_proj_a", "out_proj"])
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad = False

    ds = TextPairsDataset(args.pairs, processor, max_pairs=args.max_pairs,
                          holdout=args.holdout)
    print(f"dataset: {len(ds)} train pairs + {len(ds.holdout_rows)} holdout",
          flush=True)
    if len(ds) == 0:
        raise SystemExit("no usable pairs")
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = max(1, len(ds) // (args.batch * args.accum))
    total_opt_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_opt_steps = min(total_opt_steps, args.max_steps)
    print(f"opt steps planned: {total_opt_steps} "
          f"({steps_per_epoch}/epoch x {args.epochs})", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_opt_steps)
    model.train()
    model.gradient_checkpointing_enable()
    report = {"pairs": len(ds) + len(ds.holdout_rows),
              "train_pairs": len(ds), "holdout_pairs": len(ds.holdout_rows),
              "epochs": args.epochs, "lora_r": args.lora_r, "lr": args.lr,
              "accum": args.accum, "epoch_evals": [], "best_epoch": None,
              "time_budget_stop": None}

    def save_ckpt(sub):
        d = os.path.join(args.out, sub)
        os.makedirs(d, exist_ok=True)
        model.save_pretrained(d)
        return d

    def write_report():
        report["opt_steps"] = step
        report["total_seconds"] = round(time.time() - t0, 1)
        report["avg_loss"] = round(sum(losses) / len(losses), 4) if losses else None
        report["final_lr"] = sched.get_last_lr()[0]
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "text-report.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    step = 0
    micro = 0
    opt.zero_grad()
    t_step = time.time()
    losses = []
    dts = []
    done = False
    best_score = -1.0
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
                dts.append(dt)
                print(f"step {step}/{total_opt_steps} loss "
                      f"{sum(losses[-args.accum:]) / args.accum:.4f} "
                      f"{dt:.1f}s/step", flush=True)
                if step == 3 and args.time_budget_minutes > 0:
                    avg = sum(dts) / len(dts)
                    proj = (time.time() - t0) + avg * (total_opt_steps - step)
                    print(f"projection: {proj / 60:.0f}min total "
                          f"(budget {args.time_budget_minutes:.0f}min)",
                          flush=True)
                    if proj > args.time_budget_minutes * 60:
                        save_ckpt(f"epoch-{epoch + 1}-partial")
                        report["time_budget_stop"] = {
                            "at_step": step,
                            "projected_minutes": round(proj / 60, 1)}
                        write_report()
                        print("TIME-BUDGET STOP: projected run exceeds budget; "
                              "partial adapter saved", flush=True)
                        return
                if step >= total_opt_steps:
                    done = True
                    break
        if micro % args.accum != 0:
            opt.zero_grad()
            micro = 0
        if step > 0 and step % steps_per_epoch == 0:
            save_ckpt(f"epoch-{epoch + 1}")
            ev = eval_holdout_text(model, processor, ds.holdout_rows)
            ev["epoch"] = epoch + 1
            ev["opt_steps"] = step
            report["epoch_evals"].append(ev)
            score = ev["exact"] + ev["mean_sim"]
            if score > best_score:
                best_score = score
                report["best_epoch"] = epoch + 1
            write_report()
            print(f"EPOCH {epoch + 1} DONE: exact={ev['exact']} "
                  f"mean_sim={ev['mean_sim']} (best epoch {report['best_epoch']})",
                  flush=True)
        if done:
            break
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    write_report()
    print(f"TEXT DONE: {step} opt steps in {time.time() - t0:.0f}s total",
          flush=True)


if __name__ == "__main__":
    main()
