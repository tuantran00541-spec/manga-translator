"""Score Laya checkpoints (int8: quantizes one) on held-out tasks against simple baselines; writes summary.json and per-row predictions."""
import json
from pathlib import Path
import re
import sys
import time

from gate import load

# The harness's own claim regex (app/agent/session.py PASS_CLAIM), the baseline to beat.
PASS_CLAIM = re.compile(r"\b(?:pass(?:ed|es|ing)?|green|succeed(?:ed|s)?)\b|xanh|\bđạt\b", re.I)


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def ece(scores, labels, bins=10):
    total, err = len(scores), 0.0
    for b in range(bins):
        idx = [i for i, s in enumerate(scores) if b / bins <= s < (b + 1) / bins or (b == bins - 1 and s == 1.0)]
        if idx:
            err += len(idx) / total * abs(sum(scores[i] for i in idx) / len(idx) - sum(labels[i] for i in idx) / len(idx))
    return round(err, 4)


def report(scores, labels):
    """Accuracy at 0.5, AUC, ECE, and how many failed runs are caught while flagging at most 10% of passed ones."""
    fails = [s for s, y in zip(scores, labels) if not y]
    passes = sorted(s for s, y in zip(scores, labels) if y)
    cut = passes[int(0.1 * len(passes))] if passes else 0.5
    return {"accuracy": round(sum((s >= 0.5) == y for s, y in zip(scores, labels)) / len(labels), 4), "auc": auc(scores, labels),
            "ece": ece(scores, labels), "fails_caught_at_10pct_false_alarm": round(sum(s < cut for s in fails) / max(len(fails), 1), 4),
            "threshold": round(cut, 4)}


def main(data, models, out):
    rows = [json.loads(line) for line in open(data / "eval.jsonl", encoding="utf-8")]
    labels = [r["expected"]["verified"] for r in rows]
    question = rows[0]["questions"]
    summary = {"rows": len(rows), "passed": sum(labels),
               "always_pass": report([1.0] * len(rows), labels),
               "claim_regex": report([1.0 if PASS_CLAIM.search(r["meta"]["final"]) else 0.0 for r in rows], labels)}
    predictions = {r["meta"]["trial"]: {"label": y} for r, y in zip(rows, labels)}
    for name, path in models.items():
        agent = load(path)
        started = time.time()
        results = agent.predict_batch([r["state"] for r in rows], question, batch_size=4)
        took = time.time() - started
        scores = [res["answers"]["verified"]["noul"] for res in results]
        summary[name] = {**report(scores, labels), "cpu_ms_per_row": round(1000 * took / len(rows), 1)}
        for r, s in zip(rows, scores):
            predictions[r["meta"]["trial"]][name] = s
    if "tuned" in models and "int8" in models:
        # How often the int8 export answers differently from the full model, at 0.5 and at the gate's threshold.
        pairs = [(p["tuned"], p["int8"]) for p in predictions.values()]
        cut = summary["int8"]["threshold"]
        summary["int8_vs_tuned"] = {"flips_at_half": sum((a >= 0.5) != (b >= 0.5) for a, b in pairs),
                                    "flips_at_threshold": sum((a >= cut) != (b >= cut) for a, b in pairs),
                                    "max_abs_diff": round(max(abs(a - b) for a, b in pairs), 4)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "predictions.json").write_text(json.dumps(predictions, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), dict(arg.split("=", 1) for arg in sys.argv[3:]), Path(sys.argv[2]))
