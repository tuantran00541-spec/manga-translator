"""Average the weights of Laya checkpoints tuned from the same base on different shards into one checkpoint (a model soup)."""
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def main(out, *ckpts):
    """Keeps the first checkpoint's config and tokenizer; deletes each input's weights once added, to save disk."""
    shutil.copytree(ckpts[0], out, ignore=shutil.ignore_patterns("model.safetensors", "checkpoint_latest"))
    total, dtypes = {}, {}
    for path in ckpts:
        for k, v in load_file(str(Path(path) / "model.safetensors")).items():
            if k not in total:
                total[k], dtypes[k] = v.double() if v.is_floating_point() else v, v.dtype
            elif v.is_floating_point():
                total[k] += v.double()
        (Path(path) / "model.safetensors").unlink()
    save_file({k: (v / len(ckpts) if v.is_floating_point() else v).to(dtypes[k]).contiguous() for k, v in total.items()},
              str(Path(out) / "model.safetensors"))
    print(f"averaged {len(ckpts)} checkpoints into {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
