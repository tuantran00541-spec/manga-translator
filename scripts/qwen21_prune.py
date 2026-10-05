"""One-off: keep only some transformer blocks of a Qwen-Image-2.1 GGUF, renumbered, to see how far it can be cut."""
from __future__ import annotations

import argparse
import re

from gguf import GGUFReader, GGUFWriter, GGUFValueType

BLOCK = re.compile(r"^(transformer_blocks\.)(\d+)(\..*)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--keep", required=True, help="comma-separated block indices to keep, in order")
    args = parser.parse_args()
    keep = [int(i) for i in args.keep.split(",")]
    new_index = {old: new for new, old in enumerate(keep)}
    reader = GGUFReader(args.src)
    arch = str(bytes(reader.fields["general.architecture"].parts[-1]), "utf-8")
    writer = GGUFWriter(args.dst, arch=arch)
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue
        kind = field.types[0]
        if kind == GGUFValueType.STRING:
            writer.add_string(field.name, str(bytes(field.parts[-1]), "utf-8"))
        elif kind != GGUFValueType.ARRAY:
            writer.add_key_value(field.name, field.parts[-1][0], kind)
    kept = []
    for tensor in reader.tensors:
        match = BLOCK.match(tensor.name)
        name = tensor.name
        if match:
            old = int(match.group(2))
            if old not in new_index:
                continue
            name = f"{match.group(1)}{new_index[old]}{match.group(3)}"
        kept.append((name, tensor))
    for name, tensor in kept:
        writer.add_tensor_info(name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, tensor in kept:
        writer.write_tensor_data(tensor.data)
    writer.close()
    print(f"kept {len(keep)} blocks, {len(kept)} tensors -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
