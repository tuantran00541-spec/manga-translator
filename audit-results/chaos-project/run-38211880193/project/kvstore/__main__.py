"""Command line interface: python -m kvstore --file data.json <command> ...

Commands:
    set KEY VALUE [--ttl N]  store VALUE (parsed as JSON when possible)
    get KEY                  print the value as JSON (exit 1 if missing)
    del KEY                  delete KEY (prints True/False)
    keys                     print one live key per line

A missing file is treated as an empty store.  set/del save the store
back to the file afterwards.
"""

import argparse
import json
import sys

from .store import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kvstore",
        description="A tiny JSON-backed key/value store.",
    )
    parser.add_argument("--file", default="kvstore.json", help="store file path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_set = sub.add_parser("set", help="store VALUE under KEY")
    p_set.add_argument("key")
    p_set.add_argument("value", help="stored as JSON when parseable, else as string")
    p_set.add_argument("--ttl", type=float, default=None, metavar="N",
                       help="time to live in seconds")

    p_get = sub.add_parser("get", help="print the value of KEY as JSON")
    p_get.add_argument("key")

    p_del = sub.add_parser("del", help="delete KEY")
    p_del.add_argument("key")

    sub.add_parser("keys", help="print live keys, one per line")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = Store.load(args.file)
    except FileNotFoundError:
        store = Store()  # a missing file means an empty store

    if args.command == "set":
        try:
            value = json.loads(args.value)
        except ValueError:
            value = args.value
        store.set(args.key, value, ttl=args.ttl)
        store.save(args.file)
        return 0

    if args.command == "get":
        value = store.get(args.key)
        if value is None:
            print(f"kvstore: key not found: {args.key}", file=sys.stderr)
            return 1
        print(json.dumps(value))
        return 0

    if args.command == "del":
        deleted = store.delete(args.key)
        store.save(args.file)
        print("True" if deleted else "False")
        return 0

    # args.command == "keys"
    for key in store.keys():
        print(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
