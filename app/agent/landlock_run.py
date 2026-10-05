"""Run one command under a Landlock policy: writes only beneath given folders, optionally no TCP."""
from __future__ import annotations

import argparse
import ctypes
import os
import struct
import sys

CREATE_RULESET, ADD_RULE, RESTRICT_SELF = 444, 445, 446
VERSION_FLAG = 1
PATH_BENEATH = 1
NO_NEW_PRIVS = 38
WRITE_FILE, REMOVE_DIR, REMOVE_FILE = 1 << 1, 1 << 4, 1 << 5
MAKE_ALL = sum(1 << bit for bit in range(6, 13))
REFER, TRUNCATE = 1 << 13, 1 << 14
NET_BIND, NET_CONNECT = 1 << 0, 1 << 1

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def abi() -> int:
    """The kernel's Landlock ABI version, or 0 when Landlock is off."""
    try:
        return max(0, int(_libc.syscall(CREATE_RULESET, None, 0, VERSION_FLAG)))
    except (AttributeError, OSError):
        return 0


def write_access(version: int) -> int:
    access = WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_ALL
    if version >= 2:
        access |= REFER
    if version >= 3:
        access |= TRUNCATE
    return access


def restrict(writable: list[str], network: bool) -> None:
    version = abi()
    if version < 1:
        raise OSError("Landlock is not available in this kernel")
    access = write_access(version)
    net = NET_BIND | NET_CONNECT if (not network and version >= 4) else 0
    attr = struct.pack("QQ", access, net) if version >= 4 else struct.pack("Q", access)
    buf = ctypes.create_string_buffer(attr, len(attr))
    ruleset = _libc.syscall(CREATE_RULESET, buf, ctypes.c_size_t(len(attr)), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    for path in writable:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            continue
        # Directories take every write right; a lone file such as /dev/null takes the file rights only.
        allowed = access if os.path.isdir(path) else access & (WRITE_FILE | TRUNCATE)
        rule = ctypes.create_string_buffer(struct.pack("=Qi", allowed, fd), 12)
        result = _libc.syscall(ADD_RULE, ruleset, PATH_BENEATH, rule, 0)
        os.close(fd)
        if result < 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
    if _libc.prctl(NO_NEW_PRIVS, 1, 0, 0, 0) != 0 or _libc.syscall(RESTRICT_SELF, ruleset, 0) < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    os.close(ruleset)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="append", default=[])
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print(abi())
        return 0
    try:
        restrict(args.write, not args.no_network)
    except OSError as exc:
        print(f"sandbox: {exc}", file=sys.stderr)
        return 126
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
