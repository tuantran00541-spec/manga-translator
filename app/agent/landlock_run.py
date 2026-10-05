"""Run one command under a Landlock policy: writes only beneath given folders, no network sockets, some folders unreadable."""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import resource
import struct
import sys

CREATE_RULESET, ADD_RULE, RESTRICT_SELF = 444, 445, 446
VERSION_FLAG = 1
PATH_BENEATH = 1
NO_NEW_PRIVS = 38
READ_FILE, READ_DIR = 1 << 2, 1 << 3
WRITE_FILE, REMOVE_DIR, REMOVE_FILE = 1 << 1, 1 << 4, 1 << 5
SCOPE_UNIX_SOCKET, SCOPE_SIGNAL = 1 << 0, 1 << 1
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


def _allowed_siblings(denied: list[str]) -> list[str]:
    """Every path to leave readable so that exactly the denied ones are not: the siblings along the way to each of them."""
    tree: dict = {}
    for path in denied:
        node = tree
        for part in [p for p in os.path.realpath(path).split("/") if p]:
            node = node.setdefault(part, {})
    allowed: list[str] = []

    def walk(folder: str, node: dict) -> None:
        try:
            names = os.listdir(folder)
        except OSError:
            return
        for name in names:
            full = os.path.join(folder, name)
            if name in node:
                if node[name]:
                    walk(full, node[name])
            elif not os.path.islink(full):
                allowed.append(full)

    walk("/", tree)
    return allowed


def restrict(writable: list[str], network: bool, deny_read: list[str] | None = None) -> None:
    version = abi()
    if version < 1:
        raise OSError("Landlock is not available in this kernel")
    write = write_access(version)
    reads = READ_FILE | READ_DIR if deny_read else 0
    net = NET_BIND | NET_CONNECT if (not network and version >= 4) else 0
    scoped = SCOPE_UNIX_SOCKET | SCOPE_SIGNAL if version >= 6 else 0
    if version >= 6:
        attr = struct.pack("QQQ", write | reads, net, scoped)
    elif version >= 4:
        attr = struct.pack("QQ", write | reads, net)
    else:
        attr = struct.pack("Q", write | reads)
    buf = ctypes.create_string_buffer(attr, len(attr))
    ruleset = _libc.syscall(CREATE_RULESET, buf, ctypes.c_size_t(len(attr)), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    def allow(path: str, rights: int) -> None:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError:
            return
        is_dir = os.path.isdir(path)
        # Directories take every right; a lone file such as /dev/null takes the file rights only.
        granted = rights if is_dir else rights & (WRITE_FILE | TRUNCATE | READ_FILE)
        if granted:
            rule = ctypes.create_string_buffer(struct.pack("=Qi", granted, fd), 12)
            result = _libc.syscall(ADD_RULE, ruleset, PATH_BENEATH, rule, 0)
            if result < 0:
                os.close(fd)
                raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
        os.close(fd)

    for path in writable:
        allow(path, write | reads)
    for path in _allowed_siblings(deny_read or []) if deny_read else []:
        allow(path, reads)
    if _libc.prctl(NO_NEW_PRIVS, 1, 0, 0, 0) != 0 or _libc.syscall(RESTRICT_SELF, ruleset, 0) < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    os.close(ruleset)


class _Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class _Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_Filter))]


ARCHES = {"x86_64": (0xC000003E, 41), "AMD64": (0xC000003E, 41), "aarch64": (0xC00000B7, 198), "arm64": (0xC00000B7, 198)}
IO_URING_SETUP = 425
# Internet, packet, Bluetooth and VM-host sockets; local Unix sockets stay possible.
BLOCKED_DOMAINS = (2, 10, 17, 31, 40)
SECCOMP_RET_KILL_PROCESS, SECCOMP_RET_ALLOW, SECCOMP_RET_EPERM = 0x80000000, 0x7FFF0000, 0x00050001


def _program(arch: int, socket_nr: int) -> list[tuple[int, int, int, int]]:
    """A seccomp filter: socket() for the blocked domains and io_uring_setup() fail with EPERM; a foreign ABI or x32 call is killed."""
    steps: list = [("ld", 4), ("jeq", arch, "arch_ok", "kill"), ("label", "kill"), ("ret", SECCOMP_RET_KILL_PROCESS), ("label", "arch_ok"),
                   ("ld", 0), ("jge", 0x40000000, "kill_x32", "nr_ok"), ("label", "nr_ok"), ("jeq", socket_nr, "is_socket", "not_socket"), ("label", "not_socket"), ("jeq", IO_URING_SETUP, "deny", "allow"),
                   ("label", "is_socket"), ("ld", 16)]
    for domain in BLOCKED_DOMAINS:
        steps.append(("jeq", domain, "deny", "next"))
        steps.append(("label", "next"))
    steps += [("label", "allow"), ("ret", SECCOMP_RET_ALLOW), ("label", "deny"), ("ret", SECCOMP_RET_EPERM),
              ("label", "kill_x32"), ("ret", SECCOMP_RET_KILL_PROCESS)]
    # Resolve labels: each label is the index of the instruction that follows it; "next" always means the following instruction.
    out, labels, pending = [], {}, []
    index = 0
    for step in steps:
        if step[0] == "label":
            labels.setdefault(step[1], []).append(index)
        else:
            pending.append((index, step))
            index += 1
    for index, step in pending:
        if step[0] == "ld":
            out.append((0x20, 0, 0, step[1]))
        elif step[0] == "ret":
            out.append((0x06, 0, 0, step[1]))
        else:
            def target(name: str) -> int:
                options = [i for i in labels[name] if i > index]
                return min(options) - index - 1
            out.append((0x35 if step[0] == "jge" else 0x15, target(step[2]), target(step[3]), step[1]))
    return out


def block_sockets() -> bool:
    """Network sockets and io_uring are refused for this process and its children; false where the CPU is not supported."""
    arch = ARCHES.get(platform.machine())
    if arch is None:
        return False
    rows = _program(*arch)
    array = (_Filter * len(rows))(*[_Filter(*row) for row in rows])
    program = _Program(len(rows), array)
    if _libc.prctl(NO_NEW_PRIVS, 1, 0, 0, 0) != 0 or _libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "seccomp filter failed")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="append", default=[])
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--deny-read", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print(abi())
        return 0
    try:
        # No core dumps, and no single file larger than 1 GiB: a runaway command cannot fill the disk.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 30, 1 << 30))
    except (ValueError, OSError):
        pass
    try:
        restrict(args.write, not args.no_network, args.deny_read)
        if args.no_network:
            block_sockets()
    except OSError as exc:
        print(f"sandbox: {exc}", file=sys.stderr)
        return 126
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
