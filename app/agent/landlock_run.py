from __future__ import annotations

import argparse
import ctypes
import os
import platform
import resource
import stat
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
_libc.prctl.restype = ctypes.c_int

_DENY_LEAF = object()


def abi() -> int:
    try:
        return max(
            0,
            int(
                _libc.syscall(
                    ctypes.c_long(CREATE_RULESET),
                    ctypes.c_void_p(0),
                    ctypes.c_size_t(0),
                    ctypes.c_uint32(VERSION_FLAG),
                )
            ),
        )
    except (AttributeError, OSError):
        return 0


def write_access(version: int) -> int:
    access = WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_ALL
    if version >= 2:
        access |= REFER
    if version >= 3:
        access |= TRUNCATE
    return access


def _allowed_siblings(denied: list[str], base: str = "/") -> list[str]:
    base_real = os.path.realpath(base)
    base_parts = [p for p in base_real.split("/") if p]
    tree: dict = {}

    for path in denied:
        real = os.path.realpath(path)
        parts = [p for p in real.split("/") if p]
        if parts[: len(base_parts)] != base_parts:
            continue
        rel_parts = parts[len(base_parts) :]
        if not rel_parts:
            return []
        node = tree
        blocked = False
        for part in rel_parts[:-1]:
            existing = node.get(part)
            if existing is _DENY_LEAF:
                blocked = True
                break
            if existing is None:
                existing = {}
                node[part] = existing
            node = existing
        if not blocked:
            node[rel_parts[-1]] = _DENY_LEAF

    if not tree:
        return [base_real]

    allowed: list[str] = []

    def walk(folder: str, node: dict) -> None:
        try:
            names = os.listdir(folder)
        except OSError:
            return
        for name in names:
            full = os.path.join(folder, name)
            entry = node.get(name)
            if entry is _DENY_LEAF:
                continue
            if isinstance(entry, dict):
                walk(full, entry)
            elif not os.path.islink(full):
                allowed.append(full)

    walk(base_real, tree)
    return allowed


def _is_under(candidate: str, root: str) -> bool:
    c = os.path.realpath(candidate)
    r = os.path.realpath(root)
    return c == r or c.startswith(r.rstrip("/") + "/")


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
    ruleset = int(
        _libc.syscall(
            ctypes.c_long(CREATE_RULESET),
            ctypes.byref(buf),
            ctypes.c_size_t(len(attr)),
            ctypes.c_uint32(0),
        )
    )
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    def allow(path: str, rights: int) -> None:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError:
            return
        try:
            st = os.fstat(fd)
            if stat.S_ISLNK(st.st_mode):
                return
            is_dir = stat.S_ISDIR(st.st_mode)
            granted = rights if is_dir else rights & (WRITE_FILE | TRUNCATE | READ_FILE)
            if granted:
                rule = ctypes.create_string_buffer(struct.pack("=Qi", granted, fd), 12)
                result = int(
                    _libc.syscall(
                        ctypes.c_long(ADD_RULE),
                        ctypes.c_int(ruleset),
                        ctypes.c_int(PATH_BENEATH),
                        ctypes.byref(rule),
                        ctypes.c_uint32(0),
                    )
                )
                if result < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
        finally:
            os.close(fd)

    try:
        denied = [os.path.realpath(p) for p in (deny_read or [])]
        # Directory-only traversal/creation rights that do NOT grant reading or
        # modifying existing denied files beneath the directory.
        dir_traverse_create = (READ_DIR if reads else 0) | MAKE_ALL | (REFER if version >= 2 else 0)
        for raw_path in writable:
            real_path = os.path.realpath(raw_path)
            if denied and any(_is_under(d, real_path) for d in denied):
                # Allow listing and creating new entries at the ancestor without
                # granting READ_FILE/WRITE_FILE/REMOVE/TRUNCATE on denied secrets.
                allow(real_path, dir_traverse_create)
                for safe_child in _allowed_siblings(denied, base=real_path):
                    allow(safe_child, write | reads)
            else:
                allow(real_path, write | reads)
        if denied:
            for path in _allowed_siblings(denied):
                allow(path, reads)
        if (
            _libc.prctl(
                ctypes.c_int(NO_NEW_PRIVS),
                ctypes.c_ulong(1),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
            )
            != 0
            or int(
                _libc.syscall(
                    ctypes.c_long(RESTRICT_SELF),
                    ctypes.c_int(ruleset),
                    ctypes.c_uint32(0),
                )
            )
            < 0
        ):
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset)


class _Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class _Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_Filter))]


ARCHES = {"x86_64": (0xC000003E, 41), "AMD64": (0xC000003E, 41), "aarch64": (0xC00000B7, 198), "arm64": (0xC00000B7, 198)}
IO_URING_SETUP = 425
BLOCKED_DOMAINS = (2, 10, 17, 31, 40)
SECCOMP_RET_KILL_PROCESS, SECCOMP_RET_ALLOW, SECCOMP_RET_EPERM = 0x80000000, 0x7FFF0000, 0x00050001


def _program(arch: int, socket_nr: int) -> list[tuple[int, int, int, int]]:
    steps: list = [
        ("ld", 4), ("jeq", arch, "arch_ok", "kill"), ("label", "kill"), ("ret", SECCOMP_RET_KILL_PROCESS),
        ("label", "arch_ok"), ("ld", 0), ("jge", 0x40000000, "kill_x32", "nr_ok"),
        ("label", "nr_ok"), ("jeq", socket_nr, "is_socket", "not_socket"),
        ("label", "not_socket"), ("jeq", IO_URING_SETUP, "deny", "allow"),
        ("label", "is_socket"), ("ld", 16),
    ]
    for domain in BLOCKED_DOMAINS:
        steps.append(("jeq", domain, "deny", "next"))
        steps.append(("label", "next"))
    steps += [
        ("label", "allow"), ("ret", SECCOMP_RET_ALLOW),
        ("label", "deny"), ("ret", SECCOMP_RET_EPERM),
        ("label", "kill_x32"), ("ret", SECCOMP_RET_KILL_PROCESS),
    ]
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
    arch = ARCHES.get(platform.machine())
    if arch is None:
        return False
    rows = _program(*arch)
    array = (_Filter * len(rows))(*[_Filter(*row) for row in rows])
    program = _Program(len(rows), array)
    if (
        _libc.prctl(
            ctypes.c_int(NO_NEW_PRIVS),
            ctypes.c_ulong(1),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
        != 0
        or _libc.prctl(
            ctypes.c_int(22),
            ctypes.c_ulong(2),
            ctypes.byref(program),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
        != 0
    ):
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
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        _soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        cap = 1 << 30
        if hard != resource.RLIM_INFINITY:
            cap = min(cap, hard)
        resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))
    except (ValueError, OSError):
        pass
    try:
        restrict(args.write, not args.no_network, args.deny_read)
        if args.no_network:
            blocked = block_sockets()
            if not blocked and abi() < 4:
                raise OSError(f"cannot enforce network isolation on architecture {platform.machine()!r}")
    except OSError as exc:
        print(f"sandbox: {exc}", file=sys.stderr)
        return 126
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
