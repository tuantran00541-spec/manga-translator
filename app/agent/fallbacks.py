"""Small stand-ins for common commands a task's machine may lack; the shell reaches them only where the real one is missing."""
import argparse
import os
import sys
import time
import urllib.error
import urllib.request

NOTE = "[{name}: not installed here; this is the harness's small stand-in with the usual options]"


def _note(name: str) -> None:
    """Say once per session that a stand-in answered, so it is never mistaken for the machine's own tool."""
    mark = os.path.join(os.environ.get("AGENT_STATE") or os.environ.get("TMPDIR") or "/tmp", f".fallback-{name}")
    if not os.path.exists(mark):
        try:
            open(mark, "w").close()
        except OSError:
            pass
        print(NOTE.format(name=name), file=sys.stderr)


MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "PNG image data"), (b"\xff\xd8\xff", "JPEG image data"), (b"GIF8", "GIF image data"),
    (b"%PDF", "PDF document"), (b"PK\x03\x04", "Zip archive data"), (b"\x1f\x8b", "gzip compressed data"),
    (b"7z\xbc\xaf\x27\x1c", "7-zip archive data"), (b"BZh", "bzip2 compressed data"), (b"\xfd7zXZ\x00", "XZ compressed data"),
    (b"SQLite format 3\x00", "SQLite 3.x database"), (b"RIFF", "RIFF data"), (b"\x00asm", "WebAssembly (wasm) binary module"),
    (b"\xca\xfe\xba\xbe", "compiled Java class data"), (b"MZ", "PE32 executable (MS Windows)"), (b"BM", "PC bitmap"),
]
ELF_MACHINE = {3: "Intel 80386", 8: "MIPS", 20: "PowerPC", 40: "ARM", 62: "x86-64", 183: "ARM aarch64", 243: "RISC-V"}


def describe(path: str) -> str:
    """What a file is, from its first bytes."""
    if os.path.isdir(path):
        return "directory"
    if os.path.islink(path):
        return f"symbolic link to {os.readlink(path)}"
    with open(path, "rb") as handle:
        head = handle.read(4096)
    if not head:
        return "empty"
    if head.startswith(b"\x7fELF") and len(head) >= 20:
        bits = "64-bit" if head[4] == 2 else "32-bit"
        order = "LSB" if head[5] == 1 else "MSB"
        kind = {1: "relocatable", 2: "executable", 3: "shared object", 4: "core file"}.get(int.from_bytes(head[16:18], "little" if order == "LSB" else "big"), "file")
        machine = ELF_MACHINE.get(int.from_bytes(head[18:20], "little" if order == "LSB" else "big"), "unknown machine")
        return f"ELF {bits} {order} {kind}, {machine}"
    if head.startswith(b"\x89PNG") and len(head) >= 26:
        modes = {0: "grayscale", 2: "RGB", 3: "colormap", 4: "gray+alpha", 6: "RGBA"}
        return (f"PNG image data, {int.from_bytes(head[16:20], 'big')} x {int.from_bytes(head[20:24], 'big')}, "
                f"{head[24]}-bit/color {modes.get(head[25], '')}")
    for magic, name in MAGIC:
        if head.startswith(magic):
            return name
    if len(head) > 262 and head[257:262] == b"ustar":
        return "POSIX tar archive"
    if head.startswith(b"#!"):
        words = head[2:].split(b"\n")[0].decode(errors="replace").split()
        program = os.path.basename(words[1] if words and words[0].endswith("/env") and len(words) > 1 else words[0] if words else "")
        kind = {"sh": "POSIX shell", "bash": "Bourne-Again shell", "perl": "Perl", "node": "Node.js"}.get(program)
        return f"{kind or ('Python' if program.startswith('python') else program)} script, ASCII text executable"
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return "data"
    if any(ord(c) < 9 or 13 < ord(c) < 32 for c in text):
        return "data"
    return ("ASCII text" if text.isascii() else "Unicode text, UTF-8 text") + (", with CRLF line terminators" if "\r\n" in text else "")


def cmd_file(args: list[str]) -> int:
    """file: what each path is."""
    parser = argparse.ArgumentParser(prog="file")
    parser.add_argument("-b", "--brief", action="store_true")
    parser.add_argument("-L", "--dereference", action="store_true")
    parser.add_argument("paths", nargs="+")
    opts = parser.parse_args(args)
    code = 0
    for path in opts.paths:
        try:
            kind = describe(os.path.realpath(path) if opts.dereference else path)
        except OSError as exc:
            kind, code = f"cannot open `{path}' ({exc.strerror})", 1
        print(kind if opts.brief else f"{path}: {kind}")
    return code


def cmd_xxd(args: list[str]) -> int:
    """xxd: a hex dump, plain hex with -p, or back to bytes with -r."""
    parser = argparse.ArgumentParser(prog="xxd")
    parser.add_argument("-l", type=lambda v: int(v, 0))
    parser.add_argument("-s", type=lambda v: int(v, 0), default=0)
    parser.add_argument("-c", type=int)
    parser.add_argument("-g", type=int, default=2)
    parser.add_argument("-p", "-ps", action="store_true")
    parser.add_argument("-r", action="store_true")
    parser.add_argument("infile", nargs="?")
    parser.add_argument("outfile", nargs="?")
    opts = parser.parse_args(args)
    source = open(opts.infile, "rb") if opts.infile and opts.infile != "-" else sys.stdin.buffer
    out = open(opts.outfile, "wb") if opts.outfile else sys.stdout.buffer
    if opts.r:
        text = source.read().decode(errors="replace")
        if opts.p:
            out.write(bytes.fromhex("".join(text.split())))
        else:
            for line in text.splitlines():
                if ":" in line:
                    out.write(bytes.fromhex("".join(line.split(":", 1)[1].split("  ")[0].split())))
        return 0
    source.seek(opts.s) if opts.s and source is not sys.stdin.buffer else source.read(opts.s)
    data = source.read() if opts.l is None else source.read(opts.l)
    if opts.p:
        width = opts.c or 30
        out.write(b"".join(data[i:i + width].hex().encode() + b"\n" for i in range(0, len(data), width)))
        return 0
    width, group = opts.c or 16, max(1, opts.g)
    for i in range(0, len(data), width):
        row = data[i:i + width]
        hexes = " ".join(row[j:j + group].hex() for j in range(0, len(row), group))
        full = width * 2 + (width + group - 1) // group - 1
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        out.write(f"{opts.s + i:08x}: {hexes.ljust(full)}  {text}\n".encode())
    return 0


def cmd_strings(args: list[str]) -> int:
    """strings: runs of printable characters."""
    parser = argparse.ArgumentParser(prog="strings")
    parser.add_argument("-n", "--bytes", type=int, default=4)
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("-t", choices=["d", "o", "x"])
    parser.add_argument("paths", nargs="*")
    opts = parser.parse_args(args)
    for path in opts.paths or ["-"]:
        data = sys.stdin.buffer.read() if path == "-" else open(path, "rb").read()
        start, run = 0, bytearray()
        for i, b in enumerate(data + b"\x00"):
            if 32 <= b < 127 or b == 9:
                if not run:
                    start = i
                run.append(b)
                continue
            if len(run) >= opts.bytes:
                where = {"d": f"{start:7d} ", "o": f"{start:7o} ", "x": f"{start:7x} "}.get(opts.t or "", "")
                print(where + run.decode())
            run = bytearray()
    return 0


def _processes() -> list[dict]:
    """Every process from /proc: pid, parent, state, user id and command line."""
    rows = []
    for pid in sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int):
        try:
            stat = open(f"/proc/{pid}/stat").read()
            name = stat[stat.index("(") + 1:stat.rindex(")")]
            fields = stat[stat.rindex(")") + 2:].split()
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip() or f"[{name}]"
            uid = os.stat(f"/proc/{pid}").st_uid
        except (OSError, ValueError):
            continue
        rows.append({"pid": int(pid), "ppid": int(fields[1]), "state": fields[0], "uid": uid, "name": name, "cmd": cmd})
    return rows


def cmd_ps(args: list[str]) -> int:
    """ps: every process; any option shows full command lines."""
    rows = _processes()
    print(f"{'UID':>6} {'PID':>7} {'PPID':>7} S CMD")
    for row in rows:
        print(f"{row['uid']:>6} {row['pid']:>7} {row['ppid']:>7} {row['state']} {row['cmd'] if args else row['name']}")
    return 0


def cmd_pgrep(args: list[str]) -> int:
    """pgrep: ids of processes whose name, or command line with -f, matches."""
    parser = argparse.ArgumentParser(prog="pgrep")
    parser.add_argument("-f", action="store_true")
    parser.add_argument("-l", action="store_true")
    parser.add_argument("-a", action="store_true")
    parser.add_argument("pattern")
    opts = parser.parse_args(args)
    import re
    found = [r for r in _processes() if r["pid"] != os.getpid() and re.search(opts.pattern, r["cmd"] if opts.f else r["name"])]
    for row in found:
        print(row["pid"], *([row["cmd"]] if opts.a else [row["name"]] if opts.l else []))
    return 0 if found else 1


def cmd_free(args: list[str]) -> int:
    """free: memory from /proc/meminfo."""
    info = {}
    for line in open("/proc/meminfo"):
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0]) * 1024
    human = "-h" in args
    unit = 1 << 30 if "-g" in args else 1 << 20 if "-m" in args else 1024

    def show(n: int) -> str:
        if not human:
            return str(n // unit)
        for suffix, size in (("Gi", 1 << 30), ("Mi", 1 << 20), ("Ki", 1 << 10)):
            if n >= size:
                return f"{n / size:.1f}{suffix}"
        return f"{n}B"

    total, free_, avail = info.get("MemTotal", 0), info.get("MemFree", 0), info.get("MemAvailable", 0)
    cache = info.get("Buffers", 0) + info.get("Cached", 0)
    print(f"{'':>8}{'total':>12}{'used':>12}{'free':>12}{'shared':>12}{'buff/cache':>12}{'available':>12}")
    print(f"{'Mem:':<8}{show(total):>12}{show(total - avail):>12}{show(free_):>12}{show(info.get('Shmem', 0)):>12}{show(cache):>12}{show(avail):>12}")
    swap = info.get("SwapTotal", 0)
    print(f"{'Swap:':<8}{show(swap):>12}{show(swap - info.get('SwapFree', 0)):>12}{show(info.get('SwapFree', 0)):>12}")
    return 0


def cmd_uptime(args: list[str]) -> int:
    """uptime: time up and load average."""
    up = float(open("/proc/uptime").read().split()[0])
    load = open("/proc/loadavg").read().split()[:3]
    print(f" {time.strftime('%H:%M:%S')} up {int(up // 86400)} days, {int(up % 86400 // 3600)}:{int(up % 3600 // 60):02d},  load average: {', '.join(load)}")
    return 0


def cmd_curl(args: list[str]) -> int:
    """curl: one HTTP request with the options scripts use most."""
    parser = argparse.ArgumentParser(prog="curl", add_help=False)
    for short, long in (("-s", "--silent"), ("-S", "--show-error"), ("-L", "--location"), ("-f", "--fail"), ("-k", "--insecure"),
                        ("-i", "--include"), ("-I", "--head"), ("-v", "--verbose"), ("-O", "--remote-name")):
        parser.add_argument(short, long, action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument("-X", "--request")
    parser.add_argument("-H", "--header", action="append", default=[])
    parser.add_argument("-d", "--data", "--data-raw", "--data-binary", action="append")
    parser.add_argument("--json")
    parser.add_argument("-w", "--write-out")
    parser.add_argument("-m", "--max-time", type=float, default=300)
    parser.add_argument("--connect-timeout", type=float)
    parser.add_argument("-A", "--user-agent", default="curl/8 (harness stand-in)")
    parser.add_argument("url")
    opts, unknown = parser.parse_known_args(args)
    if unknown:
        print(f"curl stand-in: unsupported option(s) {' '.join(unknown)}; use Python's urllib or requests instead", file=sys.stderr)
        return 2
    url = opts.url if "://" in opts.url else "http://" + opts.url
    body = None
    headers = {"User-Agent": opts.user_agent}
    if opts.json is not None:
        body, headers["Content-Type"], headers["Accept"] = opts.json.encode(), "application/json", "application/json"
    elif opts.data:
        parts = [open(d[1:], "rb").read() if d.startswith("@") else d.encode() for d in opts.data]
        body, headers["Content-Type"] = b"&".join(parts), "application/x-www-form-urlencoded"
    for header in opts.header:
        key, _, value = header.partition(":")
        headers[key.strip()] = value.strip()
    method = opts.request or ("HEAD" if opts.head else "POST" if body is not None else "GET")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    context = None
    if opts.insecure:
        import ssl
        context = ssl._create_unverified_context()
    opener = urllib.request.build_opener(*([] if opts.location else [_NoRedirect()]), urllib.request.HTTPSHandler(context=context))
    try:
        reply = opener.open(request, timeout=opts.connect_timeout or opts.max_time)
    except urllib.error.HTTPError as exc:
        reply = exc
    except (urllib.error.URLError, OSError) as exc:
        if not opts.silent or opts.show_error:
            print(f"curl: ({getattr(exc, 'reason', exc)})", file=sys.stderr)
        if opts.write_out:
            sys.stdout.write(opts.write_out.replace("%{http_code}", "000").replace("\\n", "\n"))
        return 7
    status = reply.status if hasattr(reply, "status") else reply.code
    if opts.fail and status >= 400:
        if not opts.silent or opts.show_error:
            print(f"curl: (22) The requested URL returned error: {status}", file=sys.stderr)
        return 22
    head = f"HTTP/1.1 {status} {reply.reason}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in reply.headers.items()) + "\r\n"
    data = b"" if method == "HEAD" else reply.read()
    target = opts.output or (url.rstrip("/").rsplit("/", 1)[-1] if opts.remote_name else None)
    out = open(target, "wb") if target else sys.stdout.buffer
    if opts.include or opts.head:
        out.write(head.encode())
    out.write(data)
    out.flush()
    if opts.write_out:
        sys.stdout.write(opts.write_out.replace("%{http_code}", str(status)).replace("%{size_download}", str(len(data)))
                         .replace("%{url_effective}", reply.geturl()).replace("\\n", "\n"))
    return 0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Stops at a redirect, as curl does without -L."""

    def redirect_request(self, *args, **kwargs):
        return None


COMMANDS = {"file": cmd_file, "xxd": cmd_xxd, "strings": cmd_strings, "ps": cmd_ps, "pgrep": cmd_pgrep, "free": cmd_free,
            "uptime": cmd_uptime, "curl": cmd_curl}


def main(argv: list[str]) -> int:
    """Run one stand-in by name."""
    name, args = argv[0], argv[1:]
    _note(name)
    try:
        return COMMANDS[name](args) or 0
    except SystemExit as exc:
        return int(exc.code or 0)
    except (OSError, ValueError) as exc:
        print(f"{name}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) if len(sys.argv) > 1 else print("usage: agent_fallbacks NAME [ARGS]", file=sys.stderr) or 2)
