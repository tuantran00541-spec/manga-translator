from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
import re
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

NOTE = "[{name}: not installed here; this is the harness's small stand-in with the usual options]"


def _note(name: str) -> None:
    base = os.environ.get("AGENT_STATE") or os.environ.get("TMPDIR") or tempfile.gettempdir()
    mark = os.path.join(base, f".fallback-{name}")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(mark, flags, 0o600)
        os.close(fd)
    except FileExistsError:
        return
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
    st = os.lstat(path)
    mode = st.st_mode
    if stat.S_ISLNK(mode):
        return f"symbolic link to {os.readlink(path)}"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISFIFO(mode):
        return "fifo (named pipe)"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character special"
    if stat.S_ISBLK(mode):
        return "block special"
    if not stat.S_ISREG(mode):
        return "special file"
    with open(path, "rb") as handle:
        head = handle.read(4096)
    if not head:
        return "empty"
    if head.startswith(b"\x7fELF") and len(head) >= 20:
        bits = "64-bit" if head[4] == 2 else "32-bit"
        order = "LSB" if head[5] == 1 else "MSB"
        kind = {1: "relocatable", 2: "executable", 3: "shared object", 4: "core file"}.get(
            int.from_bytes(head[16:18], "little" if order == "LSB" else "big"), "file"
        )
        machine = ELF_MACHINE.get(
            int.from_bytes(head[18:20], "little" if order == "LSB" else "big"), "unknown machine"
        )
        return f"ELF {bits} {order} {kind}, {machine}"
    if head.startswith(b"\x89PNG") and len(head) >= 26:
        modes = {0: "grayscale", 2: "RGB", 3: "colormap", 4: "gray+alpha", 6: "RGBA"}
        return (
            f"PNG image data, {int.from_bytes(head[16:20], 'big')} x {int.from_bytes(head[20:24], 'big')}, "
            f"{head[24]}-bit/color {modes.get(head[25], '')}"
        )
    for magic, name in MAGIC:
        if head.startswith(magic):
            return name
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "POSIX tar archive"
    if head.startswith(b"#!"):
        words = head[2:].split(b"\n")[0].decode(errors="replace").split()
        program = os.path.basename(
            words[1] if words and words[0].endswith("/env") and len(words) > 1 else words[0] if words else ""
        )
        kind = {"sh": "POSIX shell", "bash": "Bourne-Again shell", "perl": "Perl", "node": "Node.js"}.get(program)
        return f"{kind or ('Python' if program.startswith('python') else program)} script, ASCII text executable"
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return "data"
    if any(ord(c) < 9 or 13 < ord(c) < 32 for c in text):
        return "data"
    return (
        ("ASCII text" if text.isascii() else "Unicode text, UTF-8 text")
        + (", with CRLF line terminators" if "\r\n" in text else "")
    )


def cmd_file(args: list[str]) -> int:
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
    if opts.s < 0:
        raise ValueError("negative -s offset is not supported")
    if opts.l is not None and opts.l < 0:
        raise ValueError("-l length must be non-negative")
    if opts.c is not None and opts.c <= 0:
        raise ValueError("-c columns must be positive")
    if opts.infile and opts.infile != "-" and opts.outfile:
        try:
            if os.path.exists(opts.outfile) and os.path.samefile(opts.infile, opts.outfile):
                raise ValueError("infile and outfile cannot be the same file")
        except OSError:
            pass

    with ExitStack() as stack:
        source = (
            stack.enter_context(open(opts.infile, "rb"))
            if opts.infile and opts.infile != "-"
            else sys.stdin.buffer
        )
        out = stack.enter_context(open(opts.outfile, "wb")) if opts.outfile else sys.stdout.buffer
        if opts.r:
            text = source.read().decode(errors="replace")
            if opts.p:
                out.write(bytes.fromhex("".join(text.split())))
            else:
                for line in text.splitlines():
                    if ":" in line:
                        out.write(bytes.fromhex("".join(line.split(":", 1)[1].split("  ")[0].split())))
            return 0

        if opts.s:
            if source is not sys.stdin.buffer:
                source.seek(opts.s)
            else:
                remaining_skip = opts.s
                while remaining_skip > 0:
                    skipped = source.read(min(65536, remaining_skip))
                    if not skipped:
                        break
                    remaining_skip -= len(skipped)

        remaining = opts.l
        width = opts.c or (30 if opts.p else 16)
        group = max(1, opts.g)
        offset = opts.s
        full = width * 2 + (width + group - 1) // group - 1
        chunk_rows = max(1, (65536 // width) * width)

        while remaining is None or remaining > 0:
            to_read = chunk_rows if remaining is None else min(chunk_rows, remaining)
            data = source.read(to_read)
            if not data:
                break
            if remaining is not None:
                remaining -= len(data)
            if opts.p:
                out.write(
                    b"".join(data[i : i + width].hex().encode() + b"\n" for i in range(0, len(data), width))
                )
            else:
                for i in range(0, len(data), width):
                    row = data[i : i + width]
                    hexes = " ".join(row[j : j + group].hex() for j in range(0, len(row), group))
                    text = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
                    out.write(f"{offset + i:08x}: {hexes.ljust(full)}  {text}\n".encode())
            offset += len(data)
        return 0


def cmd_strings(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="strings")
    parser.add_argument("-n", "--bytes", type=int, default=4)
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("-t", choices=["d", "o", "x"])
    parser.add_argument("paths", nargs="*")
    opts = parser.parse_args(args)
    if opts.bytes < 1:
        raise ValueError("-n/--bytes must be >= 1")

    def flush_run(start: int, run: bytearray) -> None:
        if len(run) >= opts.bytes:
            where = {"d": f"{start:7d} ", "o": f"{start:7o} ", "x": f"{start:7x} "}.get(opts.t or "", "")
            print(where + run.decode("ascii"))

    for path in opts.paths or ["-"]:
        with ExitStack() as stack:
            stream = sys.stdin.buffer if path == "-" else stack.enter_context(open(path, "rb"))
            offset = 0
            start = 0
            run = bytearray()
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                for idx, b in enumerate(chunk):
                    if 32 <= b < 127 or b == 9:
                        if not run:
                            start = offset + idx
                        run.append(b)
                    else:
                        if run:
                            flush_run(start, run)
                            run.clear()
                offset += len(chunk)
            if run:
                flush_run(start, run)
    return 0


def _processes() -> list[dict]:
    rows = []
    for pid in sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int):
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
                stat_text = f.read()
            name = stat_text[stat_text.index("(") + 1 : stat_text.rindex(")")]
            fields = stat_text[stat_text.rindex(")") + 2 :].split()
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip() or f"[{name}]"
            uid = os.stat(f"/proc/{pid}").st_uid
            rows.append(
                {
                    "pid": int(pid),
                    "ppid": int(fields[1]),
                    "state": fields[0],
                    "uid": uid,
                    "name": name,
                    "cmd": cmd,
                }
            )
        except (OSError, ValueError, IndexError):
            continue
    return rows


def cmd_ps(args: list[str]) -> int:
    rows = _processes()
    print(f"{'UID':>6} {'PID':>7} {'PPID':>7} S CMD")
    for row in rows:
        print(f"{row['uid']:>6} {row['pid']:>7} {row['ppid']:>7} {row['state']} {row['cmd'] if args else row['name']}")
    return 0


def cmd_pgrep(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="pgrep")
    parser.add_argument("-f", action="store_true")
    parser.add_argument("-l", action="store_true")
    parser.add_argument("-a", action="store_true")
    parser.add_argument("pattern")
    opts = parser.parse_args(args)
    try:
        pattern = re.compile(opts.pattern)
    except re.error as exc:
        print(f"pgrep: invalid pattern: {exc}", file=sys.stderr)
        return 2
    found = [
        r for r in _processes() if r["pid"] != os.getpid() and pattern.search(r["cmd"] if opts.f else r["name"])
    ]
    for row in found:
        print(row["pid"], *([row["cmd"]] if opts.a else [row["name"]] if opts.l else []))
    return 0 if found else 1


def cmd_free(args: list[str]) -> int:
    info: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8", errors="replace") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parts = value.split()
            if not parts:
                continue
            try:
                info[key] = int(parts[0]) * 1024
            except ValueError:
                continue
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
    with open("/proc/uptime", encoding="utf-8") as f:
        up = float(f.read().split()[0])
    with open("/proc/loadavg", encoding="utf-8") as f:
        load = f.read().split()[:3]
    print(
        f" {time.strftime('%H:%M:%S')} up {int(up // 86400)} days, "
        f"{int(up % 86400 // 3600)}:{int(up % 3600 // 60):02d},  load average: {', '.join(load)}"
    )
    return 0


def _http_only_opener(follow_redirects: bool, ssl_context) -> urllib.request.OpenerDirector:
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler(context=ssl_context))
    opener.add_handler(_SafeHttpRedirect() if follow_redirects else _NoRedirect())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    return opener


def _safe_remote_filename(url: str) -> str:
    path = urllib.parse.urlparse(url).path.rstrip("/")
    name = os.path.basename(path)
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("cannot derive safe local filename from URL for -O")
    return name


def cmd_curl(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="curl", add_help=False)
    for short, long in (
        ("-s", "--silent"), ("-S", "--show-error"), ("-L", "--location"), ("-f", "--fail"), ("-k", "--insecure"),
        ("-i", "--include"), ("-I", "--head"), ("-v", "--verbose"), ("-O", "--remote-name"),
    ):
        parser.add_argument(short, long, action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("-o", "--output")
    parser.add_argument("-X", "--request")
    parser.add_argument("-H", "--header", action="append", default=[])
    parser.add_argument("-d", "--data", "--data-binary", dest="data", action="append")
    parser.add_argument("--data-raw", dest="data_raw", action="append")
    parser.add_argument("--json")
    parser.add_argument("-w", "--write-out")
    parser.add_argument("-m", "--max-time", type=float, default=300)
    parser.add_argument("--connect-timeout", type=float)
    parser.add_argument("-A", "--user-agent", default="curl/8 (harness stand-in)")
    parser.add_argument("url")
    opts, unknown = parser.parse_known_args(args)
    if unknown:
        print(
            f"curl stand-in: unsupported option(s) {' '.join(unknown)}; use Python's urllib or requests instead",
            file=sys.stderr,
        )
        return 2
    url = opts.url if "://" in opts.url else "http://" + opts.url
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        print(f"curl: unsupported protocol '{parsed.scheme}'", file=sys.stderr)
        return 1

    body = None
    headers = {"User-Agent": opts.user_agent}
    if opts.json is not None:
        body, headers["Content-Type"], headers["Accept"] = opts.json.encode(), "application/json", "application/json"
    elif opts.data or opts.data_raw:
        parts: list[bytes] = []
        for d in opts.data or []:
            if d.startswith("@"):
                with open(d[1:], "rb") as f:
                    parts.append(f.read())
            else:
                parts.append(d.encode())
        for d in opts.data_raw or []:
            parts.append(d.encode())
        body, headers["Content-Type"] = b"&".join(parts), "application/x-www-form-urlencoded"
    for header in opts.header:
        key, _, value = header.partition(":")
        if key.strip():
            headers[key.strip()] = value.strip()
    method = opts.request or ("HEAD" if opts.head else "POST" if body is not None else "GET")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    context = None
    if opts.insecure:
        import ssl

        context = ssl._create_unverified_context()

    opener = _http_only_opener(opts.location, context)
    deadline = time.monotonic() + max(0.1, float(opts.max_time))
    sock_timeout = opts.connect_timeout if opts.connect_timeout and opts.connect_timeout > 0 else opts.max_time
    reply = None
    try:
        try:
            reply = opener.open(request, timeout=min(sock_timeout, max(0.1, opts.max_time)))
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

        target = opts.output or (_safe_remote_filename(url) if opts.remote_name else None)
        head = (
            f"HTTP/1.1 {status} {reply.reason}\r\n"
            + "".join(f"{k}: {v}\r\n" for k, v in reply.headers.items())
            + "\r\n"
        )
        size_download = 0
        with ExitStack() as stack:
            out = stack.enter_context(open(target, "wb")) if target else sys.stdout.buffer
            if opts.include or opts.head:
                out.write(head.encode())
            if method != "HEAD":
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Operation timed out")
                    chunk = reply.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
                    size_download += len(chunk)
            out.flush()
        if opts.write_out:
            sys.stdout.write(
                opts.write_out.replace("%{http_code}", str(status))
                .replace("%{size_download}", str(size_download))
                .replace("%{url_effective}", reply.geturl())
                .replace("\\n", "\n")
            )
        return 0
    finally:
        if reply is not None:
            try:
                reply.close()
            except Exception:
                pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class _SafeHttpRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        scheme = urllib.parse.urlparse(newurl).scheme.lower()
        if scheme not in ("http", "https"):
            raise urllib.error.URLError(f"redirect to unsupported protocol '{scheme}'")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


COMMANDS = {
    "file": cmd_file, "xxd": cmd_xxd, "strings": cmd_strings, "ps": cmd_ps,
    "pgrep": cmd_pgrep, "free": cmd_free, "uptime": cmd_uptime, "curl": cmd_curl,
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        name = argv[0] if argv else "<none>"
        print(f"agent_fallbacks: unknown command {name!r}", file=sys.stderr)
        return 2
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
    sys.exit(
        main(sys.argv[1:])
        if len(sys.argv) > 1
        else print("usage: agent_fallbacks NAME [ARGS]", file=sys.stderr) or 2
    )
