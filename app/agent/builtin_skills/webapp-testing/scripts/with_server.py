#!/usr/bin/env python3

import argparse
import os
import signal
import socket
import subprocess
import sys
import time


def is_server_ready(port: int, timeout: float = 30, process: subprocess.Popen | None = None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except OSError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run command with one or more servers")
    parser.add_argument("--server", action="append", dest="servers", required=True, help="Server command (can be repeated)")
    parser.add_argument("--port", action="append", dest="ports", type=int, required=True, help="Port for each server (must match --server count)")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds per server (default: 30)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after server(s) ready")

    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    if not args.command:
        print("Error: No command specified to run", file=sys.stderr)
        sys.exit(1)

    if len(args.servers) != len(args.ports):
        print("Error: Number of --server and --port arguments must match", file=sys.stderr)
        sys.exit(1)

    if args.timeout <= 0:
        print("Error: --timeout must be positive", file=sys.stderr)
        sys.exit(1)

    for port in args.ports:
        if not 1 <= port <= 65535:
            print(f"Error: Invalid port {port}; must be 1-65535", file=sys.stderr)
            sys.exit(1)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(143))

    servers = [{"cmd": cmd, "port": port} for cmd, port in zip(args.servers, args.ports)]
    server_processes: list[subprocess.Popen] = []

    try:
        for i, server in enumerate(servers):
            print(f"Starting server {i + 1}/{len(servers)}: {server['cmd']}")
            popen_kwargs: dict = {
                "shell": True,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            process = subprocess.Popen(server["cmd"], **popen_kwargs)
            server_processes.append(process)

            print(f"Waiting for server on port {server['port']}...")
            if not is_server_ready(server["port"], timeout=args.timeout, process=process):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited early with code {process.returncode} before port {server['port']} became ready"
                    )
                raise RuntimeError(
                    f"Server failed to start on port {server['port']} within {args.timeout}s"
                )

            print(f"Server ready on port {server['port']}")

        print(f"\nAll {len(servers)} server(s) ready")
        print(f"Running: {' '.join(args.command)}\n")
        result = subprocess.run(args.command)
        sys.exit(result.returncode)
    finally:
        print(f"\nStopping {len(server_processes)} server(s)...")
        for i, process in enumerate(server_processes):
            _stop_process(process)
            print(f"Server {i + 1} stopped")
        print("All servers stopped")


if __name__ == "__main__":
    main()
