"""Runs one shell command on Windows at low (or untrusted) integrity, so it cannot write to anything labelled above that."""
from __future__ import annotations

import ctypes
from ctypes import wintypes as wt
import os
import sys

LEVELS = {"low": "S-1-16-4096", "untrusted": "S-1-16-0"}
TOKEN_RIGHTS = 0x0001 | 0x0002 | 0x0008 | 0x0080  # assign primary, duplicate, query, adjust default
CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT = 0x4, 0x400
KILL_ON_JOB_CLOSE, EXTENDED_LIMITS = 0x2000, 9
STARTF_USESTDHANDLES, HANDLE_FLAG_INHERIT = 0x100, 0x1


class SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wt.DWORD)]


class StartupInfo(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR), ("lpTitle", wt.LPWSTR),
                ("dwX", wt.DWORD), ("dwY", wt.DWORD), ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD),
                ("dwXCountChars", wt.DWORD), ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD), ("dwFlags", wt.DWORD),
                ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD), ("lpReserved2", ctypes.c_void_p),
                ("hStdInput", wt.HANDLE), ("hStdOutput", wt.HANDLE), ("hStdError", wt.HANDLE)]


class ProcessInformation(ctypes.Structure):
    _fields_ = [("hProcess", wt.HANDLE), ("hThread", wt.HANDLE), ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD)]


class BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wt.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("Basic", BasicLimits), ("Io", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _check(ok, what: str):
    if not ok:
        raise OSError(f"{what} failed: {ctypes.FormatError(ctypes.get_last_error())}")
    return ok


def label(path: str, level: str) -> None:
    """Give a folder (and what it holds) an integrity label, so commands at that level may write there."""
    advapi, kernel = ctypes.WinDLL("advapi32", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
    descriptor, sacl = ctypes.c_void_p(), ctypes.c_void_p()
    present, defaulted = wt.BOOL(), wt.BOOL()
    sddl = f"S:(ML;OICI;NW;;;{LEVELS[level]})"
    _check(advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None), "ConvertStringSecurityDescriptor")
    try:
        _check(advapi.GetSecurityDescriptorSacl(descriptor, ctypes.byref(present), ctypes.byref(sacl), ctypes.byref(defaulted)), "GetSecurityDescriptorSacl")
        # LABEL_SECURITY_INFORMATION on a file object; children that inherit take the label too.
        error = advapi.SetNamedSecurityInfoW(ctypes.c_wchar_p(path), 1, 0x10, None, None, None, sacl)
        if error:
            raise OSError(f"could not label {path}: {ctypes.FormatError(error)}")
    finally:
        kernel.LocalFree(descriptor)


def launch(level: str, command: str) -> int:
    """Start the command through cmd.exe with a lowered token inside a kill-on-close job, and return its exit code."""
    advapi, kernel = ctypes.WinDLL("advapi32", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wt.HANDLE
    kernel.CreateJobObjectW.restype = wt.HANDLE
    kernel.GetStdHandle.restype = wt.HANDLE
    token, low, sid = wt.HANDLE(), wt.HANDLE(), ctypes.c_void_p()
    _check(advapi.OpenProcessToken(wt.HANDLE(kernel.GetCurrentProcess()), TOKEN_RIGHTS, ctypes.byref(token)), "OpenProcessToken")
    _check(advapi.DuplicateTokenEx(token, TOKEN_RIGHTS, None, 2, 1, ctypes.byref(low)), "DuplicateTokenEx")
    _check(advapi.ConvertStringSidToSidW(LEVELS[level], ctypes.byref(sid)), "ConvertStringSidToSidW")
    mandatory = SidAndAttributes(sid, 0x20)
    size = ctypes.sizeof(mandatory) + advapi.GetLengthSid(sid)
    _check(advapi.SetTokenInformation(low, 25, ctypes.byref(mandatory), size), "SetTokenInformation")
    job = _check(kernel.CreateJobObjectW(None, None), "CreateJobObjectW")
    limits = ExtendedLimits()
    limits.Basic.LimitFlags = KILL_ON_JOB_CLOSE
    _check(kernel.SetInformationJobObject(wt.HANDLE(job), EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)), "SetInformationJobObject")
    startup = StartupInfo(cb=ctypes.sizeof(StartupInfo), dwFlags=STARTF_USESTDHANDLES)
    handles = [kernel.GetStdHandle(n) for n in (-10, -11, -12)]
    for handle in handles:
        kernel.SetHandleInformation(wt.HANDLE(handle), HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)
    startup.hStdInput, startup.hStdOutput, startup.hStdError = handles
    shell = os.environ.get("ComSpec") or "cmd.exe"
    line = ctypes.create_unicode_buffer(f'"{shell}" /d /s /c "{command}"')
    info = ProcessInformation()
    _check(advapi.CreateProcessAsUserW(low, None, line, None, None, True, CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
                                       None, None, ctypes.byref(startup), ctypes.byref(info)), "CreateProcessAsUserW")
    _check(kernel.AssignProcessToJobObject(wt.HANDLE(job), wt.HANDLE(info.hProcess)), "AssignProcessToJobObject")
    kernel.ResumeThread(wt.HANDLE(info.hThread))
    kernel.WaitForSingleObject(wt.HANDLE(info.hProcess), wt.DWORD(0xFFFFFFFF))
    code = wt.DWORD()
    kernel.GetExitCodeProcess(wt.HANDLE(info.hProcess), ctypes.byref(code))
    return code.value


if __name__ == "__main__":
    # Usage: win_low.py LEVEL -- COMMAND
    level, command = sys.argv[1], " ".join(sys.argv[sys.argv.index("--") + 1:])
    try:
        sys.exit(launch(level, command))
    except OSError as exc:
        print(f"[sandbox] {exc}", file=sys.stderr)
        sys.exit(126)
