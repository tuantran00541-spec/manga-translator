from __future__ import annotations

import ctypes
from ctypes import wintypes as wt
import os
import sys

LOW = "S-1-16-4096"
TOKEN_RIGHTS = 0x0001 | 0x0002 | 0x0008 | 0x0080
CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT = 0x4, 0x400
KILL_ON_JOB_CLOSE, EXTENDED_LIMITS = 0x2000, 9
STARTF_USESTDHANDLES, HANDLE_FLAG_INHERIT = 0x100, 0x1
_INVALID_HANDLES = {None, 0, -1, (1 << 32) - 1, (1 << 64) - 1}


class SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wt.DWORD)]


class StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR), ("lpTitle", wt.LPWSTR),
        ("dwX", wt.DWORD), ("dwY", wt.DWORD), ("dwXSize", wt.DWORD), ("dwYSize", wt.DWORD),
        ("dwXCountChars", wt.DWORD), ("dwYCountChars", wt.DWORD), ("dwFillAttribute", wt.DWORD), ("dwFlags", wt.DWORD),
        ("wShowWindow", wt.WORD), ("cbReserved2", wt.WORD), ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wt.HANDLE), ("hStdOutput", wt.HANDLE), ("hStdError", wt.HANDLE),
    ]


class ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wt.HANDLE), ("hThread", wt.HANDLE),
        ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD),
    ]


class BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD),
    ]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("Basic", BasicLimits), ("Io", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _win32():
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wt.LPCWSTR, wt.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wt.ULONG),
    ]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wt.BOOL
    advapi.GetSecurityDescriptorSacl.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(wt.BOOL), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wt.BOOL),
    ]
    advapi.GetSecurityDescriptorSacl.restype = wt.BOOL
    advapi.SetNamedSecurityInfoW.argtypes = [
        wt.LPWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    advapi.SetNamedSecurityInfoW.restype = wt.DWORD
    advapi.OpenProcessToken.argtypes = [wt.HANDLE, wt.DWORD, ctypes.POINTER(wt.HANDLE)]
    advapi.OpenProcessToken.restype = wt.BOOL
    advapi.DuplicateTokenEx.argtypes = [
        wt.HANDLE, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, ctypes.POINTER(wt.HANDLE),
    ]
    advapi.DuplicateTokenEx.restype = wt.BOOL
    advapi.ConvertStringSidToSidW.argtypes = [wt.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    advapi.ConvertStringSidToSidW.restype = wt.BOOL
    advapi.GetLengthSid.argtypes = [ctypes.c_void_p]
    advapi.GetLengthSid.restype = wt.DWORD
    advapi.SetTokenInformation.argtypes = [wt.HANDLE, wt.DWORD, ctypes.c_void_p, wt.DWORD]
    advapi.SetTokenInformation.restype = wt.BOOL
    advapi.CreateProcessAsUserW.argtypes = [
        wt.HANDLE, wt.LPCWSTR, wt.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, wt.BOOL,
        wt.DWORD, ctypes.c_void_p, wt.LPCWSTR, ctypes.POINTER(StartupInfo), ctypes.POINTER(ProcessInformation),
    ]
    advapi.CreateProcessAsUserW.restype = wt.BOOL

    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wt.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    kernel.CreateJobObjectW.restype = wt.HANDLE
    kernel.SetInformationJobObject.argtypes = [wt.HANDLE, wt.DWORD, ctypes.c_void_p, wt.DWORD]
    kernel.SetInformationJobObject.restype = wt.BOOL
    kernel.GetStdHandle.argtypes = [wt.DWORD]
    kernel.GetStdHandle.restype = wt.HANDLE
    kernel.SetHandleInformation.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD]
    kernel.SetHandleInformation.restype = wt.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    kernel.AssignProcessToJobObject.restype = wt.BOOL
    kernel.ResumeThread.argtypes = [wt.HANDLE]
    kernel.ResumeThread.restype = wt.DWORD
    kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    kernel.WaitForSingleObject.restype = wt.DWORD
    kernel.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
    kernel.GetExitCodeProcess.restype = wt.BOOL
    kernel.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
    kernel.TerminateProcess.restype = wt.BOOL
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.CloseHandle.restype = wt.BOOL
    return advapi, kernel


def _check(ok, what: str):
    if not ok:
        raise OSError(f"{what} failed: {ctypes.FormatError(ctypes.get_last_error())}")
    return ok


def label(path: str) -> None:
    if os.path.islink(path):
        raise OSError(f"refusing to label symlink: {path}")
    advapi, kernel = _win32()
    descriptor, sacl = ctypes.c_void_p(), ctypes.c_void_p()
    present, defaulted = wt.BOOL(), wt.BOOL()
    sddl = f"S:(ML;OICI;NW;;;{LOW})"
    _check(
        advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, ctypes.byref(descriptor), None
        ),
        "ConvertStringSecurityDescriptor",
    )
    try:
        _check(
            advapi.GetSecurityDescriptorSacl(
                descriptor, ctypes.byref(present), ctypes.byref(sacl), ctypes.byref(defaulted)
            ),
            "GetSecurityDescriptorSacl",
        )
        error = advapi.SetNamedSecurityInfoW(path, 1, 0x10, None, None, None, sacl)
        if error:
            raise OSError(f"could not label {path}: {ctypes.FormatError(error)}")
    finally:
        kernel.LocalFree(descriptor)


def launch(command: str) -> int:
    advapi, kernel = _win32()
    token, low, sid = wt.HANDLE(), wt.HANDLE(), ctypes.c_void_p()
    job = wt.HANDLE()
    info = ProcessInformation()
    resumed = False
    try:
        _check(
            advapi.OpenProcessToken(kernel.GetCurrentProcess(), TOKEN_RIGHTS, ctypes.byref(token)),
            "OpenProcessToken",
        )
        _check(
            advapi.DuplicateTokenEx(token, TOKEN_RIGHTS, None, 2, 1, ctypes.byref(low)),
            "DuplicateTokenEx",
        )
        _check(advapi.ConvertStringSidToSidW(LOW, ctypes.byref(sid)), "ConvertStringSidToSidW")
        size = ctypes.sizeof(SidAndAttributes) + advapi.GetLengthSid(sid)
        mandatory_buf = ctypes.create_string_buffer(size)
        mandatory = SidAndAttributes.from_buffer(mandatory_buf)
        mandatory.Sid = sid.value
        mandatory.Attributes = 0x20
        _check(advapi.SetTokenInformation(low, 25, mandatory_buf, size), "SetTokenInformation")

        job_raw = _check(kernel.CreateJobObjectW(None, None), "CreateJobObjectW")
        job = wt.HANDLE(job_raw)
        limits = ExtendedLimits()
        limits.Basic.LimitFlags = KILL_ON_JOB_CLOSE
        _check(
            kernel.SetInformationJobObject(job, EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)),
            "SetInformationJobObject",
        )

        startup = StartupInfo(cb=ctypes.sizeof(StartupInfo), dwFlags=STARTF_USESTDHANDLES)
        handles = [kernel.GetStdHandle(wt.DWORD(n & 0xFFFFFFFF)) for n in (-10, -11, -12)]
        for handle in handles:
            if handle not in _INVALID_HANDLES:
                kernel.SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)
        startup.hStdInput, startup.hStdOutput, startup.hStdError = handles

        shell = os.environ.get("ComSpec") or "cmd.exe"
        line = ctypes.create_unicode_buffer(f'"{shell}" /d /s /c "{command}"')
        _check(
            advapi.CreateProcessAsUserW(
                low, None, line, None, None, True,
                CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
                None, None, ctypes.byref(startup), ctypes.byref(info),
            ),
            "CreateProcessAsUserW",
        )
        _check(kernel.AssignProcessToJobObject(job, info.hProcess), "AssignProcessToJobObject")
        if kernel.ResumeThread(info.hThread) == 0xFFFFFFFF:
            raise OSError(f"ResumeThread failed: {ctypes.FormatError(ctypes.get_last_error())}")
        resumed = True
        if kernel.WaitForSingleObject(info.hProcess, wt.DWORD(0xFFFFFFFF)) == 0xFFFFFFFF:
            raise OSError(f"WaitForSingleObject failed: {ctypes.FormatError(ctypes.get_last_error())}")
        code = wt.DWORD()
        _check(kernel.GetExitCodeProcess(info.hProcess, ctypes.byref(code)), "GetExitCodeProcess")
        return code.value - (1 << 32) if code.value >= 1 << 31 else code.value
    finally:
        if info.hProcess and not resumed:
            kernel.TerminateProcess(info.hProcess, 1)
        for h in (info.hThread, info.hProcess, low, token, job):
            val = h.value if isinstance(h, wt.HANDLE) else h
            if val not in _INVALID_HANDLES:
                kernel.CloseHandle(h)
        if sid.value:
            kernel.LocalFree(sid)


if __name__ == "__main__":
    if "--" not in sys.argv:
        print("usage: win_low.py -- COMMAND...", file=sys.stderr)
        sys.exit(2)
    command = " ".join(sys.argv[sys.argv.index("--") + 1 :])
    try:
        sys.exit(launch(command))
    except OSError as exc:
        print(f"[sandbox] {exc}", file=sys.stderr)
        sys.exit(126)
