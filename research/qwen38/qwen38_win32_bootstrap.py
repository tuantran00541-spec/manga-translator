#!/usr/bin/env python3
"""Windows-only bootstrap for the exact Qwen3.8 runtime import graph.

Legacy evidence modules intentionally keep their proven Linux code unchanged.
On Windows, Python lacks the ``resource`` module, although these modules use it
only for peak-RSS diagnostics.  This bootstrap installs a minimal compatibility
module before importing the generator and binds UCRT ``expf`` into the existing
exact arithmetic wrapper.
"""
from __future__ import annotations

import ctypes
import importlib
import sys
import types


def _peak_working_set_kib() -> int:
    if sys.platform != "win32":
        raise RuntimeError("Win32 peak working set requested on non-Windows host")

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    # Linux resource.ru_maxrss is KiB; preserve that unit so the existing
    # evidence helpers continue to divide by 1024**2 to report GiB.
    return int(counters.PeakWorkingSetSize) // 1024


def install_resource_compat():
    """Return native ``resource`` or install the minimal Win32 diagnostic shim."""
    try:
        return importlib.import_module("resource")
    except ImportError:
        if sys.platform != "win32":
            raise

    existing = sys.modules.get("resource")
    if existing is not None:
        return existing

    module = types.ModuleType("resource")
    module.RUSAGE_SELF = 0

    def getrusage(who: int):
        if int(who) != int(module.RUSAGE_SELF):
            raise ValueError("Win32 resource compatibility supports RUSAGE_SELF only")
        return types.SimpleNamespace(ru_maxrss=_peak_working_set_kib())

    module.getrusage = getrusage
    sys.modules["resource"] = module
    return module


def _bind_ucrt_expf(exact_module):
    """Bind the existing exact wrapper to Windows UCRT ``expf``."""
    if sys.platform != "win32":
        return getattr(exact_module, "_LIBM", None)
    ucrt = ctypes.CDLL("ucrtbase.dll")
    ucrt.expf.argtypes = [ctypes.c_float]
    ucrt.expf.restype = ctypes.c_float
    exact_module._LIBM = ucrt
    return ucrt


def import_generator():
    """Import the proven generator graph without editing its legacy modules."""
    install_resource_compat()
    generator = importlib.import_module("qwen35_k3_generate")
    _bind_ucrt_expf(generator.exact)
    return generator


def sanity() -> None:
    if sys.platform != "win32":
        raise SystemExit("Win32 bootstrap sanity must run on Windows")
    resource = install_resource_compat()
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if rss <= 0:
        raise SystemExit(f"invalid Win32 peak RSS: {rss}")
    generator = import_generator()
    if int(generator.N_LAYER) != 64:
        raise SystemExit(f"unexpected generator layer count: {generator.N_LAYER}")
    if generator.exact._LIBM is None:
        raise SystemExit("Windows UCRT expf was not bound")
    if float(generator.exact.expf(0.0)) != 1.0:
        raise SystemExit("Windows UCRT expf sanity failed")
    print("QWEN38_WIN32_GENERATOR_BOOTSTRAP_PASS")


if __name__ == "__main__":
    sanity()
