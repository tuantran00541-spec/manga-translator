from __future__ import annotations
import os
import sys
import time
import math
import hashlib
import platform
import subprocess
from pathlib import Path
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import psutil
except ImportError:
    psutil = None

from .schema import EnvironmentMetadata, ModelMetadata, TimingStats, MemoryStats


def calculate_stats(times_ms: list[float]) -> TimingStats:
    if not times_ms:
        return TimingStats()

    arr = np.array(times_ms, dtype=np.float64)
    count = len(arr)
    mean_val = float(np.mean(arr))
    p50_val = float(np.percentile(arr, 50))
    p95_val = float(np.percentile(arr, 95))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    stddev_val = float(np.std(arr))

    return TimingStats(
        count=count,
        mean_ms=round(mean_val, 4),
        p50_ms=round(p50_val, 4),
        p95_ms=round(p95_val, 4),
        min_ms=round(min_val, 4),
        max_ms=round(max_val, 4),
        stddev_ms=round(stddev_val, 4),
    )


def get_model_sha256(model_path: Path | str) -> str:
    path = Path(model_path)
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_environment_metadata() -> EnvironmentMetadata:
    logical_cpus = os.cpu_count() or 1
    physical_cpus = logical_cpus
    if psutil:
        try:
            physical_cpus = psutil.cpu_count(logical=False) or logical_cpus
        except Exception:
            pass

    git_commit = ""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            git_commit = res.stdout.strip()
    except Exception:
        pass

    return EnvironmentMetadata(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        os=platform.system(),
        cpu_model=platform.processor() or "Unknown",
        logical_cpus=logical_cpus,
        physical_cpus=physical_cpus,
        numpy_version=np.__version__,
        opencv_version=cv2.__version__,
        onnxruntime_version=ort.__version__ if ort else "N/A",
        git_commit=git_commit,
    )


def get_model_metadata(model_path: Path | str) -> ModelMetadata:
    path = Path(model_path)
    sha = get_model_sha256(path)
    size = path.stat().st_size if path.is_file() else 0
    return ModelMetadata(
        model_name=path.name,
        model_path=str(path.resolve()) if path.is_file() else str(path),
        model_sha256=sha,
        model_size_bytes=size,
        input_resolution=[512, 512],
        data_type="FP32",
        execution_provider="CPUExecutionProvider",
    )


class MemoryTracker:
    def __init__(self):
        self.rss_start_mb = 0.0
        self.rss_peak_mb = 0.0
        self.rss_end_mb = 0.0
        self.measured = False
        self.note = ""

    def _get_rss_mb(self) -> float | None:
        if psutil:
            try:
                proc = psutil.Process()
                return proc.memory_info().rss / (1024 * 1024)
            except Exception:
                pass

        if platform.system() == "Windows":
            try:
                import ctypes
                from ctypes import wintypes

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    return counters.WorkingSetSize / (1024 * 1024)
            except Exception:
                pass

        if platform.system() == "Linux":
            try:
                with open("/proc/self/statm", "r") as f:
                    pages = int(f.read().split()[1])
                    return (pages * os.sysconf("SC_PAGE_SIZE")) / (1024 * 1024)
            except Exception:
                pass

        return None

    def start(self):
        rss = self._get_rss_mb()
        if rss is not None:
            self.rss_start_mb = rss
            self.rss_peak_mb = rss
            self.measured = True
        else:
            self.measured = False
            self.note = "Memory metrics unavailable on this environment"

    def sample(self):
        if not self.measured:
            return
        rss = self._get_rss_mb()
        if rss is not None and rss > self.rss_peak_mb:
            self.rss_peak_mb = rss

    def finish(self) -> MemoryStats:
        if self.measured:
            rss = self._get_rss_mb()
            self.rss_end_mb = rss if rss is not None else self.rss_peak_mb
            return MemoryStats(
                rss_start_mb=round(self.rss_start_mb, 2),
                rss_peak_mb=round(self.rss_peak_mb, 2),
                rss_end_mb=round(self.rss_end_mb, 2),
                measured=True,
                note="",
            )
        return MemoryStats(
            rss_start_mb=0.0,
            rss_peak_mb=0.0,
            rss_end_mb=0.0,
            measured=False,
            note=self.note or "Unavailable",
        )
