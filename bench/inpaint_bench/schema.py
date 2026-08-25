from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np

SCHEMA_VERSION = "1.2.6"


def is_finite_number(val: Any) -> bool:
    if isinstance(val, bool) or val is None:
        return False
    if isinstance(val, (int, float)):
        return not (math.isnan(val) or math.isinf(val))
    return False


def is_finite_int(val: Any) -> bool:
    if isinstance(val, bool) or val is None:
        return False
    return isinstance(val, int)


@dataclass
class QualityThresholds:
    min_psnr: float = 30.0
    min_ssim: float = 0.85
    max_mae: float = 5.0
    max_psnr_drop: float = 2.0
    max_ssim_drop: float = 0.05
    max_mae_increase: float = 2.0


@dataclass
class EnvironmentMetadata:
    timestamp: str = ""
    python_version: str = ""
    platform: str = ""
    os: str = ""
    cpu_model: str = ""
    logical_cpus: int = 1
    physical_cpus: int = 1
    numpy_version: str = ""
    opencv_version: str = ""
    onnxruntime_version: str = ""
    git_commit: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvironmentMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelMetadata:
    model_name: str = "lama.onnx"
    model_path: str = ""
    model_sha256: str = ""
    model_size_bytes: int = 0
    input_resolution: list[int] = field(default_factory=lambda: [512, 512])
    data_type: str = "FP32"
    execution_provider: str = "CPUExecutionProvider"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TimingStats:
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    stddev_ms: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimingStats:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MemoryStats:
    rss_start_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_end_mb: float = 0.0
    measured: bool = False
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryStats:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MetricSummary:
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    stddev: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    invariant: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetricSummary:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def summarize_metric(values: list[float | int]) -> MetricSummary:
    if not values:
        return MetricSummary()
    arr = np.array(values, dtype=np.float64)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    mean_v = float(np.mean(arr))
    stddev_v = float(np.std(arr))
    p50_v = float(np.percentile(arr, 50))
    p95_v = float(np.percentile(arr, 95))
    invariant = (min_v == max_v)
    return MetricSummary(
        min=round(min_v, 4),
        max=round(max_v, 4),
        mean=round(mean_v, 4),
        stddev=round(stddev_v, 4),
        p50=round(p50_v, 4),
        p95=round(p95_v, 4),
        invariant=invariant,
    )


@dataclass
class TelemetryAggregate:
    model_calls: MetricSummary = field(default_factory=MetricSummary)
    tile_count: MetricSummary = field(default_factory=MetricSummary)
    active_tile_count: MetricSummary = field(default_factory=MetricSummary)
    shortcut_count: MetricSummary = field(default_factory=MetricSummary)
    cluster_count: MetricSummary = field(default_factory=MetricSummary)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TelemetryAggregate:
        kwargs = {}
        for k in ("model_calls", "tile_count", "active_tile_count", "shortcut_count", "cluster_count"):
            if k in d and isinstance(d[k], dict):
                kwargs[k] = MetricSummary.from_dict(d[k])
        return cls(**kwargs)


@dataclass
class InvocationTelemetry:
    invocation_index: int = 0
    latency_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    model_calls: int = 0
    cluster_count: int = 0
    tile_count: int = 0
    active_tile_count: int = 0
    shortcut_count: int = 0
    shortcut_types: list[str] = field(default_factory=list)
    crop_dimensions: list[list[int]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InvocationTelemetry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def summarize_telemetry(invocations: list[InvocationTelemetry]) -> TelemetryAggregate:
    if not invocations:
        return TelemetryAggregate()
    return TelemetryAggregate(
        model_calls=summarize_metric([inv.model_calls for inv in invocations]),
        tile_count=summarize_metric([inv.tile_count for inv in invocations]),
        active_tile_count=summarize_metric([inv.active_tile_count for inv in invocations]),
        shortcut_count=summarize_metric([inv.shortcut_count for inv in invocations]),
        cluster_count=summarize_metric([inv.cluster_count for inv in invocations]),
    )


@dataclass
class CaseResult:
    case_id: str = ""
    level: str = "level1_model"
    image_width: int = 0
    image_height: int = 0
    mask_type: str = ""
    mask_ratio: float = 0.0
    mask_area_pixels: int = 0
    expected_execution: str = "model_required"
    expected_shortcut_type: str | None = None
    session_create_ms: float = 0.0
    first_inference_ms: float = 0.0
    cold_total_ms: float = 0.0
    warmup_count: int = 0
    repetitions: int = 0
    thread_count: int = 1
    timing: TimingStats = field(default_factory=TimingStats)
    preprocess_timing: TimingStats = field(default_factory=TimingStats)
    inference_timing: TimingStats = field(default_factory=TimingStats)
    postprocess_timing: TimingStats = field(default_factory=TimingStats)
    model_calls_per_invocation: int | None = None
    model_calls_total: int = 0
    telemetry_summary: TelemetryAggregate = field(default_factory=TelemetryAggregate)
    cluster_count: int | None = None
    tile_count: int | None = None
    active_tile_count: int | None = None
    shortcut_count: int | None = None
    shortcut_types: list[str] = field(default_factory=list)
    crop_dimensions: list[list[int]] = field(default_factory=list)
    invocations: list[InvocationTelemetry] = field(default_factory=list)
    memory: MemoryStats = field(default_factory=MemoryStats)
    golden_output_path: str = ""
    status: str = "ok"
    error_message: str = ""
    workload_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaseResult:
        kwargs = dict(d)
        if "timing" in kwargs and isinstance(kwargs["timing"], dict):
            kwargs["timing"] = TimingStats.from_dict(kwargs["timing"])
        if "preprocess_timing" in kwargs and isinstance(kwargs["preprocess_timing"], dict):
            kwargs["preprocess_timing"] = TimingStats.from_dict(kwargs["preprocess_timing"])
        if "inference_timing" in kwargs and isinstance(kwargs["inference_timing"], dict):
            kwargs["inference_timing"] = TimingStats.from_dict(kwargs["inference_timing"])
        if "postprocess_timing" in kwargs and isinstance(kwargs["postprocess_timing"], dict):
            kwargs["postprocess_timing"] = TimingStats.from_dict(kwargs["postprocess_timing"])
        if "memory" in kwargs and isinstance(kwargs["memory"], dict):
            kwargs["memory"] = MemoryStats.from_dict(kwargs["memory"])
        if "telemetry_summary" in kwargs and isinstance(kwargs["telemetry_summary"], dict):
            kwargs["telemetry_summary"] = TelemetryAggregate.from_dict(kwargs["telemetry_summary"])
        if "invocations" in kwargs and isinstance(kwargs["invocations"], list):
            kwargs["invocations"] = [
                InvocationTelemetry.from_dict(inv) if isinstance(inv, dict) else inv
                for inv in kwargs["invocations"]
            ]
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in kwargs.items() if k in valid_fields})


def validate_case_execution(case: dict[str, Any] | CaseResult) -> tuple[bool, str]:
    c = case if isinstance(case, dict) else case.to_dict()
    status = c.get("status", "ok")
    if status != "ok":
        return False, f"Case status is {status}: {c.get('error_message', '')}"

    exec_mode = c.get("expected_execution", "model_required")
    sc_type = c.get("expected_shortcut_type")
    calls = c.get("model_calls_per_invocation")
    total_calls = c.get("model_calls_total", 0)

    if exec_mode == "model_required":
        if calls is not None and calls < 1:
            return False, f"model_required case executed with {calls} model calls (expected >= 1)"
        if calls is None and total_calls < 1:
            return False, f"model_required case executed with {total_calls} total model calls (expected >= 1)"
    elif exec_mode == "shortcut":
        if calls is not None and calls != 0:
            return False, f"shortcut case ({sc_type}) executed with {calls} model calls (expected 0)"
        if calls is None and total_calls != 0:
            return False, f"shortcut case ({sc_type}) executed with {total_calls} total model calls (expected 0)"
        shortcuts = c.get("shortcut_count")
        if shortcuts is not None and shortcuts < 1:
            return False, f"shortcut case ({sc_type}) recorded {shortcuts} shortcuts (expected >= 1)"

    timing = c.get("timing", {})
    p50 = timing.get("p50_ms")
    if p50 is not None:
        if not is_finite_number(p50) or float(p50) < 0:
            return False, f"Invalid or negative p50 timing: {p50}"

    return True, ""


def validate_case_payload_for_comparison(c: dict[str, Any]) -> None:
    if not isinstance(c, dict):
        raise ValueError("Case entry must be a dictionary")

    case_id = c.get("case_id")
    if not case_id or not isinstance(case_id, str):
        raise ValueError(f"Case missing or invalid case_id: {case_id}")

    timing = c.get("timing")
    if not isinstance(timing, dict):
        raise ValueError(f"Case {case_id}: missing timing dict")

    for field_name in ("p50_ms", "p95_ms", "mean_ms", "min_ms", "max_ms", "stddev_ms"):
        val = timing.get(field_name)
        if val is None or not is_finite_number(val):
            raise ValueError(f"Case {case_id}: invalid/missing numeric timing field {field_name}={val}")
        if float(val) < 0:
            raise ValueError(f"Case {case_id}: negative timing field {field_name}={val}")

    calls = c.get("model_calls_per_invocation")
    if calls is not None:
        if not is_finite_int(calls) or calls < 0:
            raise ValueError(f"Case {case_id}: invalid model_calls_per_invocation: {calls}")

    total_calls = c.get("model_calls_total")
    if total_calls is not None:
        if not is_finite_int(total_calls) or total_calls < 0:
            raise ValueError(f"Case {case_id}: invalid model_calls_total: {total_calls}")

    telemetry = c.get("telemetry_summary")
    if telemetry is not None:
        if not isinstance(telemetry, dict):
            raise ValueError(f"Case {case_id}: telemetry_summary must be a dictionary")
        for metric_name in ("model_calls", "tile_count", "active_tile_count", "shortcut_count", "cluster_count"):
            msum = telemetry.get(metric_name)
            if msum is not None:
                if not isinstance(msum, dict):
                    raise ValueError(f"Case {case_id}: telemetry {metric_name} must be a dict")
                for subf in ("min", "max", "mean", "stddev", "p50", "p95"):
                    sval = msum.get(subf)
                    if sval is not None:
                        if not is_finite_number(sval) or float(sval) < 0:
                            raise ValueError(f"Case {case_id}: telemetry {metric_name}.{subf} invalid: {sval}")


def validate_benchmark_payload_for_comparison(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Benchmark payload must be a JSON dictionary")

    version = payload.get("schema_version")
    if not version or not isinstance(version, str):
        raise ValueError(f"Missing or invalid schema_version in benchmark payload: {version}")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("Benchmark payload missing model dictionary")
    if not model.get("model_sha256"):
        raise ValueError("Benchmark payload model missing model_sha256")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Benchmark payload cases must be a list")

    seen_cases = set()
    for c in cases:
        validate_case_payload_for_comparison(c)
        cid = c["case_id"]
        if cid in seen_cases:
            raise ValueError(f"Duplicate case_id in benchmark payload: {cid}")
        seen_cases.add(cid)


@dataclass
class BenchmarkRunResult:
    schema_version: str = SCHEMA_VERSION
    mode: str = "all"
    thread_configurations: list[int] = field(default_factory=lambda: [1])
    repetitions: int = 30
    warmup_count: int = 3
    environment: EnvironmentMetadata = field(default_factory=EnvironmentMetadata)
    model: ModelMetadata = field(default_factory=ModelMetadata)
    baseline_commit_sha: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "thread_configurations": self.thread_configurations,
            "repetitions": self.repetitions,
            "warmup_count": self.warmup_count,
            "environment": asdict(self.environment),
            "model": asdict(self.model),
            "baseline_commit_sha": self.baseline_commit_sha,
            "summary": self.summary,
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkRunResult:
        kwargs = dict(d)
        if "environment" in kwargs and isinstance(kwargs["environment"], dict):
            kwargs["environment"] = EnvironmentMetadata.from_dict(kwargs["environment"])
        if "model" in kwargs and isinstance(kwargs["model"], dict):
            kwargs["model"] = ModelMetadata.from_dict(kwargs["model"])
        if "cases" in kwargs and isinstance(kwargs["cases"], list):
            kwargs["cases"] = [
                CaseResult.from_dict(c) if isinstance(c, dict) else c
                for c in kwargs["cases"]
            ]
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in kwargs.items() if k in valid_fields})


@dataclass
class ComparisonDelta:
    case_id: str
    workload_sha256: str = ""
    baseline_p50_ms: float | None = None
    candidate_p50_ms: float | None = None
    delta_p50_ms: float | None = None
    p50_diff_pct: float | None = None
    baseline_p95_ms: float | None = None
    candidate_p95_ms: float | None = None
    delta_p95_ms: float | None = None
    p95_diff_pct: float | None = None
    baseline_model_calls: int | None = None
    candidate_model_calls: int | None = None
    model_calls_delta: int | None = None
    model_calls_mean_delta: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    mae: float | None = None
    psnr_drop: float | None = None
    ssim_drop: float | None = None
    mae_increase: float | None = None
    psnr_delta: float | None = None
    ssim_delta: float | None = None
    mae_delta: float | None = None
    quality_regression: bool = False
    regression: bool = False
    incompatible: bool = False
    note: str = ""
