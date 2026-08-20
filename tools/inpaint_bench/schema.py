from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np

SCHEMA_VERSION = "1.2.2"


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
    intra_op_threads: int = 1
    inter_op_threads: int = 1
    execution_mode: str = "ORT_SEQUENTIAL"

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
    measured: bool = True
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryStats:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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


@dataclass
class MetricSummary:
    min: int = 0
    max: int = 0
    mean: float = 0.0
    invariant: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetricSummary:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TelemetryAggregate:
    model_calls: MetricSummary = field(default_factory=MetricSummary)
    cluster_count: MetricSummary = field(default_factory=MetricSummary)
    tile_count: MetricSummary = field(default_factory=MetricSummary)
    active_tile_count: MetricSummary = field(default_factory=MetricSummary)
    shortcut_count: MetricSummary = field(default_factory=MetricSummary)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TelemetryAggregate:
        kwargs = {}
        for f in ["model_calls", "cluster_count", "tile_count", "active_tile_count", "shortcut_count"]:
            if f in d and isinstance(d[f], dict):
                kwargs[f] = MetricSummary.from_dict(d[f])
            elif f in d and isinstance(d[f], MetricSummary):
                kwargs[f] = d[f]
        return cls(**kwargs)


def summarize_metric(values: list[int]) -> MetricSummary:
    if not values:
        return MetricSummary()
    min_v = int(min(values))
    max_v = int(max(values))
    mean_v = float(np.mean(values))
    return MetricSummary(
        min=min_v,
        max=max_v,
        mean=round(mean_v, 2),
        invariant=(min_v == max_v),
    )


def summarize_telemetry(invocations: list[InvocationTelemetry]) -> TelemetryAggregate:
    if not invocations:
        return TelemetryAggregate()
    return TelemetryAggregate(
        model_calls=summarize_metric([inv.model_calls for inv in invocations]),
        cluster_count=summarize_metric([inv.cluster_count for inv in invocations]),
        tile_count=summarize_metric([inv.tile_count for inv in invocations]),
        active_tile_count=summarize_metric([inv.active_tile_count for inv in invocations]),
        shortcut_count=summarize_metric([inv.shortcut_count for inv in invocations]),
    )


def validate_case_execution(case: CaseResult) -> tuple[bool, str]:
    if not case.invocations:
        return False, "No invocations recorded for case"

    exp_exec = case.expected_execution
    exp_sc_type = case.expected_shortcut_type

    if case.level == "level2_pipeline":
        if exp_exec != "model_required":
            return False, f"Level 2 pipeline only supports 'model_required', but got '{exp_exec}'"
        for i, inv in enumerate(case.invocations):
            if inv.model_calls < 1:
                return False, f"Invocation {i}: Expected Level 2 model_calls >= 1 but got {inv.model_calls}"
            if inv.shortcut_count > 0 or len(inv.shortcut_types) > 0:
                return False, f"Invocation {i}: Level 2 pipeline does not support shortcuts but got {inv.shortcut_count} ({inv.shortcut_types})"
        return True, ""

    if exp_exec == "model_required":
        for i, inv in enumerate(case.invocations):
            if inv.model_calls < 1:
                return False, f"Invocation {i}: Expected model_calls >= 1 but got {inv.model_calls} (unwanted shortcut activation)"
            if inv.shortcut_count > 0 or len(inv.shortcut_types) > 0:
                return False, f"Invocation {i}: Expected 0 shortcuts but got {inv.shortcut_count} ({inv.shortcut_types})"
        return True, ""

    if exp_exec in ("shortcut", "shortcut_expected") or (exp_exec and exp_exec.startswith("shortcut_")):
        allowed_types = {"white", "black", "low_std"}
        target_type = exp_sc_type
        if not target_type and exp_exec.startswith("shortcut_"):
            target_type = exp_exec[len("shortcut_"):]

        if target_type not in allowed_types:
            return False, f"Unsupported expected shortcut type: '{target_type}'. Allowed: {sorted(list(allowed_types))}"

        for i, inv in enumerate(case.invocations):
            if inv.model_calls != 0:
                return False, f"Invocation {i}: Expected model_calls == 0 for shortcut case but got {inv.model_calls}"
            if inv.shortcut_count != 1:
                return False, f"Invocation {i}: Expected shortcut_count == 1 but got {inv.shortcut_count}"
            if inv.shortcut_types != [target_type]:
                return False, f"Invocation {i}: Expected shortcut_types == ['{target_type}'] but got {inv.shortcut_types}"
        return True, ""

    if exp_exec == "mixed":
        for i, inv in enumerate(case.invocations):
            if inv.model_calls < 1:
                return False, f"Invocation {i}: Mixed case expected model_calls >= 1 but got {inv.model_calls}"
            if inv.shortcut_count < 1:
                return False, f"Invocation {i}: Mixed case expected shortcut_count >= 1 but got {inv.shortcut_count}"
        return True, ""

    return False, f"Unknown expected_execution archetype: '{exp_exec}'"


@dataclass
class CaseResult:
    case_id: str = ""
    level: str = ""
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
    warmup_count: int = 3
    repetitions: int = 30
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

    @property
    def model_calls(self) -> int | None:
        return self.model_calls_per_invocation

    @property
    def cold_start_ms(self) -> float:
        return self.first_inference_ms

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaseResult:
        kwargs = dict(d)
        for timing_f in ["timing", "preprocess_timing", "inference_timing", "postprocess_timing"]:
            if timing_f in kwargs and isinstance(kwargs[timing_f], dict):
                kwargs[timing_f] = TimingStats.from_dict(kwargs[timing_f])

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


@dataclass
class BenchmarkRunResult:
    schema_version: str = SCHEMA_VERSION
    mode: str = "all"
    threads: int = 1
    warmup_count: int = 3
    repetitions: int = 30
    environment: EnvironmentMetadata = field(default_factory=EnvironmentMetadata)
    model: ModelMetadata = field(default_factory=ModelMetadata)
    cases: list[CaseResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
    baseline_p50_ms: float
    candidate_p50_ms: float
    delta_p50_ms: float
    p50_diff_pct: float
    baseline_p95_ms: float
    candidate_p95_ms: float
    delta_p95_ms: float
    p95_diff_pct: float
    baseline_model_calls: int | None
    candidate_model_calls: int | None
    model_calls_delta: int | None
    model_calls_mean_delta: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    mae: float = 0.0
    regression: bool = False
    incompatible: bool = False
    note: str = ""
