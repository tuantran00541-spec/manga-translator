from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np

SCHEMA_VERSION = "1.2.1"


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


@dataclass
class TimingStats:
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    stddev_ms: float = 0.0


@dataclass
class MemoryStats:
    rss_start_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_end_mb: float = 0.0
    measured: bool = True
    note: str = ""


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


@dataclass
class MetricSummary:
    min: int = 0
    max: int = 0
    mean: float = 0.0
    invariant: bool = True


@dataclass
class TelemetryAggregate:
    model_calls: MetricSummary = field(default_factory=MetricSummary)
    cluster_count: MetricSummary = field(default_factory=MetricSummary)
    tile_count: MetricSummary = field(default_factory=MetricSummary)
    active_tile_count: MetricSummary = field(default_factory=MetricSummary)
    shortcut_count: MetricSummary = field(default_factory=MetricSummary)


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
    exp_exec = case.expected_execution
    exp_sc_type = case.expected_shortcut_type

    observed_sc_types = set()
    for inv in case.invocations:
        for st in inv.shortcut_types:
            observed_sc_types.add(st)
    for st in case.shortcut_types:
        observed_sc_types.add(st)

    if exp_exec == "model_required":
        if case.telemetry_summary.model_calls.max == 0:
            return False, "Expected model execution but observed 0 model calls (unwanted shortcut activation)"
        if case.telemetry_summary.shortcut_count.max > 0:
            return False, f"Expected model_required with 0 shortcuts but observed {case.telemetry_summary.shortcut_count.max} shortcuts ({sorted(list(observed_sc_types))})"

    elif exp_exec in ("shortcut", "shortcut_expected") or exp_exec.startswith("shortcut_"):
        if case.telemetry_summary.model_calls.max > 0:
            return False, f"Expected shortcut execution with 0 model calls but observed {case.telemetry_summary.model_calls.max} model calls"
        if case.telemetry_summary.shortcut_count.max == 0:
            return False, "Expected shortcut execution but shortcut was not activated"

        expected_type = exp_sc_type
        if not expected_type and exp_exec.startswith("shortcut_"):
            expected_type = exp_exec[len("shortcut_"):]

        if expected_type and expected_type not in observed_sc_types:
            return False, f"Expected shortcut type '{expected_type}' but observed {sorted(list(observed_sc_types))}"

    return True, ""


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
    model_calls_per_invocation: int | None = 0
    model_calls_total: int = 0
    telemetry_summary: TelemetryAggregate = field(default_factory=TelemetryAggregate)
    cluster_count: int = 0
    tile_count: int = 0
    active_tile_count: int = 0
    shortcut_count: int = 0
    shortcut_types: list[str] = field(default_factory=list)
    crop_dimensions: list[list[int]] = field(default_factory=list)
    invocations: list[InvocationTelemetry] = field(default_factory=list)
    memory: MemoryStats = field(default_factory=MemoryStats)
    golden_output_path: str = ""
    status: str = "ok"
    error_message: str = ""

    @property
    def model_calls(self) -> int:
        if self.model_calls_per_invocation is not None:
            return self.model_calls_per_invocation
        return int(round(self.telemetry_summary.model_calls.mean))

    @property
    def cold_start_ms(self) -> float:
        return self.first_inference_ms


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
    baseline_model_calls: int
    candidate_model_calls: int
    model_calls_delta: int
    psnr: float = 0.0
    ssim: float = 0.0
    mae: float = 0.0
    regression: bool = False
