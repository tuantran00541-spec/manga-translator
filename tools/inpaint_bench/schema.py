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

    if exp_exec not in ("model_required", "shortcut"):
        return False, f"Unsupported expected_execution archetype: '{exp_exec}'. Allowed: ['model_required', 'shortcut']"

    if exp_exec == "model_required":
        if exp_sc_type is not None:
            return False, f"expected_shortcut_type must be None for model_required, but got '{exp_sc_type}'"
        for i, inv in enumerate(case.invocations):
            if inv.model_calls < 1:
                return False, f"Invocation {i}: Expected model_calls >= 1 but got {inv.model_calls}"
            if inv.shortcut_count > 0 or len(inv.shortcut_types) > 0:
                return False, f"Invocation {i}: Expected 0 shortcuts but got {inv.shortcut_count} ({inv.shortcut_types})"
            if inv.tile_count > 0:
                if inv.active_tile_count > inv.tile_count:
                    return False, f"Invocation {i}: active_tile_count ({inv.active_tile_count}) > tile_count ({inv.tile_count})"
            if inv.tile_count > 1 and inv.active_tile_count >= 1:
                if inv.model_calls != inv.active_tile_count:
                    return False, f"Invocation {i}: Tiled execution model_calls ({inv.model_calls}) != active_tile_count ({inv.active_tile_count})"
            if inv.cluster_count > 1:
                if len(inv.crop_dimensions) != inv.cluster_count or inv.model_calls != inv.cluster_count:
                    return False, f"Invocation {i}: Multi-cluster execution mismatch: cluster_count={inv.cluster_count}, crops={len(inv.crop_dimensions)}, calls={inv.model_calls}"
                for c_dim in inv.crop_dimensions:
                    if case.image_width > 0 and (c_dim[0] > case.image_width or c_dim[1] > case.image_height):
                        return False, f"Invocation {i}: Crop dimensions {c_dim} exceed image dimensions ({case.image_width}x{case.image_height})"
        return True, ""

    if exp_exec == "shortcut":
        allowed_types = {"white", "black", "low_std"}
        if not exp_sc_type or exp_sc_type not in allowed_types:
            return False, f"Unsupported or missing expected shortcut type: '{exp_sc_type}'. Allowed: {sorted(list(allowed_types))}"

        for i, inv in enumerate(case.invocations):
            if inv.model_calls != 0:
                return False, f"Invocation {i}: Expected model_calls == 0 for shortcut case but got {inv.model_calls}"
            if inv.shortcut_count != 1:
                return False, f"Invocation {i}: Expected shortcut_count == 1 but got {inv.shortcut_count}"
            if inv.shortcut_types != [exp_sc_type]:
                return False, f"Invocation {i}: Expected shortcut_types == ['{exp_sc_type}'] but got {inv.shortcut_types}"
        return True, ""

    return False, f"Unknown expected_execution archetype: '{exp_exec}'"


def validate_case_payload_for_comparison(case: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(case, dict):
        return False, "Case payload is not a dictionary"

    case_id = case.get("case_id")
    if not case_id or not isinstance(case_id, str) or not case_id.strip():
        return False, f"Invalid or missing case_id: {case_id!r}"

    status = case.get("status")
    if status != "ok":
        return False, f"Case status is not 'ok': {status!r}"

    level = case.get("level")
    if not level or not isinstance(level, str) or level not in ("level1_model", "level2_pipeline", "level3_e2e"):
        return False, f"Missing or invalid 'level': {level!r}"

    timing = case.get("timing")
    if not isinstance(timing, dict):
        return False, "Missing or invalid 'timing' dictionary"

    for field_name in ["count", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "stddev_ms"]:
        val = timing.get(field_name)
        if not is_finite_number(val):
            return False, f"Missing, non-numeric, or non-finite timing field: 'timing.{field_name}' ({val!r})"
        if field_name == "count" and not is_finite_int(val):
            return False, f"'timing.count' must be an integer: {val!r}"
        if float(val) < 0.0:
            return False, f"Negative timing value in 'timing.{field_name}': {val!r}"

    exp_exec = case.get("expected_execution")
    if not exp_exec or not isinstance(exp_exec, str) or exp_exec not in ("model_required", "shortcut"):
        return False, f"Missing or invalid 'expected_execution': {exp_exec!r}"

    if exp_exec == "shortcut":
        sc_type = case.get("expected_shortcut_type")
        if not sc_type or sc_type not in ("white", "black", "low_std"):
            return False, f"Missing or invalid expected_shortcut_type: {sc_type!r}"

    if level in ("level2_pipeline", "level3_e2e"):
        invs = case.get("invocations")
        if not isinstance(invs, list) or len(invs) == 0:
            return False, f"Level {level} requires a non-empty 'invocations' list"

        if timing.get("count") != len(invs):
            return False, f"Timing count ({timing.get('count')}) != len(invocations) ({len(invs)})"

        for idx, inv in enumerate(invs):
            if not isinstance(inv, dict):
                return False, f"Invocation {idx} is not a dictionary"

            inv_idx = inv.get("invocation_index")
            if not is_finite_int(inv_idx) or inv_idx != idx:
                return False, f"Invocation at index {idx} has non-contiguous or invalid 'invocation_index': {inv_idx!r} (expected {idx})"

            for f_time in ["latency_ms", "preprocess_ms", "inference_ms", "postprocess_ms"]:
                val = inv.get(f_time)
                if not is_finite_number(val) or float(val) < 0.0:
                    return False, f"Invocation {idx} missing or invalid timing '{f_time}': {val!r}"

            mc = inv.get("model_calls")
            if not is_finite_int(mc) or mc < 0:
                return False, f"Invocation {idx} missing or negative integer 'model_calls': {mc!r}"

            if level == "level3_e2e":
                for f_count in ["cluster_count", "tile_count", "active_tile_count", "shortcut_count"]:
                    val = inv.get(f_count)
                    if not is_finite_int(val) or val < 0:
                        return False, f"Invocation {idx} missing or negative integer '{f_count}': {val!r}"

                if not isinstance(inv.get("shortcut_types"), list):
                    return False, f"Invocation {idx} missing 'shortcut_types' list"
                for st in inv.get("shortcut_types", []):
                    if st not in ("white", "black", "low_std"):
                        return False, f"Invocation {idx} invalid shortcut type '{st}'"

                crops = inv.get("crop_dimensions")
                if not isinstance(crops, list):
                    return False, f"Invocation {idx} missing 'crop_dimensions' list"
                for crop in crops:
                    if not isinstance(crop, list) or len(crop) != 2 or not all(is_finite_int(d) and d > 0 for d in crop):
                        return False, f"Invocation {idx} invalid crop dimension: {crop!r}"

        # Check total model calls
        total_mc = case.get("model_calls_total")
        actual_mc_total = sum(inv.get("model_calls", 0) for inv in invs)
        if is_finite_int(total_mc) and total_mc != actual_mc_total:
            return False, f"model_calls_total ({total_mc}) != sum of invocation model_calls ({actual_mc_total})"

        telem_sum = case.get("telemetry_summary")
        if not isinstance(telem_sum, dict):
            return False, f"Level {level} requires 'telemetry_summary' dictionary"

        # Recompute and verify telemetry aggregate consistency
        for metric_name in ["model_calls", "cluster_count", "tile_count", "active_tile_count", "shortcut_count"]:
            if metric_name not in telem_sum or not isinstance(telem_sum[metric_name], dict):
                return False, f"Missing 'telemetry_summary.{metric_name}' dictionary"
            m_dict = telem_sum[metric_name]
            for mf in ["min", "max", "mean", "invariant"]:
                if mf not in m_dict:
                    return False, f"Missing field '{mf}' in 'telemetry_summary.{metric_name}'"
                if mf in ("min", "max") and not is_finite_int(m_dict[mf]):
                    return False, f"Non-integer '{mf}' in 'telemetry_summary.{metric_name}': {m_dict[mf]!r}"
                if mf == "mean" and (not is_finite_number(m_dict[mf]) or float(m_dict[mf]) < 0.0):
                    return False, f"Non-finite or negative mean in 'telemetry_summary.{metric_name}': {m_dict[mf]!r}"
                if mf == "invariant" and not isinstance(m_dict[mf], bool):
                    return False, f"Non-boolean invariant in 'telemetry_summary.{metric_name}': {m_dict[mf]!r}"

            # Validate against actual invocation values
            actual_vals = [inv.get(metric_name, 0) for inv in invs]
            exp_min = int(min(actual_vals))
            exp_max = int(max(actual_vals))
            exp_mean = round(float(np.mean(actual_vals)), 2)
            exp_inv = (exp_min == exp_max)

            if m_dict["min"] != exp_min or m_dict["max"] != exp_max or m_dict["invariant"] != exp_inv or abs(float(m_dict["mean"]) - exp_mean) > 1e-2:
                return False, f"Contradiction between invocations and summary for '{metric_name}': expected min={exp_min}, max={exp_max}, mean={exp_mean}, inv={exp_inv}; got min={m_dict['min']}, max={m_dict['max']}, mean={m_dict['mean']}, inv={m_dict['invariant']}"

        # Check invariant scalar consistency
        mc_summary = telem_sum["model_calls"]
        mc_per_inv = case.get("model_calls_per_invocation")
        if mc_summary["invariant"]:
            if mc_per_inv != mc_summary["min"]:
                return False, f"model_calls_per_invocation ({mc_per_inv}) != invariant model_calls ({mc_summary['min']})"
        else:
            if mc_per_inv is not None:
                return False, f"model_calls_per_invocation must be null/None for non-invariant telemetry, but got {mc_per_inv}"

    return True, ""


def validate_benchmark_payload_for_comparison(payload: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Benchmark payload is not a dictionary"

    schema_v = payload.get("schema_version")
    if not schema_v or not isinstance(schema_v, str) or schema_v != SCHEMA_VERSION:
        return False, f"Schema mismatch or missing: expected {SCHEMA_VERSION!r}, got {schema_v!r}"

    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) == 0:
        return False, "Benchmark payload must have a non-empty 'cases' list"

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return False, "Benchmark payload must have a 'summary' dictionary"

    total_c = summary.get("total_cases")
    ok_c = summary.get("ok_cases")
    err_c = summary.get("error_cases")

    if not is_finite_int(total_c) or total_c != len(cases):
        return False, f"Summary total_cases ({total_c}) != actual cases count ({len(cases)})"

    actual_ok = sum(1 for c in cases if isinstance(c, dict) and c.get("status") == "ok")
    actual_err = sum(1 for c in cases if isinstance(c, dict) and c.get("status") == "error")

    if not is_finite_int(ok_c) or ok_c != actual_ok:
        return False, f"Summary ok_cases ({ok_c}) != actual ok count ({actual_ok})"
    if not is_finite_int(err_c) or err_c != actual_err:
        return False, f"Summary error_cases ({err_c}) != actual error count ({actual_err})"

    case_ids = set()
    for idx, c in enumerate(cases):
        if not isinstance(c, dict):
            return False, f"Case at index {idx} is not a dictionary"
        cid = c.get("case_id")
        if not cid or not isinstance(cid, str) or not cid.strip():
            return False, f"Case at index {idx} has invalid case_id: {cid!r}"
        if cid in case_ids:
            return False, f"Duplicate case_id found in benchmark payload: {cid!r}"
        case_ids.add(cid)

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
    original_sha256: str = ""
    mask_sha256: str = ""
    workload_sha256: str = ""
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
