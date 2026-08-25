import argparse
from pathlib import Path

import pytest

from bench.scripts import benchmark_slicer


def test_slicer_input_accepts_repository_local_data(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    bench_root = tmp_path / "bench"
    models_root = tmp_path / "models"
    results_root = bench_root / "results"
    data_root.mkdir()
    bench_root.mkdir()
    models_root.mkdir()
    source = data_root / "page.png"
    source.write_bytes(b"not-an-image")

    monkeypatch.setattr(benchmark_slicer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark_slicer, "DATA_ROOT", data_root.resolve())
    monkeypatch.setattr(benchmark_slicer, "BENCH_ROOT", bench_root.resolve())
    monkeypatch.setattr(benchmark_slicer, "MODELS_ROOT", models_root.resolve())
    monkeypatch.setattr(benchmark_slicer, "RESULTS_ROOT", results_root.resolve())

    assert benchmark_slicer._input_path("data/page.png") == source.resolve()


def test_slicer_input_rejects_escape_outside_repository(tmp_path, monkeypatch):
    data_root = tmp_path / "repo" / "data"
    bench_root = tmp_path / "repo" / "bench"
    models_root = tmp_path / "repo" / "models"
    data_root.mkdir(parents=True)
    bench_root.mkdir()
    models_root.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")

    monkeypatch.setattr(benchmark_slicer, "PROJECT_ROOT", tmp_path / "repo")
    monkeypatch.setattr(benchmark_slicer, "DATA_ROOT", data_root.resolve())
    monkeypatch.setattr(benchmark_slicer, "BENCH_ROOT", bench_root.resolve())
    monkeypatch.setattr(benchmark_slicer, "MODELS_ROOT", models_root.resolve())

    with pytest.raises(argparse.ArgumentTypeError):
        benchmark_slicer._input_path(str(outside))


def test_slicer_input_rejects_symlink_escape(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data_root = repo / "data"
    bench_root = repo / "bench"
    models_root = repo / "models"
    data_root.mkdir(parents=True)
    bench_root.mkdir()
    models_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret")
    link = data_root / "linked.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    monkeypatch.setattr(benchmark_slicer, "PROJECT_ROOT", repo)
    monkeypatch.setattr(benchmark_slicer, "DATA_ROOT", data_root.resolve())
    monkeypatch.setattr(benchmark_slicer, "BENCH_ROOT", bench_root.resolve())
    monkeypatch.setattr(benchmark_slicer, "MODELS_ROOT", models_root.resolve())

    with pytest.raises(argparse.ArgumentTypeError):
        benchmark_slicer._input_path("data/linked.png")


def test_slicer_output_is_filename_only_and_fixed_to_results(tmp_path, monkeypatch):
    results_root = tmp_path / "bench" / "results"
    monkeypatch.setattr(benchmark_slicer, "RESULTS_ROOT", results_root.resolve())

    assert benchmark_slicer._result_path("slicer.json") == results_root.resolve() / "slicer.json"
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark_slicer._result_path("../escape.json")
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark_slicer._result_path("nested/report.json")
