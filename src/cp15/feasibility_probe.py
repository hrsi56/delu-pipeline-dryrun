"""One-origin Chronos-2 API/resource probe; never a scored CP-15 candidate.

Run with the isolated environment in reports/cp15/feasibility/requirements.freeze.txt.
Only fold-1 proper-training rows are scanned; delivery-day target is never loaded.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WEIGHT_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
DAY = date(2020, 4, 1)
THREADS = 2
MAX_RSS_BYTES = 4 * 1024**3
TIMEOUT_SECONDS = 600
REPORT = Path("reports/cp15/feasibility")
CACHE = Path("data/cp15-probe-cache")
ATTRIBUTION = "Data: ENTSO-E Transparency Platform; Bundesnetzagentur | SMARD.de — CC BY 4.0."


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_probe_day(day: date, partitions: dict) -> None:
    training = partitions["development_folds"][0]["proper_training"]
    if not date.fromisoformat(training["start"]) <= day <= date.fromisoformat(training["end"]):
        raise ValueError("probe must remain in fold-1 proper training")
    for fold in partitions["development_folds"]:
        evaluation = fold["evaluation"]
        if date.fromisoformat(evaluation["start"]) <= day <= date.fromisoformat(evaluation["end"]):
            raise ValueError("outer evaluation is forbidden")


def prepare_inputs():
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds

    protocol = json.loads(Path("reports/cp15/protocol.json").read_text())
    for name in ["data/snapshot.parquet", "data/partitions.json"]:
        if sha256(Path(name)) != protocol["input_sha256"][name]:
            raise ValueError(f"input differs from committed protocol: {name}")
    partitions = json.loads(Path("data/partitions.json").read_text())
    validate_probe_day(DAY, partitions)
    start = pd.Timestamp(DAY, tz="Europe/Berlin").tz_convert("UTC")
    context_start = start - pd.Timedelta(hours=168)
    end = pd.Timestamp(DAY, tz="Europe/Berlin") + pd.DateOffset(days=1)
    end = end.tz_convert("UTC")
    validate_probe_day(context_start.tz_convert("Europe/Berlin").date(), partitions)
    source = ds.dataset("data/snapshot.parquet", format="parquet")
    ts = ds.field("timestamp_utc")
    # Predicate and projection precede materialization; no full-snapshot frame.
    history = source.to_table(
        columns=["timestamp_utc", "price_eur_mwh", "load_forecast_mw"],
        filter=(ts >= context_start.to_pydatetime()) & (ts < start.to_pydatetime()),
    ).to_pandas().sort_values("timestamp_utc").reset_index(drop=True)
    future = source.to_table(
        columns=["timestamp_utc", "load_forecast_mw"],
        filter=(ts >= start.to_pydatetime()) & (ts < end.to_pydatetime()),
    ).to_pandas().sort_values("timestamp_utc").reset_index(drop=True)
    assert history.timestamp_utc.tolist() == pd.date_range(context_start, start, freq="h", inclusive="left").tolist()
    assert future.timestamp_utc.tolist() == pd.date_range(start, end, freq="h", inclusive="left").tolist()
    assert len(history) == 168 and len(future) == 24
    assert np.isfinite(history[["price_eur_mwh", "load_forecast_mw"]]).all().all()
    assert np.isfinite(future[["load_forecast_mw"]]).all().all()
    history = history.rename(columns={"timestamp_utc": "timestamp", "price_eur_mwh": "target"})
    future = future.rename(columns={"timestamp_utc": "timestamp"})
    for frame in [history, future]:
        frame["item_id"] = "DE-LU"
        # Canonical UTC -> naive UTC solely because Chronos expects regular time.
        frame["timestamp"] = frame.timestamp.dt.tz_convert("UTC").dt.tz_localize(None)
    REPORT.mkdir(parents=True, exist_ok=True)
    history.to_csv(REPORT / "context.csv", index=False)
    future.to_csv(REPORT / "future_load.csv", index=False)
    metadata = {
        "delivery_day": str(DAY), "partition": "fold_1.proper_training",
        "origin_utc": "2020-03-31T11:00:00Z", "origin_convention": "D-1 noon fixed CET UTC+01:00",
        "context_first_utc": context_start.isoformat(),
        "context_last_utc": (start-pd.Timedelta(hours=1)).isoformat(),
        "price_boundary_exclusive_utc": start.isoformat(), "context_hours": len(history),
        "forecast_first_utc": start.isoformat(), "forecast_end_exclusive_utc": end.isoformat(),
        "forecast_hours": len(future), "context_includes_DST_transition": True,
        "load_provenance": "Inherited A65 known-future assumption from protocol; snapshot does not prove original issue vintage.",
        "price_availability": "D-1 day-ahead prices were published at the previous auction; D price never read.",
        "external_input": "load_forecast_mw, historical context and 24 known-future values via future_df",
        "prohibited_columns_materialized": [], "scored": False,
        "attribution": ATTRIBUTION,
        "input_sha256": {str(p): sha256(p) for p in [Path("data/snapshot.parquet"), Path("data/partitions.json"), Path("reports/cp15/protocol.json"), REPORT/"context.csv", REPORT/"future_load.csv"]},
    }
    (REPORT / "input_manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return history, future


def worker(offline: bool) -> None:
    # These are set before numpy/torch imports, and do not enable remote logging.
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = str(THREADS)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_XET="1", DO_NOT_TRACK="1", WANDB_DISABLED="true")
    import numpy as np
    import torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import snapshot_download

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    history, future = prepare_inputs()
    verification = json.loads((REPORT/"model_verification.json").read_text())
    assert verification["revision"] == REVISION and verification["license"] == "apache-2.0"
    assert not verification["private"] and not verification["gated"]
    begin = time.monotonic()
    model_path = Path(snapshot_download(
        MODEL_ID, revision=REVISION, cache_dir=str(CACHE/"huggingface"),
        allow_patterns=["config.json", "model.safetensors", "README.md"],
        max_workers=1, token=False, local_files_only=offline,
    ))
    download_seconds = time.monotonic()-begin
    model_files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(model_path.iterdir()) if p.is_file()}
    assert sum(v["bytes"] for v in model_files.values()) < 1024**3
    assert model_files["model.safetensors"]["sha256"] == WEIGHT_SHA256
    begin = time.monotonic()
    pipeline = Chronos2Pipeline.from_pretrained(str(model_path), device_map="cpu", local_files_only=True, torch_dtype=torch.float32)
    load_seconds = time.monotonic()-begin
    args = dict(prediction_length=24, quantile_levels=[0.1, 0.5, 0.9], batch_size=4, context_length=168, cross_learning=False)
    begin = time.monotonic()
    with torch.inference_mode():
        predicted = pipeline.predict_df(history, future_df=future, **args)
    inference_seconds = time.monotonic()-begin
    # Positive control changes ONLY synthetic future load; no outcome is consulted.
    changed = future.copy()
    changed["load_forecast_mw"] *= 1.2
    begin = time.monotonic()
    with torch.inference_mode():
        perturbed = pipeline.predict_df(history, future_df=changed, **args)
    control_seconds = time.monotonic()-begin
    numeric = [c for c in predicted.columns if c not in ["item_id", "timestamp", "target_name"]]
    assert len(predicted) == 24
    assert np.isfinite(predicted[numeric].to_numpy()).all()
    delta = float(np.max(np.abs(predicted[numeric].to_numpy()-perturbed[numeric].to_numpy())))
    assert delta > 0, "future load positive control failed"
    predicted.to_csv(REPORT/"probe_predictions_unscored.csv", index=False)
    result = {
        "status": "completed_unscored", "evidence_class": "development_post_selection",
        "model_id": MODEL_ID, "revision": REVISION, "model_files": model_files,
        "model_cache_snapshot": str(model_path), "device": "cpu", "dtype": "float32",
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "seed": 42, "training_or_finetuning": False, "prediction_calls": 2,
        "download_or_cache_seconds": download_seconds, "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds, "positive_control_seconds": control_seconds,
        "prediction_rows": len(predicted), "prediction_columns": list(predicted.columns),
        "external_covariate_control": {"change": "only future load multiplied by 1.2", "maximum_prediction_delta_eur_mwh": delta, "outcomes_used": False},
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "attribution": ATTRIBUTION,
        "hashes": {str(p): sha256(p) for p in [Path(__file__).resolve().relative_to(Path.cwd()), REPORT/"requirements.freeze.txt", REPORT/"input_manifest.json", REPORT/"probe_predictions_unscored.csv"]},
        "limitations": ["Single origin and two inference calls, not a neural benchmark or accuracy claim.", "Foundation-model pretraining overlap with historical electricity data is not ruled out.", "Inherited load vintage assumption is retained, not independently proven.", "No GPU/MPS run; process RSS does not measure total unified-memory pressure."]
    }
    (REPORT/"probe_result.json").write_text(json.dumps(result, indent=2)+"\n")


def supervise(offline: bool) -> int:
    import psutil
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/"probe_result.json").write_text(json.dumps({"status": "running"})+"\n")
    begin = time.monotonic()
    command = [sys.executable, str(Path(__file__)), "--worker"] + (["--offline"] if offline else [])
    peak = 0
    stop_reason = None
    with (REPORT/"probe_execution.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss = monitored.memory_info().rss + sum(c.memory_info().rss for c in monitored.children(recursive=True))
                peak = max(peak, rss)
            except psutil.Error:
                pass
            if peak > MAX_RSS_BYTES or time.monotonic()-begin > TIMEOUT_SECONDS:
                stop_reason = "memory_bound" if peak > MAX_RSS_BYTES else "runtime_bound"
                process.kill()
                break
            time.sleep(0.1)
        exit_code = process.wait()
    resource = {"exit_code": exit_code, "elapsed_seconds": time.monotonic()-begin,
                "peak_sampled_process_tree_rss_bytes": peak, "rss_sample_period_seconds": 0.1,
                "rss_limit_bytes": MAX_RSS_BYTES, "wall_time_limit_seconds": TIMEOUT_SECONDS,
                "termination_reason": stop_reason, "worker_command": command,
                "host_physical_memory_bytes": psutil.virtual_memory().total}
    (REPORT/"resources.json").write_text(json.dumps(resource, indent=2)+"\n")
    if exit_code or stop_reason:
        (REPORT/"probe_result.json").write_text(json.dumps({"status": "failed", "resources": resource, "log": "probe_execution.log"}, indent=2)+"\n")
    print(json.dumps(resource, indent=2))
    return exit_code if exit_code else (1 if stop_reason else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare_inputs()
        return 0
    if args.worker:
        worker(args.offline)
        return 0
    return supervise(args.offline)


if __name__ == "__main__":
    raise SystemExit(main())
