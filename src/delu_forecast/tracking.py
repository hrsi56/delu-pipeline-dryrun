"""DagsHub-hosted MLflow tracking (§9.1), with a no-op fallback.

DagsHub's MLflow endpoint authenticates with HTTP **basic** auth, not Bearer. The
stock client reads `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`; this
module bridges them from `DAGSHUB_USER_TOKEN` when they are unset, so a reviewer
needs one variable rather than three.

The token is never printed, logged, echoed into a run tag, or written to a file.
`redact()` scrubs it from any string that leaves this process, because a client
can leak a credential into a URL or an exception string -- a CP-1 lesson.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

EXPERIMENT_NAME = "delu-cp2"
TRACKING_URI_VAR = "MLFLOW_TRACKING_URI"
TOKEN_VAR = "DAGSHUB_USER_TOKEN"


def redact(text: str) -> str:
    token = os.environ.get(TOKEN_VAR)
    if token and len(token) >= 8:
        return text.replace(token, "<redacted>")
    return text


def bridge_dagshub_basic_auth() -> None:
    token = os.environ.get(TOKEN_VAR)
    if not token:
        return
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", token)
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)


def tracking_enabled() -> bool:
    return bool(os.environ.get(TRACKING_URI_VAR))


def configure_tracking(experiment: str = EXPERIMENT_NAME) -> bool:
    """Point MLflow at the remote tracking URI. Returns False when unconfigured."""
    if not tracking_enabled():
        return False
    bridge_dagshub_basic_auth()
    import mlflow

    mlflow.set_tracking_uri(os.environ[TRACKING_URI_VAR])
    mlflow.set_experiment(experiment)
    return True


def public_tracking_url() -> str | None:
    """The `.mlflow` URI itself -- never the DagsHub repo root, which 302s anonymous
    visitors to a sign-in page."""
    return os.environ.get(TRACKING_URI_VAR)


def code_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


SOURCE_PATHS: tuple[str, ...] = ("src", "scripts", "tests", "pyproject.toml", "uv.lock", "data")


def working_tree_dirty(paths: tuple[str, ...] | None = None) -> bool:
    """Is the tree dirty? Scoped to `paths` when given.

    A pipeline stage that runs after earlier stages have rewritten their own
    generated artifacts will always see a dirty tree, which says nothing about
    provenance. What matters for a decision-bearing run is whether the *source*
    that produced it was committed, so the callers pass `SOURCE_PATHS`.
    """
    command = ["git", "status", "--porcelain=v1"]
    if paths:
        command += ["--", *paths]
    try:
        output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True
    return bool(output.strip())


def snapshot_hash(path: Path | str = "data/snapshot.parquet") -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


TIMINGS_PATH = Path("reports/cp2/timings.json")


def record_timing(stage: str, seconds: float, path: Path | str = TIMINGS_PATH) -> None:
    """Merge one stage's wall-clock into reports/cp2/timings.json.

    §9.3 asks for a stated compute footprint. Measuring it beats asserting it,
    and a merged file lets `make cp2` leave one coherent record of the whole
    pipeline on the hardware it actually ran on.
    """
    import json
    import platform

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(target.read_text()) if target.exists() else {}
    payload.setdefault("hardware", f"{platform.machine()} / {platform.system()} {platform.release()}, CPU only")
    payload.setdefault("python", platform.python_version())
    payload[stage] = round(float(seconds), 1)
    payload["total_seconds"] = round(
        sum(value for key, value in payload.items() if key not in {"hardware", "python", "total_seconds"}), 1
    )
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@contextlib.contextmanager
def run(name: str, *, enabled: bool, tags: dict[str, Any] | None = None) -> Iterator[Any]:
    """Start an MLflow run, or yield None when tracking is not configured.

    CI and the committed test suite must stay runnable with no credential, so a
    missing tracking URI degrades to a no-op instead of an error.
    """
    if not enabled:
        yield None
        return
    import mlflow

    with mlflow.start_run(run_name=name, tags=tags) as active:
        yield active


def log_decision_record(
    enabled: bool,
    *,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path | str] | None = None,
) -> None:
    """Log the §9.1 decision-bearing record: params, metrics and artifact links."""
    if not enabled:
        return
    import mlflow

    mlflow.log_params({key: redact(str(value))[:490] for key, value in params.items()})
    mlflow.log_metrics({key: float(value) for key, value in metrics.items() if value == value})
    for artifact in artifacts or []:
        path = Path(artifact)
        if path.exists():
            mlflow.log_artifact(str(path))


__all__ = [
    "EXPERIMENT_NAME",
    "bridge_dagshub_basic_auth",
    "code_sha",
    "configure_tracking",
    "log_decision_record",
    "public_tracking_url",
    "record_timing",
    "redact",
    "run",
    "SOURCE_PATHS",
    "snapshot_hash",
    "tracking_enabled",
    "working_tree_dirty",
]
