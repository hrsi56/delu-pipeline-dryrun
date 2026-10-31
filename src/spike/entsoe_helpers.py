"""Shared constants and helpers for the Month-0 ENTSO-E/SMARD data-layer spike.

All probe scripts under scripts/ import from here. Keeps the EIC code, the
client factory, and JSON-evidence formatting in one place.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient, EntsoeRawClient

# DE-LU bidding zone (capstone_V6_0.md S3 / S0 item 2).
DE_LU = "DE_LU"
DE_LU_EIC = "10Y1001A1001A82H"

# 15-minute MTU transition: 24 hourly values/day before, 96 quarter-hourly after
# (capstone_V6_0.md S0 item 4 / S4.0 / R-4).
MTU_TRANSITION_DATE = "2025-10-01"

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "data" / "spike"


def get_token() -> str:
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise RuntimeError("ENTSOE_API_TOKEN not set in environment")
    return token


def get_client() -> EntsoePandasClient:
    return EntsoePandasClient(api_key=get_token())


def get_raw_client() -> EntsoeRawClient:
    return EntsoeRawClient(api_key=get_token())


def to_pairs(obj: pd.Series | pd.DataFrame, n: int = 4) -> dict:
    """JSON-safe {columns, head, tail} summary of a Series/DataFrame with a
    tz-aware DatetimeIndex. Values are rounded floats; index entries are ISO
    strings. Used to embed small, inspectable evidence snippets in the memo
    without dumping full series."""

    def rows(sub: pd.Series | pd.DataFrame) -> list:
        if isinstance(sub, pd.DataFrame):
            out = []
            for idx, row in sub.iterrows():
                out.append([str(idx)] + [None if pd.isna(v) else round(float(v), 4) for v in row])
            return out
        return [[str(idx), None if pd.isna(val) else round(float(val), 4)] for idx, val in sub.items()]

    return {
        "columns": list(obj.columns) if isinstance(obj, pd.DataFrame) else None,
        "n_rows": len(obj),
        "first_ts": str(obj.index[0]) if len(obj) else None,
        "last_ts": str(obj.index[-1]) if len(obj) else None,
        "inferred_freq": str(pd.infer_freq(obj.index)) if len(obj) > 2 else None,
        "head": rows(obj.head(n)),
        "tail": rows(obj.tail(n)),
    }


def series_to_dict(obj: pd.Series | pd.DataFrame) -> dict:
    """Full-resolution JSON-safe {iso_timestamp: value (or {col: value})} map.
    Used by the Q2 vintage snapshots, which need every point for a later diff."""
    if isinstance(obj, pd.DataFrame):
        return {
            str(idx): {col: (None if pd.isna(v) else float(v)) for col, v in row.items()}
            for idx, row in obj.iterrows()
        }
    return {str(idx): (None if pd.isna(val) else float(val)) for idx, val in obj.items()}


def write_evidence(name: str, evidence: dict) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_DIR / f"{name}.json"
    out.write_text(json.dumps(evidence, indent=2, default=str))
    return out
