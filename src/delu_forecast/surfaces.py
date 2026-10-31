"""The four public surfaces, and whether they agree (CP-3 item 5).

Item 5 binds the README, the static Pages export, the Space metadata and the
MLflow record to the same claims. This module loads each surface into comparable
text and reports, claim by claim and surface by surface, which ones carry it.

Two deliberate choices:

* **Whitespace is normalised, nothing else.** A paragraph wrapped differently in
  Markdown and in HTML is the same claim; a number rounded differently is not.
  So `25.9078` and `25.91` are a disagreement, which is the point.
* **The MLflow surface is compared exactly, not by substring.** Its claims live
  in model-version tags, so the check is `tag == claim`, and a tag that merely
  contains the claim is a miss.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .claims import REPO_ROOT, REQUIRED_ON_EVERY_SURFACE, build_claims, forbidden_dagshub_links

README_PATH = REPO_ROOT / "README.md"
PAGES_PATH = REPO_ROOT / "docs" / "index.html"
SPACE_CARD_PATH = REPO_ROOT / "space" / "README.md"
#: M3.5/CP-3B: the card the hosted Static Space actually carries. The container
#: card above stays a surface too -- it is a public file and still makes claims.
STATIC_SPACE_CARD_PATH = REPO_ROOT / "space-wasm" / "README.md"
MLFLOW_RECORD_PATH = REPO_ROOT / "reports" / "cp3" / "mlflow_registration.json"

#: Claim key -> MLflow model-version tag key. Identity except where §9.1's tag
#: vocabulary already had a name for the thing.
MLFLOW_TAG_FOR_CLAIM: dict[str, str] = {"champion_fingerprint": "artifact_fingerprint_sha256"}

#: The four §7.1 cutoffs, called out separately because item 5 names them
#: separately: "All four cutoffs ... appear separately and identically on every
#: surface."
CUTOFF_KEYS: tuple[str, ...] = (
    "snapshot_cutoff",
    "raw_model_fit_cutoff",
    "final_calibration_window",
    "holdout_window",
)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", text).strip()


def html_to_text(document: str) -> str:
    """Strip markup and unescape entities, leaving claim strings intact.

    Tags become spaces rather than nothing: collapsing `<td>a</td><td>b</td>` to
    `ab` would invent a claim that is not on the page.
    """
    body = document
    for block in ("script", "style"):
        body = re.sub(rf"<{block}\b.*?</{block}>", " ", body, flags=re.DOTALL | re.IGNORECASE)
    # `<code>x</code>` is the HTML rendering of Markdown's `x`, so restore the
    # backticks before the tags go. Without this a claim containing a code span
    # could never match on both a Markdown surface and an HTML one, and the two
    # would be reported as disagreeing while saying exactly the same thing.
    body = re.sub(r"<code>(.*?)</code>", r"`\1`", body, flags=re.DOTALL | re.IGNORECASE)
    return normalise(html.unescape(_TAG.sub(" ", body)).replace("\u00a0", " "))


@dataclass(frozen=True)
class Surface:
    name: str
    path: Path
    text: str
    tags: dict[str, str] | None = None

    def carries(self, key: str, value: str) -> bool:
        if self.tags is not None:
            return self.tags.get(MLFLOW_TAG_FOR_CLAIM.get(key, key)) == value
        return normalise(value) in self.text


def load_surfaces() -> list[Surface]:
    """Every public surface bound to the claim set. A missing one is a hard error, not a skip."""
    surfaces: list[Surface] = []
    for name, path in (
        ("README", README_PATH),
        ("Space card", SPACE_CARD_PATH),
        ("Static Space card", STATIC_SPACE_CARD_PATH),
    ):
        surfaces.append(Surface(name, path, normalise(path.read_text())))
    surfaces.append(Surface("Pages export", PAGES_PATH, html_to_text(PAGES_PATH.read_text())))

    record = json.loads(MLFLOW_RECORD_PATH.read_text())
    surfaces.append(
        Surface(
            "MLflow record",
            MLFLOW_RECORD_PATH,
            normalise(json.dumps(record, sort_keys=True)),
            tags=dict(record.get("version_tags", {})),
        )
    )
    return surfaces


def agreement_matrix(
    surfaces: list[Surface] | None = None, keys: tuple[str, ...] = REQUIRED_ON_EVERY_SURFACE
) -> dict[str, dict[str, bool]]:
    """`{claim key: {surface name: carries it}}` for every bound claim."""
    claims = build_claims()
    surfaces = surfaces if surfaces is not None else load_surfaces()
    return {
        key: {surface.name: surface.carries(key, claims[key]) for surface in surfaces}
        for key in keys
    }


def disagreements(
    surfaces: list[Surface] | None = None, keys: tuple[str, ...] = REQUIRED_ON_EVERY_SURFACE
) -> list[str]:
    """One line per claim that is missing from at least one surface."""
    claims = build_claims()
    matrix = agreement_matrix(surfaces, keys)
    problems = []
    for key, row in matrix.items():
        missing = sorted(name for name, present in row.items() if not present)
        if missing:
            problems.append(f"{key} (= {claims[key][:70]!r}) missing from: {', '.join(missing)}")
    return problems


def link_discipline_failures(surfaces: list[Surface] | None = None) -> list[str]:
    """Gated DagsHub UI links found on any surface. The raw file is scanned, not
    the normalised text, so a link inside an HTML attribute is still seen."""
    surfaces = surfaces if surfaces is not None else load_surfaces()
    failures = []
    for surface in surfaces:
        for link in forbidden_dagshub_links(surface.path.read_text()):
            failures.append(f"{surface.name}: {link}")
    return failures


def render_matrix(matrix: dict[str, dict[str, bool]]) -> str:
    """A markdown table, surface by surface and claim by claim."""
    names = list(next(iter(matrix.values())).keys())
    lines = ["| Claim | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, row in matrix.items():
        lines.append(f"| `{key}` | " + " | ".join("yes" if row[name] else "**NO**" for name in names) + " |")
    return "\n".join(lines)


__all__ = [
    "CUTOFF_KEYS",
    "MLFLOW_RECORD_PATH",
    "MLFLOW_TAG_FOR_CLAIM",
    "PAGES_PATH",
    "README_PATH",
    "SPACE_CARD_PATH",
    "STATIC_SPACE_CARD_PATH",
    "Surface",
    "agreement_matrix",
    "disagreements",
    "html_to_text",
    "link_discipline_failures",
    "load_surfaces",
    "normalise",
    "render_matrix",
]
