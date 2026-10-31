#!/usr/bin/env python3
"""Block a push that would put a placeholder or a non-final build record on `main`
(Publication Standard v1 §9, brief W10).

GitHub Pages serves `main` as soon as it is pushed, so an incomplete page must never reach it: no
`data-unpublished` marker in `docs/index.html`, and no build record
(`reports/cp3/pages_build.json`) with `"final": false`. This guard enforces that before the push.

    pre-push <remote> <url>   run by `.githooks/pre-push` after the secret guard; reads the refs
                              being pushed on stdin and checks every commit bound for `main`
    tree                      the CI backstop: checks the checked-out tree

It fails closed: a commit it cannot read, a missing page or build record, or a record it cannot
parse blocks the push. It uses only the standard library and Git, because the hook runs it with
the system Python, outside the project's environment. It never contacts a remote.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAGE = "docs/index.html"
BUILD_RECORD = "reports/cp3/pages_build.json"
PROTECTED = "refs/heads/main"
NULL_OID = "0" * 40
#: The placeholder markers the standard names: a link waiting for an unpublished destination.
MARKERS = ("data-unpublished",)


def findings_for(page: str | None, record: str | None) -> list[str]:
    """Why this page and build record may not reach `main`; empty when they may."""
    problems = []
    if page is None:
        problems.append(f"{PAGE} is missing")
    else:
        for marker in MARKERS:
            if marker in page:
                problems.append(f"{PAGE} carries a placeholder ({marker})")
    if record is None:
        problems.append(f"{BUILD_RECORD} is missing")
    else:
        try:
            final = json.loads(record).get("final")
        except (ValueError, AttributeError):
            return problems + [f"{BUILD_RECORD} is not a readable build record"]
        if final is not True:
            problems.append(f"{BUILD_RECORD} is a non-final build record (final: {json.dumps(final)})")
    return problems


def _show(commit: str, path: str, cwd: Path) -> str | None:
    result = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=cwd, capture_output=True)
    return result.stdout.decode("utf-8", "replace") if result.returncode == 0 else None


def pre_push(stdin: str, cwd: Path) -> list[str]:
    """Check the tree of every commit bound for `main` (deletions are not checked: nothing lands)."""
    problems = []
    for line in stdin.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 4:
            problems.append(f"cannot read the pushed ref line {line!r}")
            continue
        _local_ref, local_oid, remote_ref, _remote_oid = parts
        if remote_ref != PROTECTED or local_oid == NULL_OID:
            continue
        exists = subprocess.run(["git", "cat-file", "-e", f"{local_oid}^{{commit}}"], cwd=cwd, capture_output=True)
        if exists.returncode != 0:
            problems.append(f"cannot read commit {local_oid[:12]}")
            continue
        problems += [f"{local_oid[:12]}: {problem}" for problem in
                     findings_for(_show(local_oid, PAGE, cwd), _show(local_oid, BUILD_RECORD, cwd))]
    return problems


def tree(root: Path) -> list[str]:
    page, record = root / PAGE, root / BUILD_RECORD
    return findings_for(page.read_text() if page.is_file() else None,
                        record.read_text() if record.is_file() else None)


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    try:
        if mode == "pre-push":
            problems = pre_push(sys.stdin.read(), Path.cwd())
        elif mode == "tree":
            problems = tree(Path(argv[2]) if len(argv) > 2 else ROOT)
        else:
            print("usage: publication_guard.py pre-push <remote> <url> | tree [root]", file=sys.stderr)
            return 2
    except Exception as exc:  # fail closed
        print(f"publication-guard: BLOCKED - the guard could not run ({type(exc).__name__})", file=sys.stderr)
        return 1
    for problem in problems:
        print(f"publication-guard: BLOCKED - {problem}", file=sys.stderr)
    if problems:
        print("publication-guard: build the final page (scripts/build_pages.py --final) after the MLflow "
              "routes are verified; never bypass this guard (Publication Standard v1 §9).", file=sys.stderr)
        return 1
    if mode == "tree":
        print("publication-guard: the tree carries no placeholder and a final build record")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
