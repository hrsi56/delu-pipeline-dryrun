#!/usr/bin/env python3
"""Block commits and pushes that contain the value of a local credential.

Git runs this through `.githooks/` (`core.hooksPath`), per AGENTS.md § Credentials:

    pre-commit   every staged blob
    commit-msg   the commit message
    pre-push     every blob and message in commits that no remote-tracking ref contains

It compares values, not shapes. On 2026-09-24 a DagsHub token reached a public commit
inside a pytest environment dump. The token is a bare 40-character hex string, so no
pattern scan recognised it. Values come from the environment, `launchctl getenv` and the
`export NAME="..."` lines of `~/.zshrc`. They cover the KNOWN names plus any variable whose
name looks like a credential. A finding names the variable and the path, never the value.

SECRET_GUARD_ENV_ONLY=1 restricts the sources to the process environment (used by tests).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

KNOWN = ("DAGSHUB_USER_TOKEN", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
         "ENTSOE_API_TOKEN", "HF_TOKEN")
CREDENTIAL_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|PRIVATE_KEY", re.I)
EXPORT = re.compile(r"""^\s*export\s+([A-Za-z_]\w*)=(["']?)([^"'\s$`]+)\2\s*$""", re.M)
MIN_LENGTH = 12
NULL_OID = "0" * 40


def _git(*args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], input=data, capture_output=True, check=True).stdout


def _launchctl(name: str) -> str:
    try:
        return subprocess.run(["launchctl", "getenv", name], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _zshrc_exports() -> dict[str, str]:
    try:
        text = (Path.home() / ".zshrc").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return {m.group(1): m.group(3) for m in EXPORT.finditer(text)}


def credentials() -> list[tuple[str, bytes]]:
    """(name, value) pairs, one per distinct value. Values are never printed."""
    names = set(KNOWN) | {k for k in os.environ if CREDENTIAL_NAME.search(k)}
    found = [(name, os.environ.get(name, "")) for name in sorted(names)]
    if os.environ.get("SECRET_GUARD_ENV_ONLY") != "1":
        exports = _zshrc_exports()
        names |= {k for k in exports if CREDENTIAL_NAME.search(k)}
        for name in sorted(names):
            found += [(name, _launchctl(name)), (name, exports.get(name, ""))]
    seen: dict[bytes, str] = {}
    for name, value in found:
        value = value.strip().strip("“”\"'")
        if len(value) >= MIN_LENGTH:
            seen.setdefault(value.encode(), name)
    return [(name, value) for value, name in seen.items()]


def _changed_blobs(raw: bytes) -> list[tuple[str, str]]:
    """(blob oid, path) for added or modified files in `--raw -z` diff output."""
    fields = raw.split(b"\0")
    blobs = []
    for meta, path in zip(fields[0::2], fields[1::2]):
        parts = meta.decode().lstrip(":").split()
        if (len(parts) == 5 and parts[4][0] in "ACMRT" and parts[3] != NULL_OID
                and parts[1] != "160000"):  # a submodule entry names a commit, not a blob
            blobs.append((parts[3], path.decode(errors="replace")))
    return blobs


def _read_blobs(oids: list[str]) -> dict[str, bytes]:
    out = _git("cat-file", "--batch", data="".join(f"{oid}\n" for oid in oids).encode())
    blobs, pos = {}, 0
    for oid in oids:
        header_end = out.index(b"\n", pos)
        size = int(out[pos:header_end].split()[2])
        blobs[oid] = out[header_end + 1:header_end + 1 + size]
        pos = header_end + 1 + size + 1
    return blobs


def _scan(items: list[tuple[str, bytes]], secrets: list[tuple[str, bytes]]) -> list[str]:
    return [f"the value of {name} appears in {where}"
            for where, content in items for name, value in secrets if value in content]


def _staged(secrets):
    raw = _git("diff", "--cached", "--raw", "-z", "--no-renames", "--no-abbrev")
    blobs = _changed_blobs(raw)
    contents = _read_blobs(sorted({oid for oid, _ in blobs}))
    return _scan([(f"{path} (staged)", contents[oid]) for oid, path in blobs], secrets)


def _message(path: str, secrets):
    return _scan([("the commit message", Path(path).read_bytes())], secrets)


def _outgoing(remote: str, stdin: str, secrets):
    commits: list[str] = []
    for line in stdin.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == NULL_OID:
            continue
        exclude = [f"--remotes={remote}"] if remote else []
        listed = _git("rev-list", parts[1], "--not", *exclude).decode().split()
        commits += [c for c in listed if c not in commits]
    items = []
    for commit in commits:
        raw = _git("diff-tree", "-r", "-m", "--root", "--no-commit-id", "--raw", "-z",
                   "--no-renames", "--no-abbrev", commit)
        blobs = _changed_blobs(raw)
        contents = _read_blobs(sorted({oid for oid, _ in blobs}))
        items += [(f"{path} (commit {commit[:10]})", contents[oid]) for oid, path in blobs]
        items.append((f"the message of commit {commit[:10]}", _git("log", "-1", "--format=%B", commit)))
    return _scan(items, secrets)


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    secrets = credentials()
    if not secrets:
        return 0
    if mode == "pre-commit":
        findings = _staged(secrets)
    elif mode == "commit-msg":
        findings = _message(argv[2], secrets)
    elif mode == "pre-push":
        findings = _outgoing(argv[2] if len(argv) > 2 else "", sys.stdin.read(), secrets)
    else:
        print("usage: secret_guard.py pre-commit | commit-msg <file> | pre-push <remote> <url>",
              file=sys.stderr)
        return 2
    for finding in dict.fromkeys(findings):
        print(f"secret-guard: BLOCKED - {finding}", file=sys.stderr)
    if findings:
        print("secret-guard: remove the value and retry; never bypass this guard "
              "(AGENTS.md § Credentials).", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
