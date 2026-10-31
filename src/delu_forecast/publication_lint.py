"""Publication checks over the rendered surfaces (Publication Standard v1 §4, §5, §8, §11).

This module holds the checks that read what a reader receives -- the built page, the README, the
Space cards and the MLflow export -- and compare it with the registry and the claim layer:

* **Registry consistency (§5).** Every registered generation and branch has each surface that
  derives from it, and no surface names an identity the registry does not hold.

Each check returns a list of problems, one line each; an empty list means it holds. The tests
(`tests/test_35_registry.py` and its companions) run every check on the committed surfaces and on
a deliberately broken copy, so a check that cannot fail is caught.
"""

from __future__ import annotations

import re

from . import registry as G

_VERSION_HREF = re.compile(r'href="#(v\d+)"')


def _section(document: str, opening: str, closing: str) -> str | None:
    start = document.find(opening)
    if start < 0:
        return None
    end = document.find(closing, start)
    return document[start:end if end >= 0 else len(document)]


# --------------------------------------------------------------------------- registry consistency (§5)


def page_consistency_problems(page: str) -> list[str]:
    """The page against the registry: chapters, rail, jump row, lineage and comparison rows."""
    problems: list[str] = []
    versions = [entry.version for entry in G.generations()]
    chapters = re.findall(r'<article class="chapter[^"]*" id="(v\d+)"', page)
    for version in versions:
        if version not in chapters:
            problems.append(f"page: registered generation {version} has no chapter")
    for version in chapters:
        if version not in versions:
            problems.append(f"page: chapter {version} is not a registered generation")
    newest_first = [entry.version for entry in G.generations(newest_first=True)]
    if [version for version in chapters if version in versions] != newest_first:
        problems.append(f"page: chapters are not in the registry's order {newest_first}")
    for name, opening, closing in (("rail", '<nav class="rail"', "</nav>"), ("jump row", '<nav class="jump"', "</nav>"),
                                   ("lineage", '<ol class="mainline"', "</ol>")):
        block = _section(page, opening, closing)
        if block is None:
            problems.append(f"page: no {name}")
            continue
        linked = _VERSION_HREF.findall(block)
        for version in versions:
            if version not in linked:
                problems.append(f"page: registered generation {version} is missing from the {name}")
        for version in linked:
            if version not in versions:
                problems.append(f"page: the {name} names {version}, which is not registered")
    # Every group of branch cards, compared as text: a name's version labels are declared structural.
    branches = " ".join(re.sub(r"<[^>]+>", "", match) for match in re.findall(r'<ul class="branches".*?</ul>', page, re.DOTALL))
    for entry in G.branches():
        if entry.name not in branches:
            problems.append(f"page: registered branch {entry.name!r} is missing from the lineage")
    overview = _section(page, 'data-chart-id="overview"', "</div>") or ""
    for entry in G.comparison_rows():
        source = G.comparison_source(entry)
        code = entry.code_in(source)
        if f'data-record="{G.comparison_prefix(source)}.metrics.{code}.equal_fold.S_MAE"' not in overview:
            problems.append(f"page: comparison row {entry.id} ({code}) is missing")
    return problems


def readme_consistency_problems(readme: str) -> list[str]:
    """The README against the registry: one heading per registered generation, none for another."""
    problems: list[str] = []
    headings = re.findall(r"^#{2,4} (.+)$", readme, re.MULTILINE)
    for entry in G.generations():
        if not any(heading == entry.name for heading in headings):
            problems.append(f"README: registered generation {entry.version} has no heading")
    registered = {entry.version for entry in G.generations()}
    for heading in headings:
        match = re.match(r"^(v\d+) · ", heading)
        if match and match.group(1) not in registered:
            problems.append(f"README: heading {heading!r} names an unregistered generation")
    return problems


def export_consistency_problems(files: dict[str, dict]) -> list[str]:
    """The MLflow export against the registry: the contract of `scripts/mlflow_export.py`."""
    problems = []
    runs = [run for name, content in files.items() if name != "manifest" for run in content["runs"]]
    keys = {run["run_key"] for run in runs}
    for key in G.expected_run_keys():
        if key not in keys:
            problems.append(f"export: registered run {key} is missing")
    for run in runs:
        key = run["run_key"]
        if key not in G.expected_run_keys():
            problems.append(f"export: run {key} is not registered")
        elif run["run_name"] != G.mlflow_run_name(key):
            problems.append(f"export: {key} is not named from the registry")
    return problems


__all__ = [
    "export_consistency_problems",
    "page_consistency_problems",
    "readme_consistency_problems",
]


# =========================================================================== the §4 lint

import ast  # noqa: E402
import html as _html  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from decimal import Decimal, InvalidOperation  # noqa: E402
from html.parser import HTMLParser  # noqa: E402
from pathlib import Path  # noqa: E402

from . import derived as D  # noqa: E402
from . import research as R  # noqa: E402

#: The attributes that bind a numeral: to an evidence or derived record, a v1 claim, a declared
#: structural kind, an axis scale, a release-check record or a registry status.
BINDINGS = ("data-record", "data-v1", "data-structural", "data-scale", "data-release-check", "data-status")
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


@dataclass
class Node:
    """One run of reading-path text, with the binding attributes of every element around it."""

    text: str
    bindings: dict[str, str] = field(default_factory=dict)
    chart: str | None = None


class ReadingPath(HTMLParser):
    """The reading path, mechanically (standard §1): the rendered page minus the bodies of closed
    disclosures and the value tables, counting only the desktop variant of each chart. Scripts,
    styles, the document head and an SVG's title and description are not read either."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict[str, str], bool]] = []
        self.nodes: list[Node] = []

    def _hidden(self) -> bool:
        return any(hidden for _, _, hidden in self.stack)

    def handle_starttag(self, tag, attrs):
        attributes = {key: (value or "") for key, value in attrs}
        parent_hidden = self._hidden()
        hidden = False
        if tag in ("script", "style", "head", "title", "desc", "template"):
            hidden = True
        if tag == "svg" and "m" in attributes.get("class", "").split():
            hidden = True
        if tag == "table" and "data" in attributes.get("class", "").split():
            hidden = True
        if "hidden" in attributes:
            hidden = True
        # inside a closed <details>, only its <summary> is on the reading path
        if self.stack and self.stack[-1][0] == "details" and "open" not in self.stack[-1][1] and tag != "summary":
            hidden = True
        if tag in VOID:
            return
        self.stack.append((tag, attributes, hidden or parent_hidden))

    def handle_startendtag(self, tag, attrs):
        pass  # self-closing SVG shapes carry no text

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        if self._hidden() or not data.strip():
            return
        bindings: dict[str, str] = {}
        chart = None
        for tag, attributes, _ in self.stack:
            for key in (*BINDINGS, "data-derived", "data-claim", "data-registry", "data-field", "data-display"):
                if key in attributes:
                    bindings[key] = attributes[key]
            if tag == "svg" and "data-chart" in attributes:
                chart = attributes["data-chart"]
        self.nodes.append(Node(data, bindings, chart))


def reading_path(document: str) -> list[Node]:
    parser = ReadingPath()
    parser.feed(document)
    parser.close()
    return parser.nodes


def reading_text(document: str) -> str:
    return " ".join(" ".join(node.text.split()) for node in reading_path(document))


# --------------------------------------------------------------------------- family: codes


CODE_PATTERNS = (
    ("checkpoint code", re.compile(r"\bCP-\d+\b")),
    ("claim ID", re.compile(r"\b[CPW]\d{1,3}\b")),
    ("section reference", re.compile(r"§\s?\d")),
    ("underscore identifier", re.compile(r"\b[A-Z]+_[A-Z_]+\b")),
)


def code_denylist() -> tuple[str, ...]:
    """Generated from the registry's code fields (standard §4): no hand-kept list. A code whose text is a registered
    generation's public version label (CP-24's `v5`, which CP-23 also used for its candidate) is that label on the
    reading path, not an internal code, and stays allowed; every other code stays denied."""
    versions = {entry.version for entry in G.generations()}
    return tuple(code for code in G.codes() if not re.fullmatch(r"CP-\d+", code) and code not in versions)


def code_findings(nodes: list[Node]) -> list[str]:
    findings = []
    denylist = [re.compile(rf"(?<![\w-]){re.escape(code)}(?![\w-])") for code in code_denylist()]
    for node in nodes:
        text = " ".join(node.text.split())
        for name, pattern in CODE_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(f"codes: {name} {match.group(0)!r} in {text[:80]!r}")
        for pattern in denylist:
            for match in pattern.finditer(text):
                findings.append(f"codes: registry code {match.group(0)!r} in {text[:80]!r}")
    return findings


# --------------------------------------------------------------------------- family: precision

_NUMERAL = re.compile(r"\d")
_NUMBER = re.compile(r"[−+-]?\d[\d,]*(?:\.\d+)?")

#: How each v1 claim may show on the reading path (standard §4). Invariant 4's statements keep
#: their exact form and are exempt; everything else follows the unit's rule.
V1_RULES = {
    "holdout_mae_champion": "eur", "holdout_mae_naive": "eur",
    "holdout_pinball_champion": "eur", "holdout_pinball_naive": "eur",
    "holdout_dm_p_value": "p_value", "holdout_days": "count", "wasm_cold_load_mb": "count",
    "development_dm_point_relative": "invariant_4", "development_dm_point_p_value": "invariant_4",
    "holdout_dm_label": "label", "attribution": "label", "licensing": "label",
}


def _significant(text: str) -> int:
    digits = text.replace(",", "").lstrip("−+-").replace(".", "").lstrip("0")
    return len(digits)


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def _near_zero_ok(text: str, raw: str) -> bool:
    """A value near zero keeps its sign and at least two significant figures (standard §4)."""
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return False
    sign_ok = (value > 0) == text.startswith("+") or (value < 0) == text.startswith(("−", "-"))
    return value != 0 and sign_ok and _significant(text) >= 2 and not re.fullmatch(r"[−+-]?0(\.0+)?", text)


def _unit_rule(node: Node) -> tuple[str, str | None]:
    """(rule, raw value) for a bound element: 'sig4', 'eur', 'count', 'percent', 'p_value', 'exempt'."""
    b = node.bindings
    record_id = b.get("data-record")
    if record_id:
        if D.is_derived(record_id):
            record = D.get(record_id)
            which = b.get("data-field", "value")
            raw = {"value": record.value, "ci_low": record.ci_low, "ci_high": record.ci_high}.get(which)
            return {D.UNIT_RELATIVE: "percent", D.UNIT_EUR: "eur", D.UNIT_COUNT: "count"}.get(record.unit, "exempt"), raw
        record = R.get(record_id)
        which = b.get("data-field", "value")
        raw = {"value": record.raw, "ci_low": record.ci_low_raw, "ci_high": record.ci_high_raw}.get(which)
        if record.unit in (R.UNIT_EUR, R.UNIT_EUR_DIFF):
            return "eur", raw
        if record.unit in (R.UNIT_RATIO, R.UNIT_NORM_DIFF, R.UNIT_FRACTION):
            return "sig4", raw
        if record.unit in R.THOUSANDS_UNITS or record.unit in (R.UNIT_SEED, R.UNIT_USD):
            return "count", raw
        return "exempt", raw
    if "data-v1" in b:
        rule = V1_RULES.get(b["data-v1"], "unknown")
        return ("exempt" if rule in ("invariant_4", "label") else rule), None
    return "structural", None


def precision_findings(nodes: list[Node]) -> list[str]:
    findings = []
    charts: dict[str, set[int]] = {}
    for node in nodes:
        text = " ".join(node.text.split())
        if not _NUMERAL.search(text):
            continue
        if not any(key in node.bindings for key in BINDINGS):
            findings.append(f"precision: an unbound numeral in {text[:80]!r}")
            continue
        rule, raw = _unit_rule(node)
        if rule in ("structural", "exempt"):
            continue
        shown = _NUMBER.search(text.replace(" ", ""))
        value = shown.group(0) if shown else text
        if rule == "p_value":
            if not (text.startswith("p < 10") or text.startswith("p = ")):
                findings.append(f"precision: a p-value shown as {text!r}")
            continue
        if raw is not None and rule in ("sig4", "eur"):
            try:
                shown_zero = Decimal(value.replace("−", "-").replace(",", "")) == 0
            except InvalidOperation:
                shown_zero = False
            if shown_zero and Decimal(raw) != 0:
                findings.append(f"precision: {value!r} rounds a nonzero value to zero across its sign "
                                f"({node.bindings.get('data-record')})")
                continue
        # "near zero": the value would round to zero at the unit's precision (standard §4)
        threshold = Decimal("0.05") if rule == "eur" else Decimal("0.00005")
        near_zero = raw is not None and _near_zero_ok(value, raw) and Decimal(raw).copy_abs() < threshold
        if rule == "sig4" and _significant(value) > 4 and not near_zero:
            findings.append(f"precision: {value!r} has more than four significant figures ({node.bindings.get('data-record')})")
        elif rule == "eur" and _decimals(value) != 1 and not near_zero:
            findings.append(f"precision: {value!r} in EUR/MWh is not shown to one decimal ({node.bindings.get('data-record') or node.bindings.get('data-v1')})")
        elif rule == "count" and "." in value.replace(",", ""):
            findings.append(f"precision: a count shown as {value!r}")
        elif rule == "percent" and not re.fullmatch(r"[−+-]?\d+%?", value.rstrip("%") + ("%" if text.endswith("%") else "")):
            findings.append(f"precision: a relative change shown as {text!r}, not a whole percent")
        if node.chart and rule in ("sig4", "eur") and not near_zero:
            charts.setdefault(node.chart, set()).add(_decimals(value))
    for chart, places in charts.items():
        if len(places) > 1:
            findings.append(f"precision: chart {chart} uses {len(places)} precisions {sorted(places)} (one per chart)")
    return findings


# --------------------------------------------------------------------------- family: percentages


def percent_findings(nodes: list[Node]) -> list[str]:
    """A `%` passes only as a derived relative change, a structural level or an exempt v1 statement."""
    findings = []
    for node in nodes:
        if "%" not in node.text:
            continue
        b = node.bindings
        derived = b.get("data-derived") in ("share_change", "ratio_change", "distance", "rule_margin")
        level = b.get("data-structural") == "level"
        exempt = V1_RULES.get(b.get("data-v1", ""), "") == "invariant_4"
        if not (derived or level or exempt):
            findings.append(f"percentages: {' '.join(node.text.split())[:80]!r} is not a derived relative change, "
                            "a structural level or an exempt v1 statement")
    return findings


# --------------------------------------------------------------------------- family: status words (template sources)

#: The status words the standard names, and common synonyms. They reach a surface only through a
#: registry token; a template source that types one is a finding (standard §5).
STATUS_WORDS = re.compile(
    r"\b(current(ly)?|latest|newest(?![ -]first)|still|now|today|remains?|yet|no longer|at present|so far|"
    r"is being|the demo runs|currently running)\b", re.IGNORECASE)

#: The template sources: every piece of copy a generator can put on a public surface. The registry
#: is the token source, and v1's archive is a frozen record (standard §7), so neither is linted.
TEMPLATE_SOURCES = {
    "src/delu_forecast/research_claims.py": ("BLOCKS", "headline_template", "target_sentence"),
    "scripts/build_pages.py": None,
    "scripts/readme_research.py": ("V1_READ", "build_glance", "generation_section", "branch_section", "build_block",
                                   "audit_links", "not_established"),
    "scripts/build_space.py": ("model_lines", "card_body", "build_card", "docker_deployed_section"),
    "scripts/build_wasm_space.py": ("build_card", "static_deployed_section", "startup_markup"),
    "scripts/cp3_readme.py": ("build_section",),
    "scripts/mlflow_export.py": ("EXPERIMENT_TAGS", "CHECKPOINTS", "_description", "_readme"),
}
#: Functions and constants in the page generator that are not copy: v1's frozen archive, the
#: local-only specimens, the stylesheet and the scripts.
NOT_COPY = {"v1_archive", "v1_archive_disclosure", "css", "token_sheet", "stress_chapters", "CHART_JS", "NAV_JS",
            "FIGURE_JS", "main"}


def _strings(tree: ast.AST) -> list[tuple[int, str]]:
    out = []
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            out.append((node.lineno, node.value))
    return out


def template_strings(repo_root: Path) -> list[tuple[str, int, str]]:
    """(file, line, text) of every string literal in the template sources, minus docstrings and the
    parts that are not copy. Registry tokens (`{g:…}`) are removed before a string is read."""
    out = []
    for relative, scope in TEMPLATE_SOURCES.items():
        tree = ast.parse((repo_root / relative).read_text())
        roots = []
        for node in tree.body:
            if isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom)):
                continue  # the module docstring and imports are not copy
            name = getattr(node, "name", None)
            if name is None and isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                name = getattr(target, "id", None)
            if name in NOT_COPY:
                continue
            if scope is None or name in scope:
                roots.append(node)
        for root in roots:
            for line, text in _strings(root):
                out.append((relative, line, re.sub(r"\{g:[^{}]+\}", "", text)))
    return out


def status_findings(strings: list[tuple[str, int, str]]) -> list[str]:
    findings = []
    for relative, line, text in strings:
        stripped = re.sub(r"<[^>]+>", " ", text)          # markup and attribute names are not copy
        stripped = re.sub(r"\b(aria|data)-[\w-]+", " ", stripped)
        for match in STATUS_WORDS.finditer(stripped):
            findings.append(f"status: {relative}:{line} types {match.group(0)!r} (status comes from the registry)")
    return findings


# --------------------------------------------------------------------------- the whole lint


def lint_document(document: str) -> list[str]:
    nodes = reading_path(document)
    return code_findings(nodes) + precision_findings(nodes) + percent_findings(nodes)


def lint(repo_root: Path, page: str, readme_glance_html: str) -> dict[str, list[str]]:
    """Every family on the page's reading path, the README's generated top block and the template
    sources. Empty lists mean the publication passes."""
    return {
        "page": lint_document(page),
        "readme": lint_document(readme_glance_html),
        "templates": status_findings(template_strings(repo_root)),
    }


# =========================================================================== cross-surface parity (§8)

_STATUS_SENTENCE = re.compile(
    r"In (?P<month>[A-Z][a-z]+ \d{4}), (?P<subject>v\d+|the [a-z][\w -]*?) "
    r"(?P<verb>was adopted in research|was released|was not adopted|became the final candidate|went live|was retired)")
#: A version followed by its name; a difference such as "v3 − v2 · paired interval" is not a name.
_VERSION_NAME = re.compile(r"(?<!− )(?<!- )\b(v\d+) · ([^<>\n|()*\[\]`]+)")


def _plain(text: str) -> str:
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.DOTALL)
    text = _html.unescape(re.sub(r"<[^>]+>", "", text))
    return " ".join(text.replace("`", "").replace("**", "").split())


def parity_problems(surfaces: dict[str, str], *, page_headline: str, readme_headline: str) -> list[str]:
    """The headline block, the names and the statuses agree on every surface (standard §8).

    `surfaces` maps a surface name to its text (HTML or Markdown). Outside v1's frozen archive, a
    version-prefixed name must be its generation's canonical name, a version must be registered,
    and a dated status sentence must be the registry's."""
    problems = []
    if _plain(page_headline) != _plain(readme_headline):
        problems.append("headline: the page's and the README's headline blocks differ")
    canonical = {entry.version: entry.name for entry in G.generations()}
    statuses = {G.status_sentence(entry)[:-1] for entry in G.entries() if entry.status is not None}
    for name, text in surfaces.items():
        plain = _plain(text)
        for version, rest in _VERSION_NAME.findall(plain):
            if version not in canonical:
                problems.append(f"{name}: {version} is not a registered generation")
                continue
            wanted = canonical[version].split(" · ", 1)[1]
            if not (rest.strip().startswith(wanted) or wanted.startswith(rest.strip().rstrip(".,;:"))):
                problems.append(f"{name}: '{version} · {rest.strip()[:40]}' is not the canonical name {canonical[version]!r}")
        for match in _STATUS_SENTENCE.finditer(plain):
            if match.group(0) not in statuses:
                problems.append(f"{name}: the status {match.group(0)!r} is not the registry's")
    return problems


def surface_texts(repo_root: Path) -> dict[str, str]:
    """Every public surface, v1's frozen archive and the historical CP-3 record set aside."""
    page = (repo_root / "docs" / "index.html").read_text()
    start = page.find('<details class="disclosure archive" id="v1-archive">')
    if start >= 0:
        end = page.find("</article>", start)
        page = page[:start] + page[end:]
    readme = (repo_root / "README.md").read_text()
    readme = readme[:readme.find("## CP-1 data and fixed features")] if "## CP-1 data" in readme else readme
    import json as _json

    export = []
    for name in G.parent_run_keys():
        for run in _json.loads((repo_root / "reports/presentation/mlflow-export" / f"{name}.json").read_text())["runs"]:
            export.append(run["run_name"] + " " + run["tags"].get("mlflow.note.content", "")
                          + " " + run["tags"].get("delu.public_name", ""))
    return {
        "page": page,
        "README": readme,
        "Static Space card": (repo_root / "space-wasm" / "README.md").read_text(),
        "Space card": (repo_root / "space" / "README.md").read_text(),
        "MLflow export": " ".join(export),
    }
