"""The NumPy-only import audit for DDNN-2 and its search (capstone v21-r11 §23.3, §23.7; §18.2, §21.3).

DDNN-2's model code (`src/cp24/ddnn2.py`) and its search procedure (`src/cp24/sampler.py`: the space,
the seeded sampler, successive halving, the ranking and the ensemble choice) may import NumPy and the
Python standard library only; the search may also import the audited model module itself (a relative
import of `ddnn2`). The audit parses each audited module's source -- it never imports it -- and lists
every imported top-level module, including imports nested inside functions or conditionals and
`__import__`/`importlib.import_module` calls with a literal name (`cp23.audit.imported_modules`, reused
unchanged). Anything else fails.

The common admitted feature pipeline (`cp24.design`, which reads CP-15's prepared inputs and CP-20's
frozen weather) and the orchestration around the fits (data loading, worker pools, the ledger and the
caches) are outside this rule, as §18.2 item 3 states for the feature pipeline.
"""
from __future__ import annotations

from pathlib import Path

from cp23.audit import ALLOWED_THIRD_PARTY, STDLIB, imported_modules

AUDITED = ('src/cp24/ddnn2.py', 'src/cp24/sampler.py')
#: Relative imports an audited module may make: only of another audited module.
ALLOWED_RELATIVE = frozenset({'.ddnn2'})


def violations(source: str) -> list[str]:
    found = imported_modules(source)
    return sorted({m for m in found if m not in STDLIB and m not in ALLOWED_THIRD_PARTY and m not in ALLOWED_RELATIVE})


def audit(root: Path, paths=AUDITED) -> dict:
    result = {}
    for name in paths:
        source = (Path(root) / name).read_text()
        result[name] = {'imports': sorted(set(imported_modules(source))), 'violations': violations(source)}
    return {'modules': result, 'passed': all(not r['violations'] for r in result.values()),
            'allowed': 'numpy, the Python standard library (sys.stdlib_module_names), and a relative import of the '
                       'audited ddnn2 module'}
