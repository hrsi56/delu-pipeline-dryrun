"""The NumPy-only import audit (capstone v21-r10 §21.3, §18.2).

DDNN's model code may import NumPy and the Python standard library only. The audit parses each
audited module's source (it never imports it) and lists every imported top-level module, including
imports nested inside functions or conditionals and `__import__`/`importlib.import_module` calls with
a literal name. Anything outside NumPy and `sys.stdlib_module_names` fails the audit.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

#: The modules that hold DDNN's model code (§18.2 item 1).
AUDITED = ('src/cp23/ddnn.py',)
ALLOWED_THIRD_PARTY = frozenset({'numpy'})
#: `__future__` is a compiler directive, listed with the standard library.
STDLIB = frozenset(sys.stdlib_module_names) | {'__future__'}


def imported_modules(source: str) -> list[str]:
    """Every top-level module name the source imports, in order of appearance."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name.split('.')[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.append('.' + (node.module or node.names[0].name))  # relative: refused (DDNN code is one module)
            else:
                found.append(node.module.split('.')[0])
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', None)
            if name in ('__import__', 'import_module') and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                found.append(node.args[0].value.split('.')[0])
    return found


def violations(source: str) -> list[str]:
    return sorted({m for m in imported_modules(source) if m not in STDLIB and m not in ALLOWED_THIRD_PARTY})


def audit(root: Path, paths=AUDITED) -> dict:
    result = {}
    for name in paths:
        source = (Path(root) / name).read_text()
        result[name] = {'imports': sorted(set(imported_modules(source))), 'violations': violations(source)}
    return {'modules': result, 'passed': all(not r['violations'] for r in result.values()),
            'allowed': 'numpy and the Python standard library (sys.stdlib_module_names)'}
