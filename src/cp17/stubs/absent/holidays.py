"""Pyodide-only import stand-in for `holidays`, which Pyodide 0.28.0 does not ship (frozen with the package).

`delu_forecast.features` imports it at module level and uses it only to build calendar features, which the
payloads already carry. Any attribute access raises, so a daily fit that reached it would fail rather than
run on a substitute. This directory is appended to `sys.path` only in the Pyodide hosts (Node.js and the
browser), after every real package, and never natively.
"""


def __getattr__(name):
    raise RuntimeError(f'holidays.{name} is unavailable in the Pyodide runtime; the daily fit must not reach it')
