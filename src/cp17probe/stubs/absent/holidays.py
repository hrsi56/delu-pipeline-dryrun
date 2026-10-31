"""Browser-only import stand-in for `holidays`, which Pyodide 0.28.0 does not ship.

`delu_forecast.features` imports it at module level and uses it only to build calendar features,
which the payload already carries. Any attribute access raises, so a daily fit that reached it
would fail rather than run on a substitute. This directory is appended to `sys.path` only in the
browser, after every real package, and never natively.
"""


def __getattr__(name):
    raise RuntimeError(f'holidays.{name} is unavailable in the browser probe; the daily fit must not reach it')
