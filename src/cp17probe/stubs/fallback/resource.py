"""Browser-only fallback for the POSIX `resource` module, used only if Pyodide cannot import its own.

`cp24.member.maxrss` and `cp21.lgbm` read `ru_maxrss` for a record field, never for a decision. The
fallback reports 0, which the probe's record states; the browser's memory is measured from the
WebAssembly heap instead.
"""
from types import SimpleNamespace

RUSAGE_SELF = 0
FALLBACK = True


def getrusage(who):
    return SimpleNamespace(ru_maxrss=0, ru_utime=0.0, ru_stime=0.0)
