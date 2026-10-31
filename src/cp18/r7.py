"""R7's rule wherever the pipeline verifies a bundle (r19 §24.12, R7): a raised error is a refused record.

CP-17c fixed R7's error path in `cp17.schema.verify_bundle` (a structurally malformed day of a re-hashed
`issued-forecasts.json` is listed as a fault). The rule stays anyway, for the frozen policy's life: any exception
`verify_bundle` raises, for whatever record, is turned into a refusal here, so nothing is ever issued, published or
served from a record the verifier could not judge. Every caller in the pipeline goes through `verify`, never through
`verify_bundle` directly (CP-18b's bundle and the Space's start-up check use it the same way).
"""
from __future__ import annotations


def verify(files: dict[str, bytes], verifier=None) -> dict:
    """{'accepted': bool, 'problems': [...], 'raised': None | 'ExceptionType'}: listed faults refuse the record, and
    so does a raised error."""
    if verifier is None:
        from cp17.schema import verify_bundle as verifier
    try:
        problems = list(verifier(files))
    except Exception as exc:          # R7: a raised error is a refused record, never a crash and never a pass
        return {'accepted': False, 'problems': [f'the verifier raised {type(exc).__name__}: the record is refused (R7)'],
                'raised': type(exc).__name__}
    return {'accepted': not problems, 'problems': problems, 'raised': None}
