"""The development boundary guard (capstone v21-r13 §24.2), the issue time (§24.4) and the
credential rule (AGENTS.md § Credentials). Standard library only: the GFS extraction environment
and the modelling environment both import it.

* **Rows.** A row is refused when its Europe/Berlin delivery day is after 2026-09-06.
* **GFS.** The guard tests the run's initialization, not its valid time: a run initialized at or
  after 2026-09-06 00 UTC is refused, and so is any lead outside §15.2's f021–f048. A forecast's
  valid time after the boundary is not an outcome, so the 2026-09-05 00 UTC f048 message (which
  serves delivery day 2026-09-06) is admitted.
* **URLs.** Every GFS request is refused unless its URL parses as one of the two anonymous
  archives' object, index or listing forms for an admitted (run, lead), and unless it carries no
  credential: no userinfo, and no token-like query key. Unknown shapes fail closed.
* **Credentials.** Nothing here reads a credential except to *redact* it: `redact` replaces the
  current value of each stored credential variable, inside this process, before text reaches a
  log, and `scan` reports only which variable names occur in which files, never a value.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import os
from pathlib import Path
import re
import urllib.parse
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo('Europe/Berlin')
UTC = timezone.utc
BOUNDARY = date(2026, 9, 6)                          # last development delivery day (§24.2)
FLOOR = date(2019, 1, 1)                             # the 2019 input floor (§5)
GFS_RUN_CUTOFF = datetime(2026, 9, 6, 0, 0, tzinfo=UTC)
LEAD_MIN, LEAD_MAX = 21, 48                          # §15.2's f021–f048
ISSUE_LOCAL = time(11, 0)                            # D−1 11:00 Europe/Berlin (§24.4)

CREDENTIAL_VARIABLES = ('DAGSHUB_USER_TOKEN', 'MLFLOW_TRACKING_USERNAME', 'MLFLOW_TRACKING_PASSWORD',
                        'ENTSOE_API_TOKEN', 'HF_TOKEN')
TOKEN_QUERY_KEYS = frozenset({'securitytoken', 'token', 'access_token', 'api_key', 'apikey', 'key', 'password',
                              'passwd', 'secret', 'auth', 'authorization', 'x-amz-security-token',
                              'x-amz-credential', 'x-amz-signature', 'signature', 'sig'})
MIN_REDACT_LENGTH = 6   # shorter values would redact ordinary words; none of the stored tokens is that short


class BoundaryRefused(ValueError):
    """An input on the far side of the development boundary, or outside §15.2's leads."""


class CredentialRefused(ValueError):
    """A URL that carries, or could carry, a credential."""


# ---------------------------------------------------------------------------------- rows

def delivery_day(instant: datetime) -> date:
    """The Europe/Berlin delivery day of an aware instant."""
    if instant.tzinfo is None:
        raise BoundaryRefused('naive instant: the delivery day is undefined without a timezone')
    return instant.astimezone(BERLIN).date()


def admit_row(instant: datetime) -> date:
    """Refuse a row whose Europe/Berlin delivery day is after the boundary; return its day."""
    day = delivery_day(instant)
    if day > BOUNDARY:
        raise BoundaryRefused(f'row at {instant.astimezone(UTC):%Y-%m-%dT%H:%MZ} is delivery day {day}, '
                              f'after the development boundary {BOUNDARY}')
    return day


def admit_rows(instants) -> int:
    """Refuse a batch if any row is after the boundary (fail closed: nothing is kept); return the count."""
    n = 0
    for instant in instants:
        admit_row(instant if isinstance(instant, datetime) else instant.to_pydatetime())
        n += 1
    return n


def admit_delivery_day(day: date) -> date:
    if day > BOUNDARY:
        raise BoundaryRefused(f'delivery day {day} is after the development boundary {BOUNDARY}')
    return day


# ---------------------------------------------------------------------------------- GFS

def run_init(run_day: date) -> datetime:
    """The 00 UTC initialization of a run date."""
    return datetime.combine(run_day, time(0, 0), tzinfo=UTC)


def admit_gfs(init: datetime, lead: int) -> None:
    """Refuse a run initialized at or after 2026-09-06 00 UTC, or a lead outside f021–f048."""
    if init.tzinfo is None:
        raise BoundaryRefused('naive GFS initialization')
    if init.astimezone(UTC) >= GFS_RUN_CUTOFF:
        raise BoundaryRefused(f'GFS run {init.astimezone(UTC):%Y-%m-%dT%HZ} is initialized at or after '
                              f'{GFS_RUN_CUTOFF:%Y-%m-%dT%HZ}')
    if init.astimezone(UTC) < run_init(FLOOR):
        raise BoundaryRefused(f'GFS run {init:%Y-%m-%d} is before the 2019 input floor')
    if not isinstance(lead, int) or not LEAD_MIN <= lead <= LEAD_MAX:
        raise BoundaryRefused(f'GFS lead f{lead} is outside f{LEAD_MIN:03d}-f{LEAD_MAX:03d}')


def served_delivery_day(run_day: date) -> date:
    """A D−1 00 UTC run serves the delivery day after its run date."""
    return run_day + timedelta(days=1)


_AWS_OBJECT = re.compile(r'^/gfs\.(\d{8})/(\d{2})/(?:atmos/)?gfs\.t(\d{2})z\.pgrb2\.0p25\.f(\d{3})(\.idx)?$')
_NCAR_OBJECT = re.compile(r'^/thredds/fileServer/files/g/d084001/(\d{4})/(\d{8})/gfs\.0p25\.(\d{8})(\d{2})\.f(\d{3})\.grib2$')
_NCAR_CATALOG = re.compile(r'^/thredds/catalog/files/g/d084001/(\d{4})/(\d{8})/catalog\.xml$')
_AWS_PREFIX = re.compile(r'^gfs\.(\d{8})/(\d{2})/(?:atmos/)?gfs\.t(\d{2})z\.pgrb2\.0p25\.f(\d{3})$')
AWS_HOST, NCAR_HOST = 'noaa-gfs-bdp-pds.s3.amazonaws.com', 'tds.gdex.ucar.edu'


def _day(ymd: str) -> date:
    return date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))


def refuse_credentials(url: str) -> urllib.parse.SplitResult:
    parts = urllib.parse.urlsplit(url)
    if parts.username or parts.password or '@' in parts.netloc:
        raise CredentialRefused('URL with userinfo refused')
    for key, _ in urllib.parse.parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in TOKEN_QUERY_KEYS:
            raise CredentialRefused(f'URL with credential-like query key {key!r} refused')
    if parts.fragment:
        raise CredentialRefused('URL fragment refused')
    return parts


def admit_url(url: str) -> dict:
    """Parse a GFS request URL into its (run, lead) and admit it, or refuse. Fails closed."""
    parts = refuse_credentials(url)
    if parts.scheme != 'https':
        raise BoundaryRefused('only https GFS requests are admitted')
    query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    if parts.netloc == AWS_HOST:
        if parts.path == '/' and set(query) == {'list-type', 'prefix'} and query['list-type'] == '2':
            m = _AWS_PREFIX.match(query['prefix'])
            if not m:
                raise BoundaryRefused('AWS listing prefix is not one exact (run, lead) object stem')
            ymd, cyc, tcyc, lead = m.groups()
            kind = 'aws_listing'
        else:
            if query:
                raise BoundaryRefused('unexpected AWS query')
            m = _AWS_OBJECT.match(parts.path)
            if not m:
                raise BoundaryRefused('unrecognised AWS object path')
            ymd, cyc, tcyc, lead, idx = m.groups()
            kind = 'aws_index' if idx else 'aws_object'
        if cyc != '00' or tcyc != '00':
            raise BoundaryRefused('only the 00 UTC cycle is admitted')
        run, lead = _day(ymd), int(lead)
        admit_gfs(run_init(run), lead)
        return {'endpoint': 'aws', 'kind': kind, 'run': run, 'lead': lead}
    if parts.netloc == NCAR_HOST:
        if query:
            raise BoundaryRefused('unexpected NCAR query')
        m = _NCAR_OBJECT.match(parts.path)
        if m:
            year, ymd, init_ymd, cyc, lead = m.groups()
            if init_ymd != ymd or year != ymd[:4] or cyc != '00':
                raise BoundaryRefused('inconsistent NCAR object path')
            run, lead = _day(ymd), int(lead)
            admit_gfs(run_init(run), lead)
            return {'endpoint': 'ncar', 'kind': 'ncar_object', 'run': run, 'lead': lead}
        m = _NCAR_CATALOG.match(parts.path)
        if m:
            year, ymd = m.groups()
            run = _day(ymd)
            if year != ymd[:4]:
                raise BoundaryRefused('inconsistent NCAR catalog path')
            for lead in range(LEAD_MIN, LEAD_MAX + 1, 3):
                admit_gfs(run_init(run), lead)
            return {'endpoint': 'ncar', 'kind': 'ncar_catalog', 'run': run, 'lead': None}
        raise BoundaryRefused('unrecognised NCAR path')
    raise BoundaryRefused(f'host {parts.netloc!r} is not one of the two anonymous GFS archives')


# ---------------------------------------------------------------------------------- issue time

def issue_time(delivery: date) -> datetime:
    """D−1 11:00 Europe/Berlin as an aware UTC instant; the offset is D−1's own at 11:00 (§24.4)."""
    local = datetime.combine(delivery - timedelta(days=1), ISSUE_LOCAL, tzinfo=BERLIN)
    return local.astimezone(UTC)


def strictly_before(available: datetime, delivery: date) -> bool:
    """§24.4: usable only if published strictly before the issue time; exactly at it is not."""
    if available.tzinfo is None:
        raise ValueError('naive availability instant')
    return available.astimezone(UTC) < issue_time(delivery)


# ---------------------------------------------------------------------------------- credentials

def _values() -> dict[str, str]:
    out = {}
    for name in CREDENTIAL_VARIABLES:
        value = os.environ.get(name)
        if value and len(value) >= MIN_REDACT_LENGTH:
            out[name] = value
    return out


def redact(text: str) -> str:
    """Replace any stored credential value, and any token-like query value, before text is logged."""
    for name, value in _values().items():
        if value in text:
            text = text.replace(value, f'[REDACTED:{name}]')
    return re.sub(r'(?i)([?&](?:' + '|'.join(re.escape(k) for k in sorted(TOKEN_QUERY_KEYS)) + r')=)[^&\s]*',
                  r'\1[REDACTED]', text)


def scan(paths) -> dict[str, list[str]]:
    """Which credential *names* occur, by value, in which files. Never returns or prints a value."""
    values = _values()
    found: dict[str, list[str]] = {}
    for base in paths:
        base = Path(base)
        files = [base] if base.is_file() else [p for p in base.rglob('*') if p.is_file()]
        for path in files:
            try:
                blob = path.read_bytes()
            except OSError:
                continue
            for name, value in values.items():
                if value.encode() in blob:
                    found.setdefault(name, []).append(str(path))
    return found


def variables_set() -> dict[str, bool]:
    """Set or unset only, as AGENTS.md permits."""
    return {name: bool(os.environ.get(name)) for name in CREDENTIAL_VARIABLES}
