"""The transport interface of the live-input clients: a request in, (status, headers, body bytes) out.

* **`Request`** names what is asked for: the URL and its query parameters (never a credential), an optional
  byte range and extra headers, the *name* of a stored credential variable the transport must add (ENTSO-E's
  token), and the request's **scope** - the delivery days of market data it may return, or the GFS object it
  addresses. `Request.admit()` is the red line: every delivery day of the scope goes through
  `cp25.guard.admit_delivery_day`, every GFS URL through `cp25.guard.admit_url` (run and lead), and every URL
  through `cp25.guard.refuse_credentials`. Retiring the guard's boundary is the only change this needs.
* **`ReplayTransport`** serves recorded (or rendered) responses from a folder. It is the only transport used in
  CP-18a. It admits each request through the guard as well, so a replay can never serve what a live request
  could not ask for.
* **`HttpTransport`** is the live transport. It admits the request through the guard *before* any socket is
  opened, adds the credential inside `send` only (`os.environ.get`, never logged, never in a record), and
  redacts every error text. Nothing in CP-18a instantiates it except the test that proves its refusal with
  networking patched to raise.

A replay folder holds `index.json` (request key -> status, headers, retrieval time, provenance, note, body
file) and the bodies, gzip-compressed (`<sha256>.gz`, mtime 0); the recorded SHA-256 and byte count are the
uncompressed body's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import urllib.parse

from cp25 import guard

UTC = timezone.utc
SCOPES = ('market', 'gfs', 'metadata')


class TransportError(RuntimeError):
    """A response could not be obtained; the text is redacted."""


class ReplayMiss(TransportError):
    """The replay folder holds no response for this request (treated as not retrieved)."""


class CredentialMissing(TransportError):
    """The named credential variable is unset in this process."""


def utc_now() -> str:
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Request:
    method: str
    url: str
    params: tuple = ()                    # sorted (key, value) pairs; never a credential
    byte_range: tuple | None = None       # inclusive (start, end)
    headers: tuple = ()                   # sorted (name, value) pairs; never a credential
    credential: str | None = None         # the NAME of a stored variable the live transport adds
    scope: tuple = ('metadata',)          # ('market', day, ...), ('gfs',) or ('metadata',)

    def __post_init__(self):
        if self.scope[0] not in SCOPES:
            raise ValueError(f'unknown request scope {self.scope[0]!r}')

    def full_url(self) -> str:
        """The URL with its query; without any credential (the live transport adds it)."""
        if not self.params:
            return self.url
        return self.url + '?' + urllib.parse.urlencode(list(self.params))

    def days(self) -> list[date]:
        return [date.fromisoformat(d) for d in self.scope[1:]] if self.scope[0] == 'market' else []

    def admit(self) -> None:
        """The red line, before anything is sent or served: cp25.guard on every day, run and URL."""
        url = self.full_url()
        guard.refuse_credentials(url)
        for name, _ in self.headers:
            if name.lower() in ('authorization', 'x-amz-security-token', 'cookie'):
                raise guard.CredentialRefused(f'credential-bearing header {name!r} refused')
        if self.scope[0] == 'market':
            if not self.scope[1:]:
                raise guard.BoundaryRefused('a market-data request must name its delivery days')
            for d in self.days():
                guard.admit_delivery_day(d)
        elif self.scope[0] == 'gfs':
            guard.admit_url(url)

    def key(self) -> str:
        """The replay key: method, URL, query, range. Headers and the credential name are not part of it."""
        value = {'method': self.method, 'url': self.url, 'params': [list(p) for p in self.params],
                 'range': list(self.byte_range) if self.byte_range else None}
        return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())

    def redacted(self) -> dict:
        """What an archive record keeps of the request: no credential value can be in it."""
        return {'method': self.method, 'url': guard.redact(self.url),
                'params': {k: guard.redact(str(v)) for k, v in self.params},
                'range': list(self.byte_range) if self.byte_range else None,
                'headers': {k: guard.redact(str(v)) for k, v in self.headers},
                'credential_variable': self.credential, 'scope': list(self.scope)}


@dataclass
class Response:
    status: int
    headers: dict
    body: bytes
    retrieved_utc: str
    provenance: str = 'live'              # 'live', 'recorded' or 'rendered'
    note: str = ''
    meta: dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return sha256(self.body)


class Transport:
    def send(self, request: Request) -> Response:      # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ replay
def _gz(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=6, mtime=0)


class ReplayFolder:
    """Writes a replay folder (recorded or rendered responses), one body file per distinct body."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / 'index.json'
        self.index = json.loads(path.read_text()) if path.exists() else {'schema': 'cp18-live-replay-v1', 'entries': {}}

    def add(self, request: Request, response: Response) -> str:
        request.admit()
        digest = response.sha256
        name = f'{digest}.gz'
        target = self.folder / name
        if not target.exists():
            target.write_bytes(_gz(response.body))
        self.index['entries'][request.key()] = {
            'request': request.redacted(), 'status': response.status, 'headers': response.headers,
            'retrieved_utc': response.retrieved_utc, 'provenance': response.provenance, 'note': response.note,
            'meta': response.meta, 'body': name, 'sha256': digest, 'bytes': len(response.body)}
        return request.key()

    def save(self) -> None:
        (self.folder / 'index.json').write_text(json.dumps(self.index, sort_keys=True, indent=0) + '\n')


class ReplayTransport(Transport):
    """Serves the responses recorded in a replay folder; a request it does not hold is a `ReplayMiss`."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.index = json.loads((self.folder / 'index.json').read_text())['entries']
        self.served: list[str] = []

    def send(self, request: Request) -> Response:
        request.admit()
        entry = self.index.get(request.key())
        if entry is None:
            raise ReplayMiss(f'no recorded response for {guard.redact(request.full_url())}'
                             + (f' range {request.byte_range}' if request.byte_range else ''))
        body = gzip.decompress((self.folder / entry['body']).read_bytes())
        if sha256(body) != entry['sha256'] or len(body) != entry['bytes']:
            raise TransportError(f'replay body differs from its recorded SHA-256: {entry["body"]}')
        self.served.append(request.key())
        return Response(entry['status'], dict(entry['headers']), body, entry['retrieved_utc'], entry['provenance'],
                        entry.get('note', ''), dict(entry.get('meta', {})))


# ------------------------------------------------------------------------------------------------ live
class HttpTransport(Transport):
    """The live transport (stdlib `urllib`). NOT used in CP-18a: only the refusal test instantiates it.

    Order inside `send`: (1) `request.admit()` - cp25.guard on the requested period, delivery days or GFS
    run, and the credential refusal on the URL - before any socket exists; (2) the credential variable, read
    with `os.environ.get` here and nowhere else, is appended as ENTSO-E's `securityToken`; (3) the request.
    Errors are re-raised with `cp25.guard.redact`ed text and without the URL."""

    USER_AGENT = 'delu-day-ahead-forecast-cp18/1 (research; CC-BY data)'

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def send(self, request: Request) -> Response:
        import urllib.error
        import urllib.request
        request.admit()
        url = request.full_url()
        if request.credential:
            value = os.environ.get(request.credential)
            if not value:
                raise CredentialMissing(f'{request.credential} is unset')
            url = url + ('&' if '?' in url else '?') + urllib.parse.urlencode({'securityToken': value})
        headers = {'User-Agent': self.USER_AGENT, **dict(request.headers)}
        if request.byte_range:
            headers['Range'] = f'bytes={request.byte_range[0]}-{request.byte_range[1]}'
        req = urllib.request.Request(url, method=request.method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:   # noqa: S310 - https only, guarded
                status, head, body = r.status, dict(r.headers.items()), r.read()
        except urllib.error.HTTPError as exc:
            status, head, body = exc.code, dict(exc.headers.items()) if exc.headers else {}, exc.read() or b''
        except Exception as exc:  # noqa: BLE001 - every transport failure is redacted and re-raised
            raise TransportError(guard.redact(f'{type(exc).__name__}: {exc}')) from None
        finally:
            url = None   # noqa: F841 - drop the credential-bearing URL at once
        return Response(status, head, body, utc_now(), 'live')
