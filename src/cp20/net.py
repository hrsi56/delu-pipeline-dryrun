"""Metered read-only HTTP for the two named GFS endpoints (extraction environment).

Every request is charged to the CP-20 ledger *before* it is sent (expected bytes plus a
fixed header/request allowance) and logged afterwards. A request that would cross the
job's transfer stop line or the 160 GiB cap is refused before any byte moves. Only the
NCAR d084001 THREDDS file server and the NOAA AWS ``noaa-gfs-bdp-pds`` bucket are
reachable through this module.

Network hardening (Owner standing instruction, 2026-09-23):
* each request runs in its own reader thread and writes only into its own buffer; the
  caller waits at most the request's wall-clock deadline, so a trickling connection can
  never hold a request past it whatever the chunk size; on expiry the reader is cancelled,
  its socket shut down and its buffer discarded, so a late byte can never reach a later
  attempt (no partial-range resume);
* healthy keep-alive connections are reused; a failed or expired request is retried on a
  fresh connection after exponential backoff with jitter (capped at 10 minutes), without
  limit: network retries are not message attempts (Owner decision O2) but every retry's
  bytes are reserved and charged against the transfer cap before it is sent;
* only a stop request ends the retry loop (``NetworkError`` is raised to the caller then).
"""
from __future__ import annotations

import json
import random
import socket
import threading
import time
import urllib.parse
from pathlib import Path

import requests

from .budget import Budget, CapExceeded

ALLOWED_HOSTS = {'noaa-gfs-bdp-pds.s3.amazonaws.com': 'aws', 'tds.gdex.ucar.edu': 'ncar'}
ALLOWANCE = 4096  # response headers + request line, charged per request up front
USER_AGENT = 'PJM-cp20-gfs-extraction/1.0 (read-only research)'
# r11 (Owner, 2026-09-24): 15 s + 5 us per expected byte (was 30 s + 10 us); latency-bound link.
DEADLINE_BASE_S, DEADLINE_PER_BYTE_S = 15.0, 5e-6
CHUNK = 32768
BACKOFF_BASE_S, BACKOFF_MAX_S = 5.0, 600.0


class TransferError(RuntimeError):
    """A request failed or returned an unexpected status/size."""


class NetworkError(TransferError):
    """Connection failure, timeout, 5xx/429 or an expired deadline (retried; not an attempt)."""


class ObjectError(TransferError):
    """4xx or a wrong-size response for this object: use the alternate endpoint (counted, O2)."""


def backoff_delay(failures: int, rng=random) -> float:
    """Exponential backoff with +-50% jitter, never above 10 minutes."""
    return min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (failures - 1) * rng.uniform(0.5, 1.5))


class Fetcher:
    def __init__(self, budget: Budget, log_path: Path, stop_transfer_bytes: int, timeout=(15.0, 60.0),
                 stop_event: threading.Event | None = None, allowed_hosts: dict | None = None,
                 deadline=(DEADLINE_BASE_S, DEADLINE_PER_BYTE_S), sleep=None):
        self.budget, self.log_path, self.stop, self.timeout = budget, Path(log_path), stop_transfer_bytes, timeout
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.hosts = allowed_hosts or ALLOWED_HOSTS
        self.deadline = deadline
        self.stop_event = stop_event or threading.Event()
        self.sleep = sleep or self.stop_event.wait
        self._lock = threading.Lock()
        self._local = threading.local()
        self.stats = {'requests': 0, 'bytes': 0, 'network_retries': 0, 'deadline_expiries': 0}
        self.last_data = time.time()

    def _session(self):
        session = getattr(self._local, 'session', None)
        if session is None:
            session = requests.Session()
            session.headers['User-Agent'] = USER_AGENT
            self._local.session = session
        return session

    def _drop_session(self):
        self._local.session = None  # the old session is closed by its reader thread when it ends

    @staticmethod
    def _read(session, url, headers, timeout, expected, h):
        try:
            with session.get(url, headers=headers, timeout=timeout, stream=True) as resp:
                h['status'] = resp.status_code
                h['headers'] = {k.title(): v for k, v in resp.headers.items()}
                try:
                    h['sock'] = resp.raw._connection.sock
                except AttributeError:
                    h['sock'] = None
                for chunk in resp.raw.stream(CHUNK, decode_content=False):
                    if h['cancelled']:
                        break
                    h['buf'] += chunk
                    if len(h['buf']) > expected:
                        h['error'], h['kind'] = f'body exceeded bound {expected}', 'object'
                        break
        except Exception as exc:  # noqa: BLE001 - classified below
            if not h['cancelled']:
                h['error'], h['kind'] = f'{type(exc).__name__}: {exc}'[:300], 'network'
        finally:
            if h['cancelled']:
                session.close()
            h['done'].set()

    def _once(self, url, purpose, byte_range, expected, reserved, try_no):
        headers = {'Range': f'bytes={byte_range[0]}-{byte_range[1]}'} if byte_range else {}
        h = {'buf': bytearray(), 'done': threading.Event(), 'status': 0, 'headers': {}, 'error': None,
             'kind': None, 'cancelled': False, 'sock': None}
        session = self._session()
        began = time.time()
        wall = self.deadline[0] + self.deadline[1] * expected
        reader = threading.Thread(target=self._read, args=(session, url, headers, self.timeout, expected, h), daemon=True)
        reader.start()
        if not h['done'].wait(wall):  # watchdog: the caller never waits past the deadline
            h['cancelled'] = True
            sock = h.get('sock')
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            error, kind, body = f'deadline {wall:.0f} s expired with {len(h["buf"])} of {expected} bytes', 'network', b''
            with self._lock:
                self.stats['deadline_expiries'] += 1
        else:
            error, kind, body = h['error'], h['kind'], bytes(h['buf'])  # this attempt's own buffer only
        status, resp_headers = h['status'], dict(h['headers'])
        wire = len(body) + sum(len(k) + len(str(v)) + 4 for k, v in resp_headers.items()) + 300 + len(url)
        if wire > reserved:  # never under-record bytes that already moved
            try:
                self.budget.reserve(transfer_bytes=wire - reserved)
            except CapExceeded:
                self.budget.event('transfer_overage_beyond_cap', url=url, bytes=wire - reserved)
                raise
        if error is None and status >= 400:
            error, kind = f'HTTP {status}', ('network' if status >= 500 or status == 429 else 'object')
        if error is None and byte_range is not None and (status != 206 or len(body) != expected):
            error, kind = f'range mismatch status={status} got={len(body)} want={expected}', 'object'
        if error is not None:
            self._drop_session()
            if not h['cancelled']:
                session.close()
        record = {'t_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(began)), 'error_kind': kind,
                  'elapsed_s': round(time.time() - began, 3), 'endpoint': self.hosts[urllib.parse.urlsplit(url).netloc],
                  'url': urllib.parse.urlunsplit(urllib.parse.urlsplit(url)._replace(query='', fragment='')),
                  'range': list(byte_range) if byte_range else None, 'status': status, 'body_bytes': len(body),
                  'wire_bytes_estimate': wire, 'charged_bytes': max(wire, reserved), 'purpose': purpose,
                  'error': error, 'request_try': try_no}
        with self._lock:
            self.stats['requests'] += 1
            self.stats['bytes'] += max(wire, reserved)
            with self.log_path.open('a') as fh:
                fh.write(json.dumps(record, sort_keys=True) + '\n')
        if error is not None:
            self.budget.reserve(failed_requests=1)
            raise (NetworkError if kind == 'network' else ObjectError)(f'{purpose}: {error}')
        self.last_data = time.time()
        return status, resp_headers, body

    def get(self, url, purpose, *, byte_range=None, max_body=None):
        host = urllib.parse.urlsplit(url).netloc
        if host not in self.hosts:
            raise ValueError(f'endpoint not authorised for CP-20: {host}')
        if byte_range is not None:
            start, end = byte_range
            if start < 0 or end < start:
                raise ValueError('invalid byte range')
            expected = end - start + 1
        elif max_body is not None:
            expected = max_body
        else:
            raise ValueError('every CP-20 request must bound its body size')
        reserved = expected + ALLOWANCE
        failures = 0
        while True:
            # Refused here, before sending, if it would cross the stop line or the hard cap.
            self.budget.reserve(limits={'transfer_bytes': self.stop}, transfer_bytes=reserved, requests=1)
            try:
                return self._once(url, purpose, byte_range, expected, reserved, failures + 1)
            except NetworkError:
                failures += 1
                with self._lock:
                    self.stats['network_retries'] += 1
                self.budget.reserve(network_retries=1)
                if self.sleep(backoff_delay(failures)) or self.stop_event.is_set():
                    raise  # only a stop request ends network retrying
