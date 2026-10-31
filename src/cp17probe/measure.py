"""Drive Chromium and WebKit through the probe page and record what each browser measured (§19.3 item 4).

Runs under the Playwright tool environment (standard library plus `playwright`; no NumPy). It serves the
pinned Pyodide release, the page, the source bundle and the payloads from one 127.0.0.1 server that
counts the bytes of every response, opens each run in a fresh browser context (an empty cache, as on a
first visit), and samples the resident memory of the browser's processes while the fit runs. The
comparison with the native reference is `cp17probe.compare`, in the project environment.

    PLAYWRIGHT_BROWSERS_PATH=<.local/tools/ms-playwright> <playwright python> src/cp17probe/measure.py \\
        --pyodide <.local/tools/pyodide-0.28.0> --bundle <bundle folder> --payloads <payload folder> \\
        --out <browser-results.json> [--browsers chromium,webkit] [--runs A1-2026-09-06,...]
"""
from __future__ import annotations

import argparse
from functools import partial
import http.server
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PACKAGES = {'A1': 'numpy,pandas,scikit-learn,requests,tzdata',
            'v2': 'numpy,pandas,scikit-learn,requests,tzdata,lightgbm'}
RUNS = ('A1-2026-09-06', 'A1-2026-03-29', 'v2-2026-09-06')
TYPES = {'.wasm': 'application/wasm', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
         '.html': 'text/html; charset=utf-8', '.zip': 'application/zip', '.whl': 'application/zip',
         '.npz': 'application/octet-stream'}


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, routes: dict[str, Path]):
        self.routes = {k: Path(v).resolve() for k, v in routes.items()}
        self.served: list[dict] = []
        self.lock = threading.Lock()
        super().__init__(('127.0.0.1', 0), Handler)

    def resolve(self, path: str) -> Path | None:
        path = path.split('?', 1)[0]
        for prefix, folder in self.routes.items():
            if path.startswith(prefix):
                target = (folder / path[len(prefix):]).resolve()
                return target if target.is_file() and target.is_relative_to(folder) else None
        return None


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        target = self.server.resolve(self.path)
        if target is None:
            self.send_error(404)
            with self.server.lock:
                self.server.served.append({'path': self.path, 'status': 404, 'bytes': 0})
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', TYPES.get(target.suffix, 'application/octet-stream'))
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)
        with self.server.lock:
            self.server.served.append({'path': self.path.split('?', 1)[0], 'status': 200, 'bytes': len(body)})

    def log_message(self, *args):
        pass


def category(path: str) -> str:
    if path == '/app/driver.py':
        return 'code'
    if path.startswith('/app/'):
        return 'page'
    if path.startswith('/pyodide/'):
        name = path.rsplit('/', 1)[1]
        return 'packages' if name.endswith(('.whl', '.zip')) and name != 'python_stdlib.zip' else 'runtime'
    if path.startswith('/bundle/'):
        return 'code'
    if path.startswith('/payloads/'):
        return 'data'
    return 'other'


class MemorySampler(threading.Thread):
    """The summed resident set of every process whose command lies inside the browser's install folder."""

    def __init__(self, marker: str, interval: float = 0.25):
        super().__init__(daemon=True)
        self.marker, self.interval = marker, interval
        self.peak_kib, self.samples, self.stop = 0, 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            out = subprocess.run(['ps', '-axo', 'rss=,command='], capture_output=True, text=True).stdout
            total = sum(int(line.split(None, 1)[0]) for line in out.splitlines()
                        if self.marker in line and line.strip() and line.split(None, 1)[0].isdigit())
            self.peak_kib, self.samples = max(self.peak_kib, total), self.samples + 1
            self.stop.wait(self.interval)


def one_run(playwright, kind: str, base: str, stem: str, server: Server, timeout_s: float) -> dict:
    browser_type = getattr(playwright, kind)
    browser = browser_type.launch(channel='chromium', headless=True) if kind == 'chromium' \
        else browser_type.launch(headless=True)
    marker = 'chromium-1243' if kind == 'chromium' else 'webkit-2359'
    sampler = MemorySampler(marker)
    try:
        context = browser.new_context()
        blocked: list[str] = []

        def local_only(route):
            if route.request.url.startswith(base + '/'):
                route.continue_()
            else:
                blocked.append(route.request.url[:200])
                route.abort()
        context.route('**/*', local_only)
        page = context.new_page()
        console: list[str] = []
        page.on('console', lambda msg: console.append(f'{msg.type}: {msg.text}'[:500]))
        policy = stem.split('-', 1)[0]
        with server.lock:
            server.served.clear()
        sampler.start()
        started = time.perf_counter()
        page.goto(f'{base}/app/index.html?policy={policy}&stem={stem}&packages={PACKAGES[policy]}')
        page.wait_for_function('window.__probe !== null', timeout=timeout_s * 1000, polling=1000)
        wall = time.perf_counter() - started
        out = page.evaluate('window.__probe')
        sampler.stop.set()
        sampler.join()
        with server.lock:
            served = list(server.served)
        by = {}
        for item in served:
            by.setdefault(category(item['path']), 0)
            by[category(item['path'])] += item['bytes']
        out.update({'browser': kind, 'browser_version': browser.version, 'stem': stem, 'driver_wall_seconds': wall,
                    'process_rss_peak_bytes': sampler.peak_kib * 1024, 'rss_samples': sampler.samples,
                    'served': served, 'served_bytes_by_category': by, 'console_tail': console[-20:],
                    'not_found': [s['path'] for s in served if s['status'] != 200],
                    'blocked_non_local_requests': blocked})
        context.close()
        return out
    finally:
        sampler.stop.set()
        browser.close()


def main(argv=None) -> int:
    from playwright.sync_api import sync_playwright
    from importlib.metadata import version
    ap = argparse.ArgumentParser()
    ap.add_argument('--pyodide', required=True, type=Path)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--payloads', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--browsers', default='chromium,webkit')
    ap.add_argument('--runs', default=','.join(RUNS))
    ap.add_argument('--timeout', default=2400, type=float)
    args = ap.parse_args(argv)
    server = Server({'/app/': HERE / 'browser', '/pyodide/': args.pyodide, '/bundle/': args.bundle,
                     '/payloads/': args.payloads})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    report = {'schema': 'cp17-probe-browser-results-v1', 'playwright': version('playwright'),
              'machine': platform.machine(), 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
              'browsers_path': os.environ.get('PLAYWRIGHT_BROWSERS_PATH'), 'runs': []}
    with sync_playwright() as p:
        for kind in args.browsers.split(','):
            for stem in args.runs.split(','):
                started = time.time()
                try:
                    result = one_run(p, kind, base, stem, server, args.timeout)
                except Exception as exc:   # recorded: an infeasible run is a result, not a crash
                    result = {'browser': kind, 'stem': stem, 'errors': [f'{type(exc).__name__}: {exc}'[:2000]]}
                result['started_utc'] = started
                report['runs'].append(result)
                ok = not result.get('errors')
                print(kind, stem, 'ok' if ok else 'FAILED', f"{result.get('driver_wall_seconds', 0):.1f}s",
                      result.get('errors', [''])[0][:300] if not ok else '', flush=True)
                args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    server.shutdown()
    return 0 if all(not r.get('errors') for r in report['runs']) else 1


if __name__ == '__main__':
    sys.exit(main())
