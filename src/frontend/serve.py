"""Serve the telemetry dashboard next to live run artifacts. Zero dependencies.

Usage:
    python -m src.frontend.serve                       # golden artifacts, port 7777
    python -m src.frontend.serve --metrics /tmp/verdict.metrics.json \
        --ledger /tmp/verdict.ledger.jsonl --port 8000

Routes: /             -> the dashboard (src/frontend/index.html)
        /metrics.json -> the --metrics file (404 until the run writes it)
        /ledger.jsonl -> the --ledger file
        POST /solve   -> playground: run {"prompt": ...} through the real pipeline
The page polls every 2s while the run status is "running", so starting this
before (or during) a run gives a live view. The playground initializes the
pipeline (and on a GPU box, llama-server) on its first request. Never bundle
into the Docker image — the harness only wants results.json.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.core import settings
from src.frontend.playground import Playground, PlaygroundError

_INDEX = Path(__file__).with_name("index.html")
_GOLDEN_DIR = settings.REPO_ROOT / "outputs" / "golden"
_DEFAULT_PORT = 7777
_MAX_PROMPT_BYTES = 64 * 1024

_playground = Playground()


class _Handler(BaseHTTPRequestHandler):
    metrics_path: Path
    ledger_path: Path

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            self._send(_INDEX, "text/html; charset=utf-8")
        elif route == "/metrics.json":
            self._send(self.metrics_path, "application/json")
        elif route == "/ledger.jsonl":
            self._send(self.ledger_path, "application/x-ndjson")
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if self.path.split("?", 1)[0] != "/solve":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= _MAX_PROMPT_BYTES:
            self._send_json(400, {"error": "prompt missing or too large"})
            return
        try:
            body = json.loads(self.rfile.read(length))
            prompt = str(body["prompt"])
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            self._send_json(400, {"error": "body must be JSON: {\"prompt\": ...}"})
            return
        try:
            self._send_json(200, _playground.solve(prompt))
        except PlaygroundError as e:
            self._send_json(422, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface the cause to the UI
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(404, f"{path.name} not written yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # keep the terminal quiet; the dashboard is the output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=_GOLDEN_DIR / "run.metrics.json")
    parser.add_argument("--ledger", type=Path, default=_GOLDEN_DIR / "run.ledger.jsonl")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _Handler.metrics_path = args.metrics
    _Handler.ledger_path = args.ledger
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"dashboard: http://{args.host}:{args.port}/")
    print(f"  metrics: {args.metrics}")
    print(f"  ledger:  {args.ledger}")
    print("  playground: POST /solve (pipeline starts on first ask)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
