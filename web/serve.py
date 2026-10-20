#!/usr/bin/env python3
"""
web/serve.py — any static file server works for the wasm GUI; this one
just serves the repo root (the page loads the engine from
engine/build/wasm/). The single-threaded engine needs no
SharedArrayBuffer, hence no COOP/COEP headers and no coi-serviceworker.

    .venv/bin/python web/serve.py [--port 8468]
"""
import argparse
import functools
import http.server
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8468)
    args = ap.parse_args()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(ROOT))
    print(f"wasm GUI on http://127.0.0.1:{args.port}/web/ — Ctrl-C to stop")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
