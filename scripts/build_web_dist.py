#!/usr/bin/env python3
"""build_web_dist.py — assemble the static web GUI in dist/.

The page's relative paths are preserved, so the existing app and worker
need no path rewriting:

    dist/index.html
    dist/app/*.js
    dist/engine/build/wasm/chess_engine.{js,wasm}
    dist/vendor/ort-*.{mjs,wasm}, ort*.min.mjs

The net is fetched from Hugging Face at runtime; the site needs no server.
Serve it with `python3 -m http.server -d dist` or publish `dist/` to any
static host (GitHub Pages workflow: .github/workflows/pages.yml).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"build_web_dist: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist",
                    help="output directory (default: dist)")
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    if out == ROOT or out == Path(out.anchor):
        fail("refusing to write over the repo root or filesystem root")

    web = ROOT / "web"
    wasm = ROOT / "engine" / "build" / "wasm"
    vendor = web / "vendor"
    required = [
        web / "index.html",
        web / "app" / "main.js",
        wasm / "chess_engine.js",
        wasm / "chess_engine.wasm",
        vendor / "ort.all.min.mjs",
        vendor / "ort-wasm-simd-threaded.jsep.mjs",
        vendor / "ort-wasm-simd-threaded.jsep.wasm",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        print("build_web_dist: missing:", ", ".join(missing), file=sys.stderr)
        fail("build the wasm engine (`web/build.sh`) and fetch the ORT "
             "runtime (`sh web/get_vendor.sh`) first")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy2(web / "index.html", out / "index.html")
    shutil.copytree(web / "app", out / "app")

    wasm_out = out / "engine" / "build" / "wasm"
    wasm_out.mkdir(parents=True)
    for name in ("chess_engine.js", "chess_engine.wasm"):
        shutil.copy2(wasm / name, wasm_out / name)

    shutil.copytree(vendor, out / "vendor")
    (out / ".nojekyll").write_text("")

    files = [p for p in out.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"dist: {out} ({len(files)} files, {total / 1e6:.1f} MB)")
    print(f"serve: python3 -m http.server -d {out} 8000")


if __name__ == "__main__":
    main()
