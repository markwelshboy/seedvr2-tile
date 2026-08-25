from __future__ import annotations

import argparse
import functools
import http.server
import sys
import webbrowser
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedvr2-sweep-serve",
        description="Serve a SeedVR2 sweep report directory over a tiny local HTTP server.",
    )
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--bind", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="TCP port; 0 chooses a free local port (default: 0)")
    parser.add_argument("--open", action="store_true", help="open the report in the default browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    root = args.report_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"report directory not found: {root}")
    if not (root / "index.html").is_file():
        raise SystemExit(f"index.html not found in report directory: {root}")
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with http.server.ThreadingHTTPServer((args.bind, args.port), handler) as server:
        host, port = server.server_address[:2]
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        url = f"http://{display_host}:{port}/"
        print(f"Serving SeedVR2 report: {root}")
        print(f"URL: {url}")
        print("Press Ctrl-C to stop.")
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
