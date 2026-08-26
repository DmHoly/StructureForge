"""`structureforge` console entry point: runs the web GUI locally."""

from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="structureforge", description="Run the StructureForge web GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("structureforge.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
