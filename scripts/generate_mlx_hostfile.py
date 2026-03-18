#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a simple MLX ring hostfile")
    parser.add_argument("--host", action="append", dest="hosts", required=True, help="Entry formatted as ssh_host=ip")
    parser.add_argument("--output", required=True, help="Output JSON file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    entries = []
    for item in args.hosts:
        if "=" not in item:
            raise SystemExit(f"Invalid host entry {item!r}; expected ssh_host=ip")
        ssh_host, ip = item.split("=", 1)
        entries.append({"ssh": ssh_host, "ips": [ip]})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
