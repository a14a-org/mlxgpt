#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two MLX training runs and emit a markdown summary")
    parser.add_argument("--baseline-tag", required=True)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--metrics-root", default="build/mlx-train")
    parser.add_argument("--export-root", default="converted")
    parser.add_argument("--checkpoint-root", default=".mlx-checkpoints")
    parser.add_argument("--hosts", default="node-0.local,node-1.local")
    parser.add_argument("--output", default="")
    return parser


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path, event_name: str) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("event") == event_name:
            events.append(item)
    return events


def nearest_loss(events: list[dict], step: int) -> str:
    for item in events:
        if item.get("step") == step:
            return f"{item.get('loss')}"
    return "n/a"


def average_tokens_per_second(events: list[dict], trailing_steps: int = 1000) -> str:
    if not events:
        return "n/a"
    tail = events[-trailing_steps:]
    values = [float(item["tokens_per_second"]) for item in tail if item.get("tokens_per_second") is not None]
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.2f}"


def directory_size(path: Path) -> str:
    if not path.exists():
        return "n/a"
    total = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    gib = total / (1024 ** 3)
    return f"{gib:.2f} GiB"


def swap_status(host: str) -> str:
    try:
        output = subprocess.check_output(["ssh", host, "sysctl", "vm.swapusage"], text=True).strip()
    except Exception:
        return "unavailable"
    return output


def build_report(args: argparse.Namespace) -> str:
    metrics_root = Path(args.metrics_root)
    export_root = Path(args.export_root)
    checkpoint_root = Path(args.checkpoint_root)

    baseline_train = load_events(metrics_root / args.baseline_tag / "metrics_rank0.jsonl", "train_step")
    baseline_val = load_events(metrics_root / args.baseline_tag / "metrics_rank0.jsonl", "val_step")
    candidate_train = load_events(metrics_root / args.candidate_tag / "metrics_rank0.jsonl", "train_step")
    candidate_val = load_events(metrics_root / args.candidate_tag / "metrics_rank0.jsonl", "val_step")
    baseline_summary = load_json(metrics_root / args.baseline_tag / "summary.json")
    candidate_summary = load_json(metrics_root / args.candidate_tag / "summary.json")

    milestones = [1000, 2500, 4000, 8000, 12000]
    hosts = [host.strip() for host in args.hosts.split(",") if host.strip()]
    swap_lines = [f"- {host}: `{swap_status(host)}`" for host in hosts]

    lines = [
        f"# MLX Comparison: {args.candidate_tag} vs {args.baseline_tag}",
        "",
        "## Loss Milestones",
        "",
        "| step | baseline train | candidate train | baseline val | candidate val |",
        "| --- | --- | --- | --- | --- |",
    ]
    for step in milestones:
        lines.append(
            f"| {step} | {nearest_loss(baseline_train, step)} | {nearest_loss(candidate_train, step)} | "
            f"{nearest_loss(baseline_val, step)} | {nearest_loss(candidate_val, step)} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- baseline best val loss: `{baseline_summary.get('best_val_loss', 'n/a')}`",
            f"- candidate best val loss: `{candidate_summary.get('best_val_loss', 'n/a')}`",
            f"- baseline final sample: `{baseline_summary.get('last_sample_text', 'n/a')}`",
            f"- candidate final sample: `{candidate_summary.get('last_sample_text', 'n/a')}`",
            f"- baseline avg tokens/sec over last 1000 train steps: `{average_tokens_per_second(baseline_train)}`",
            f"- candidate avg tokens/sec over last 1000 train steps: `{average_tokens_per_second(candidate_train)}`",
            f"- baseline checkpoint size: `{directory_size(checkpoint_root / args.baseline_tag)}`",
            f"- candidate checkpoint size: `{directory_size(checkpoint_root / args.candidate_tag)}`",
            f"- baseline export size: `{directory_size(export_root / args.baseline_tag)}`",
            f"- candidate export size: `{directory_size(export_root / args.candidate_tag)}`",
            "",
            "## System Fit",
            "",
            *swap_lines,
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(args)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
