#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit a decision-complete markdown report for a measured MLX DP comparison run")
    parser.add_argument("--baseline-tag", default="cluster-mlx-d10-lr8e5-wd1e2")
    parser.add_argument("--candidate-tag", default="cluster-mlx-d12")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--metrics-root", default="build/mlx-train")
    parser.add_argument("--export-root", default="converted")
    parser.add_argument("--checkpoint-root", default=".mlx-checkpoints")
    parser.add_argument("--hosts", default="node-0.local,node-1.local")
    parser.add_argument("--close-ratio", type=float, default=1.05)
    parser.add_argument("--output", default="")
    return parser


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path, event_name: str) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
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


def healthy_hosts(hosts: list[str]) -> bool:
    for host in hosts:
        status = swap_status(host)
        if status == "unavailable":
            return False
        if "used = 0.00M" not in status and "used = 0.0M" not in status:
            return False
    return True


def parse_disk_trend(session_dir: Path) -> tuple[str, str]:
    review_dir = session_dir / "reviews"
    if not review_dir.exists():
        return ("n/a", "n/a")

    pattern = re.compile(r"disk_node(?P<node>[01]): .* (?P<avail>\d+)Gi ")
    points: list[dict[str, int]] = []
    for review in sorted(review_dir.glob("*.txt")):
        values: dict[str, int] = {}
        for line in review.read_text(encoding="utf-8").splitlines():
            match = pattern.search(line)
            if match:
                values[match.group("node")] = int(match.group("avail"))
        if "0" in values and "1" in values:
            points.append(values)

    if not points:
        return ("n/a", "n/a")

    start = points[0]
    end = points[-1]
    return (
        f"node-0 {start['0']}GiB -> {end['0']}GiB, node-1 {start['1']}GiB -> {end['1']}GiB",
        f"node-0 min {min(p['0'] for p in points)}GiB, node-1 min {min(p['1'] for p in points)}GiB",
    )


def build_report(args: argparse.Namespace) -> str:
    metrics_root = Path(args.metrics_root)
    export_root = Path(args.export_root)
    checkpoint_root = Path(args.checkpoint_root)
    session_dir = Path(args.session_dir)

    baseline_train = load_events(metrics_root / args.baseline_tag / "metrics_rank0.jsonl", "train_step")
    baseline_val = load_events(metrics_root / args.baseline_tag / "metrics_rank0.jsonl", "val_step")
    candidate_train = load_events(metrics_root / args.candidate_tag / "metrics_rank0.jsonl", "train_step")
    candidate_val = load_events(metrics_root / args.candidate_tag / "metrics_rank0.jsonl", "val_step")
    baseline_summary = load_json(metrics_root / args.baseline_tag / "summary.json")
    candidate_summary = load_json(metrics_root / args.candidate_tag / "summary.json")

    baseline_best = baseline_summary.get("best_val_loss")
    candidate_best = candidate_summary.get("best_val_loss")
    hosts = [host.strip() for host in args.hosts.split(",") if host.strip()]
    systems_healthy = healthy_hosts(hosts)
    disk_trend, disk_min = parse_disk_trend(session_dir)

    candidate_label = args.candidate_tag
    if candidate_best is None:
        recommendation = "stop dp scaling and start tp"
    elif baseline_best is not None and candidate_best < baseline_best and systems_healthy:
        recommendation = f"promote {candidate_label}"
    elif baseline_best is not None and candidate_best <= baseline_best * args.close_ratio:
        recommendation = f"rerun {candidate_label} on clean baseline"
    else:
        recommendation = "stop dp scaling and start tp"

    milestones = [1000, 2500, 4000, 8000, 12000]
    lines = [
        f"# MLX DP Comparison Report: {args.candidate_tag} vs {args.baseline_tag}",
        "",
        "## Outcome",
        "",
        f"- recommendation: `{recommendation}`",
        f"- baseline best val loss: `{baseline_best}`",
        f"- candidate best val loss: `{candidate_best}`",
        f"- candidate best val step: `{candidate_summary.get('best_val_step', 'n/a')}`",
        f"- candidate stop reason: `{candidate_summary.get('stop_reason', 'none')}`",
        f"- system health for promotion: `{'healthy' if systems_healthy else 'needs review'}`",
        "",
        "## Milestones",
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
            "## Throughput And Storage",
            "",
            f"- baseline trailing tokens/sec: `{average_tokens_per_second(baseline_train)}`",
            f"- candidate trailing tokens/sec: `{average_tokens_per_second(candidate_train)}`",
            f"- baseline checkpoint size: `{directory_size(checkpoint_root / args.baseline_tag)}`",
            f"- candidate checkpoint size: `{directory_size(checkpoint_root / args.candidate_tag)}`",
            f"- baseline export size: `{directory_size(export_root / args.baseline_tag)}`",
            f"- candidate export size: `{directory_size(export_root / args.candidate_tag)}`",
            f"- disk usage trend: `{disk_trend}`",
            f"- minimum observed free disk: `{disk_min}`",
            "",
            "## Samples",
            "",
            f"- baseline sample text: `{baseline_summary.get('last_sample_text', 'n/a')}`",
            f"- candidate sample text: `{candidate_summary.get('last_sample_text', 'n/a')}`",
            "",
            "## Notes",
            "",
            "- Comparisons remain directionally useful rather than fully apples-to-apples because the current cache was reconstructed after the earlier cache-loss incident.",
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
