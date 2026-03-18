from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import json

from nanochat_mlx.io import load_model


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "train_mlx_cluster.py"


class TrainMLXClusterTests(unittest.TestCase):
    def run_train(self, *args: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, str(SCRIPT_PATH), *args]
        return subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, check=True)

    def test_dp_checkpoint_resume_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            checkpoint_root = base / "checkpoints"
            export_root = base / "converted"
            run_dir = base / "run"
            initial = self.run_train(
                "--parallelism",
                "dp",
                "--backend",
                "any",
                "--model-tag",
                "unit-dp",
                "--depth",
                "2",
                "--max-seq-len",
                "32",
                "--device-batch-size",
                "2",
                "--total-batch-size",
                "64",
                "--num-iterations",
                "2",
                "--checkpoint-every",
                "1",
                "--aspect-ratio",
                "16",
                "--head-dim",
                "16",
                "--eval-every",
                "1",
                "--eval-batches",
                "2",
                "--sample-every",
                "1",
                "--sample-max-tokens",
                "8",
                "--data-mode",
                "synthetic",
                "--fallback-tokenizer",
                "gpt2",
                "--checkpoint-root",
                str(checkpoint_root),
                "--export-root",
                str(export_root),
                "--run-dir",
                str(run_dir),
                "--export-final",
            )
            self.assertIn('"event": "train_complete"', initial.stdout)
            self.assertTrue((checkpoint_root / "unit-dp" / "step_000002" / "weights_rank0.safetensors").exists())
            self.assertTrue((export_root / "unit-dp" / "weights.safetensors").exists())
            self.assertTrue((run_dir / "result.json").exists())
            metrics = [json.loads(line) for line in (run_dir / "metrics_rank0.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(item["event"] == "val_step" for item in metrics))
            self.assertTrue(any(item["event"] == "sample" for item in metrics))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("best_val_loss", summary)
            self.assertIn("last_sample_text", summary)

            resumed = self.run_train(
                "--parallelism",
                "dp",
                "--backend",
                "any",
                "--model-tag",
                "unit-dp",
                "--depth",
                "2",
                "--max-seq-len",
                "32",
                "--device-batch-size",
                "2",
                "--total-batch-size",
                "64",
                "--num-iterations",
                "3",
                "--checkpoint-every",
                "1",
                "--aspect-ratio",
                "16",
                "--head-dim",
                "16",
                "--eval-every",
                "1",
                "--eval-batches",
                "2",
                "--sample-every",
                "1",
                "--sample-max-tokens",
                "8",
                "--data-mode",
                "synthetic",
                "--fallback-tokenizer",
                "gpt2",
                "--checkpoint-root",
                str(checkpoint_root),
                "--export-root",
                str(export_root),
                "--run-dir",
                str(run_dir),
                "--resume",
                "latest",
            )
            self.assertIn('"step": 3', resumed.stdout)
            self.assertTrue((checkpoint_root / "unit-dp" / "step_000003" / "weights_rank0.safetensors").exists())
            resumed_metrics = [json.loads(line) for line in (run_dir / "metrics_rank0.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(item.get("event") == "val_step" and item.get("step") == 3 for item in resumed_metrics))

    def test_tp_export_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            checkpoint_root = base / "checkpoints"
            export_root = base / "converted"
            run_dir = base / "run"
            self.run_train(
                "--parallelism",
                "tp",
                "--backend",
                "any",
                "--model-tag",
                "unit-tp",
                "--depth",
                "2",
                "--max-seq-len",
                "32",
                "--device-batch-size",
                "2",
                "--total-batch-size",
                "64",
                "--num-iterations",
                "1",
                "--checkpoint-every",
                "1",
                "--aspect-ratio",
                "16",
                "--head-dim",
                "16",
                "--data-mode",
                "synthetic",
                "--fallback-tokenizer",
                "gpt2",
                "--checkpoint-root",
                str(checkpoint_root),
                "--export-root",
                str(export_root),
                "--run-dir",
                str(run_dir),
                "--export-final",
            )
            model = load_model(export_root / "unit-tp")
            self.assertEqual(model.config.n_layer, 2)
            self.assertEqual(model.config.sequence_len, 32)

    def test_dp_early_stop_writes_summary_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            checkpoint_root = base / "checkpoints"
            export_root = base / "converted"
            run_dir = base / "run"
            result = self.run_train(
                "--parallelism",
                "dp",
                "--backend",
                "any",
                "--model-tag",
                "unit-early-stop",
                "--depth",
                "2",
                "--max-seq-len",
                "32",
                "--device-batch-size",
                "2",
                "--total-batch-size",
                "64",
                "--num-iterations",
                "20",
                "--checkpoint-every",
                "20",
                "--aspect-ratio",
                "16",
                "--head-dim",
                "16",
                "--eval-every",
                "1",
                "--eval-batches",
                "2",
                "--sample-every",
                "0",
                "--data-mode",
                "synthetic",
                "--fallback-tokenizer",
                "gpt2",
                "--checkpoint-root",
                str(checkpoint_root),
                "--export-root",
                str(export_root),
                "--run-dir",
                str(run_dir),
                "--early-stop-min-step",
                "2",
                "--early-stop-patience-evals",
                "1",
                "--early-stop-degrade-ratio",
                "1.0",
                "--early-stop-vs-champion-ratio",
                "1.0",
                "--champion-best-val",
                "0.000001",
            )
            self.assertIn('"event": "train_complete"', result.stdout)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["stopped_early"])
            self.assertTrue(summary["stop_reason"])
            stopped_step = int(summary["step"])
            self.assertLess(stopped_step, 20)
            self.assertTrue((checkpoint_root / "unit-early-stop" / f"step_{stopped_step:06d}" / "weights_rank0.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
