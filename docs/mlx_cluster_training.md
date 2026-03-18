# MLX Cluster Training

> **Customize for your setup:** Replace `<node-0>` and `<node-1>` with your actual
> hostnames. The shell scripts accept `NODE0`, `NODE1`, and other cluster
> parameters as environment variables -- see `scripts/cluster_write_hostfiles.sh`
> for the full list.

This runbook covers the new MLX-native training path that makes both Macs participate in real training.

Current supported runtime modes:

- `dp`: data parallel across both nodes for faster training
- `tp`: tensor parallel across both nodes for larger models

With the current 2-node cluster, these are separate modes. The code is structured with `dp_size` and `tp_size` so it can be extended later, but there is no simultaneous DP+TP runtime on this 2-device setup yet.

## Artifacts

- Native MLX training checkpoints:
  - `~/<workspace>/.mlx-checkpoints/<tag>/step_<step>/`
- Exported inference-ready models:
  - `~/<workspace>/converted/<tag>/`
- Run metrics and summaries:
  - `~/<workspace>/build/mlx-train/<tag>/`

Each native checkpoint contains:

- `weights_rank<N>.safetensors`
- `optimizer_rank<N>.npz`
- `state.json`
- `model_config.json`
- `train_config.json`
- `parallel_config.json`

## Local Synthetic Smoke

This is the easiest place to start because it does not require the parquet dataset cache:

```bash
. .venv/bin/activate
python scripts/train_mlx_cluster.py \
  --parallelism dp \
  --backend any \
  --model-tag smoke-dp \
  --depth 2 \
  --max-seq-len 32 \
  --device-batch-size 2 \
  --total-batch-size 64 \
  --num-iterations 2 \
  --checkpoint-every 1 \
  --data-mode synthetic \
  --export-final
```

Tensor-parallel synthetic smoke:

```bash
. .venv/bin/activate
python scripts/train_mlx_cluster.py \
  --parallelism tp \
  --backend any \
  --model-tag smoke-tp \
  --depth 2 \
  --max-seq-len 32 \
  --device-batch-size 2 \
  --total-batch-size 64 \
  --num-iterations 1 \
  --checkpoint-every 1 \
  --data-mode synthetic \
  --export-final
```

## Real Dataset Training

Install the parquet dependency first:

```bash
. .venv/bin/activate
pip install ".[train]"
```

By default the trainer expects:

- tokenizer: `.nanochat-cache/tokenizer`
- dataset: `.nanochat-cache/base_data_climbmix`

Single-node DP-style training on one host:

```bash
. .venv/bin/activate
python scripts/train_mlx_cluster.py \
  --parallelism dp \
  --backend any \
  --model-tag mlx-d8 \
  --depth 8 \
  --max-seq-len 512 \
  --device-batch-size 2 \
  --total-batch-size 2048 \
  --num-iterations 200 \
  --checkpoint-every 50 \
  --export-final
```

Resume from the latest native checkpoint:

```bash
. .venv/bin/activate
python scripts/train_mlx_cluster.py \
  --parallelism dp \
  --backend any \
  --model-tag mlx-d8 \
  --depth 8 \
  --max-seq-len 512 \
  --device-batch-size 2 \
  --total-batch-size 2048 \
  --num-iterations 400 \
  --checkpoint-every 50 \
  --resume latest
```

## 2-Node Cluster Launch

Run these from `<node-0>` after `./scripts/check_jaccl_ready.sh` reports `READY`.

Before the first real-data run, install the parquet dependency and mirror the tokenizer/data cache to `<node-1>`:

```bash
. .venv/bin/activate
pip install ".[train]"
./scripts/cluster_sync_training_inputs.sh
```

Data-parallel training:

```bash
./scripts/cluster_run_train.sh jaccl dp \
  --model-tag cluster-mlx-d8 \
  --depth 8 \
  --max-seq-len 512 \
  --device-batch-size 2 \
  --total-batch-size 2048 \
  --num-iterations 200 \
  --checkpoint-every 50 \
  --export-final
```

Tensor-parallel training:

```bash
./scripts/cluster_run_train.sh jaccl tp \
  --model-tag cluster-mlx-tp-d10 \
  --depth 10 \
  --max-seq-len 512 \
  --device-batch-size 2 \
  --total-batch-size 1024 \
  --num-iterations 200 \
  --checkpoint-every 50 \
  --export-final
```

## Unattended DP Session

The default unattended cluster path is now the 24-hour `d10` optimization sweep in:

- `scripts/cluster_mlx_manifest.tsv`

It is designed to:

1. run a synthetic JACCL preflight smoke
2. sweep multiple `d10` learning-rate candidates
3. promote only the strongest recipes into regularization and longer-run phases
4. early-stop runs that are clearly degrading or non-competitive versus the current champion
5. export and inference-validate only new champions
6. leave a leaderboard-style session report with a single recommended next move

Launch it from `<node-0>`:

```bash
./scripts/cluster_unattended_mlx.sh
```

Dry-run the orchestration:

```bash
./scripts/cluster_unattended_mlx.sh --dry-run --stop-after-stage dp-synthetic-smoke
```

The unattended driver:

- runs preflight checks for JACCL, `pyarrow`, disk headroom, and local dataset presence
- tracks the latest train, validation, and sample events in timed review logs
- records timed reviews every 20 minutes
- retries each stage at most once using the manifest retry mode
- stops early on validation drift using the trainer's early-stop settings
- tracks a session champion seeded from `cluster-mlx-d10`
- skips promoted phases automatically unless the earlier phase result qualifies
- stops immediately on JACCL drift, metric stalls, missing checkpoints, low disk, missing validation progress, or repeated validation failure
- writes one session directory under `build/cluster/sessions/`

Useful options:

```bash
./scripts/cluster_unattended_mlx.sh \
  --session-budget-hours 24 \
  --champion-tag cluster-mlx-d10 \
  --champion-best-val 5.181989669799805 \
  --retain-final-checkpoint yes \
  --retain-best-checkpoint yes \
  --retain-latest-checkpoint yes \
  --cleanup-nonchampions yes \
  --max-failures 2
```

Checkpoint retention defaults now keep:

- the latest checkpoint
- the final checkpoint for the stage
- the best-validation checkpoint when that exact checkpoint exists

and delete intermediate checkpoints after stage completion so the cluster can continue without repeated manual pruning.

## D12 Proving Run

The historical `d10 -> d12` proving run is encoded in:

- `scripts/cluster_mlx_manifest_d12.tsv`
- `scripts/cluster_run_d12_phase.sh`
- `scripts/report_d12_phase.py`
- `docs/d10_baseline_decision.md`

The completed `d10 -> d12` phase:

1. treats `cluster-mlx-d10-lr8e5-wd1e2` as the locked `d10` baseline
2. runs one measured `cluster-mlx-d12` proving experiment
3. emitted a final markdown decision report with one of:
   - `promote d12`
   - `rerun d12 on clean baseline`
   - `stop dp scaling and start tp`

That historical proving run can still be replayed from `<node-0>`:

```bash
bash ./scripts/cluster_run_d12_phase.sh
```

After `cluster-mlx-d12` wins, the longer confirmation run is encoded in:

- `scripts/cluster_mlx_manifest_d12_long.tsv`
- `scripts/cluster_run_d12_long_phase.sh`

Launch it from `<node-0>`:

```bash
bash ./scripts/cluster_run_d12_long_phase.sh
```

That longer confirmation run is now complete and has established the current best MLX `dp` baseline:

- model tag: `cluster-mlx-d12-long`
- best val loss: `3.7967777252197266`
- best val step: `25500`
- stop step: `28000`

See [d12_long_baseline_decision.md](./d12_long_baseline_decision.md) for the locked baseline and next-step recommendation.

The trainer itself now supports early-stop and champion-aware tuning inputs:

```bash
python scripts/train_mlx_cluster.py \
  --early-stop-min-step 2500 \
  --early-stop-patience-evals 6 \
  --early-stop-degrade-ratio 1.08 \
  --early-stop-vs-champion-ratio 1.12 \
  --champion-best-val 5.181989669799805
```

## Monitoring

Every rank writes JSONL metrics into:

- `build/mlx-train/<tag>/metrics_rank0.jsonl`
- `build/mlx-train/<tag>/metrics_rank1.jsonl`

Rank 0 also writes:

- `build/mlx-train/<tag>/config.json`
- `build/mlx-train/<tag>/result.json`
- `build/mlx-train/<tag>/summary.json` with final train/val/sample summary

Useful live commands:

```bash
tail -f build/mlx-train/<tag>/metrics_rank0.jsonl
tail -f build/mlx-train/<tag>/metrics_rank1.jsonl
cat build/mlx-train/<tag>/result.json
tail -f build/cluster/sessions/<session>/session.log
```

## Notes

- `dp` mode shards data across ranks and averages gradients before the optimizer step.
- `tp` mode shards embeddings, attention projections, MLP projections, and the LM head across ranks.
- For `tp` training, logits are gathered across the vocab dimension for v1 loss computation.
- `tp` export gathers the sharded weights back into the standard `NanoChatMLX` safetensors layout so the existing inference scripts continue to work.
