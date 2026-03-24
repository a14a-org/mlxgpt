# mlxgpt

Training GPT-2 scale language models on a 2-node Mac Mini cluster using Apple's [MLX](https://github.com/ml-explore/mlx) framework.

This is an open experiment documenting our path from zero to a working GPT-2 level model, trained entirely on Apple Silicon with Thunderbolt RDMA interconnect. Follow along at [mlxgpt.com](https://mlxgpt.com).

## What this is

- A native MLX implementation of the [nanochat](https://github.com/karpathy/nanochat) transformer architecture
- Distributed training across 2 Mac Minis connected via Thunderbolt (JACCL/RDMA)
- Both Data Parallel (DP) and Tensor Parallel (TP) training modes
- Pure Python + MLX — no PyTorch required for training

## Current status

| Run | Depth | Val Loss | Steps | Status |
|-----|-------|----------|-------|--------|
| d10 | 10 | 4.904 | 7,000 | baseline |
| d12 | 12 | 4.530 | — | proved |
| d12-long | 12 | 3.797 | 28,000 | previous best |
| d14 | 14 | **3.506** | 29,750 | current best (f32) |
| d14-long bf16 | 14 | 5.012 | 20,000 | 4,530 tok/s, plateaued early |
| d14-long mixed | 14 | 3.634 | 21,800 | 3,830 tok/s, diverged late |
| d14-long opt f32 | 14 | 3.634 | 21,800 | 3,000 tok/s, same ceiling |
| d14 opt (seq512) | 14 | 4.170 | 14,750 | early stopped (champion gate) |
| d14 orig (seq512) | 14 | 4.170 | 14,750 | same — code changes are innocent |
| **d14 v2** | **14** | **3.506** | **29,750** | **champion reproduced, 4x faster (2,065 tok/s)** |

**Key findings:**
- Precomputed masks + cached RoPE + fused QK scale: 6x throughput (500 → 3,000 tok/s)
- seq_len=1024 diverges at val 3.634 regardless of precision — needs LR scheduling
- seq_len=512 reaches 3.506 without divergence — proven convergence path
- 29 automated optimization experiments identified batch size and precision as top levers
- Pure bf16 is fast (9x) but converges poorly; mixed precision is fast (7.5x) but same ceiling as f32

## Quick start

```bash
# Set up the environment
./scripts/setup_mlx_env.sh
source .venv/bin/activate

# Run a local smoke test
python scripts/smoke_test_mlx.py

# Train a small model locally (synthetic data)
python scripts/train_mlx_cluster.py \
  --parallelism dp \
  --backend any \
  --model-tag smoke-dp \
  --depth 2 \
  --max-seq-len 32 \
  --device-batch-size 2 \
  --total-batch-size 64 \
  --num-iterations 10 \
  --data-mode synthetic \
  --export-final
```

## Cluster setup

To run distributed training on your own Mac Mini cluster:

1. Connect two Macs via Thunderbolt cable
2. Configure RDMA — see [docs/reproducible_cluster_setup.md](docs/reproducible_cluster_setup.md)
3. Set your node hostnames:
   ```bash
   export NODE0="your-node-0.local"
   export NODE1="your-node-1.local"
   ```
4. Verify the link:
   ```bash
   ./scripts/check_jaccl_ready.sh
   ```
5. Launch distributed training:
   ```bash
   ./scripts/cluster_run_train.sh jaccl dp \
     --model-tag my-d12 --depth 12 --max-seq-len 512 \
     --device-batch-size 2 --total-batch-size 2048 \
     --num-iterations 200 --checkpoint-every 50 --export-final
   ```

All cluster scripts accept environment variables for node hostnames, IPs, and interfaces. See each script header for available options.

## Project structure

```
nanochat_mlx/          MLX model, training loop, data loading, distributed
scripts/               Training launchers, cluster operations, utilities
tests/                 Unit tests
docs/                  Guides and research notes
site/                  Landing page (mlxgpt.com)
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.11
- MLX >= 0.31.1

For cluster training: two Macs with Thunderbolt connection and JACCL/RDMA configured.

## Docs

- [Running nanochat on Apple Silicon](docs/nanochat_on_apple_silicon.md) — architecture decisions and MLX bridge design
- [Reproducible cluster setup](docs/reproducible_cluster_setup.md) — from-scratch 2-node RDMA guide
- [MLX cluster training](docs/mlx_cluster_training.md) — distributed training runbook
- [Rust/C++ optimization research](docs/rust_cpp_optimization_research.md) — analysis of native code optimization potential
- [Three-node expansion research](docs/three_node_m4pro_research.md) — scaling to 3 nodes

## License

MIT
