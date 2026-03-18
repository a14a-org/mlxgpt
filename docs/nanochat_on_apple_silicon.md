# nanochat on Apple Silicon

## Bottom line

- `nanochat` already has a basic Apple Silicon path through PyTorch `mps`, but upstream describes it as a dramatically smaller educational run rather than a strong training setup.
- Native MLX is a better fit if the goal is to lean into Apple Silicon for inference, small-scale training, or multi-Mac experiments.
- The bridge in this workspace is the practical middle path: keep upstream `nanochat` untouched, convert checkpoints into MLX, and run them natively on Apple hardware.

## What is proven today

### 1. Upstream `nanochat` does run on Apple Silicon through PyTorch `mps`

Karpathy’s README says the code is fairly vanilla PyTorch and should run on backends like `mps`, while warning that those code paths may have sharp edges. It also ships `runs/runcpu.sh` specifically for CPU or Apple Silicon and says this route is mostly educational rather than a way to get strong results.

Source:
- https://github.com/karpathy/nanochat/blob/master/README.md

### 2. MLX has first-party distributed support

The official MLX distributed docs describe:

- `mlx.launch` for multi-process and multi-host launch
- the ring backend over TCP sockets
- the JACCL backend for Thunderbolt RDMA
- direct environment-variable launch without `mlx.launch`

That means multi-Mac execution is not speculative anymore. It is an official MLX feature, not just a community hack.

Sources:
- https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- https://github.com/ml-explore/mlx

### 3. Thunderbolt RDMA is documented by MLX for low-latency Mac-to-Mac communication

The MLX distributed docs say that starting from macOS 26.2, RDMA over Thunderbolt is available on Macs with Thunderbolt 5 and that the JACCL backend uses it for much lower latency than the ring backend. The same docs also state that JACCL requires a fully connected topology and provide a full Thunderbolt mesh example.

For your three M4 Pro Mac minis, Apple’s current specs list three Thunderbolt 5 ports on each unit, so a full 3-node pairwise mesh is physically possible.

Sources:
- https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- https://www.apple.com/mac-mini/specs/

## What is not proven

### 1. I did not find evidence that multiple Macs can be exposed as one shared Apple Silicon processor

What MLX documents is distributed execution across separate hosts and ranks. That is very different from the OS presenting several Macs as one giant unified-memory accelerator. So the realistic architecture is distributed training or distributed inference, not true processor pooling.

This is an inference from the MLX docs and from the absence of any Apple or MLX documentation describing host-transparent processor pooling.

### 2. `nanochat` is not directly consumable by MLX tooling

`nanochat` uses a custom GPT implementation, custom checkpoint layout, custom tokenizer flow, and training assumptions built around PyTorch. Official `mlx-lm` is excellent for mainstream Hugging Face style model serving and fine-tuning, but it does not directly load `nanochat` checkpoints.

Source:
- https://github.com/ml-explore/mlx-lm

## Recommended path

### Phase 1: make the single-box path solid

1. Use upstream `nanochat` with `mps` only for small educational runs and checkpoint production.
2. Convert interesting checkpoints into MLX using `scripts/convert_nanochat_to_mlx.py`.
3. Run generation natively on Apple Silicon with `scripts/generate_mlx.py`.

### Phase 2: decide whether the cluster is for inference or training

- For inference, MLX plus JACCL over Thunderbolt 5 is the most promising route.
- For training, start with MLX ring or TCP over Ethernet first, then move to JACCL only after a single-node path is stable.

### Phase 3: cluster the three M4 Pro minis

For the three Mac mini M4 Pro machines:

- Prefer a full Thunderbolt 5 mesh if you want to test JACCL.
- Keep 10GbE available anyway for management and easier fallback.
- Treat the laptop as a controller or dev box first, not as a core cluster node.

## Why this workspace is structured this way

The bridge code here does not rewrite all of `nanochat`. Instead it gives us:

- an MLX-native model definition that matches the checkpoint layout closely
- a converter from PyTorch checkpoint files to MLX weights
- a small generation entrypoint for Apple Silicon validation

That keeps the upstream clone easy to update while still giving us a concrete MLX path we can extend.
