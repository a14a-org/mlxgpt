# Three-Node M4 Pro Research Note

This note captures the current thinking on whether adding a third `Mac mini M4 Pro / 64 GB` node would meaningfully shorten the path from the current 2-node MLX cluster toward `nanochat` "time to GPT-2" style experiments.

This is a research and planning document only. It does not imply that the repo is currently configured for a 3-node cluster.

## Short Answer

Adding a third `M4 Pro` node would likely help, but it probably would not reduce the timeline in a perfectly linear way.

Working estimate:

- likely practical DP speedup vs the current 2-node setup: about `1.3x-1.7x`
- likely impact on a rough `3-6 week` path: more like `2-4 weeks`

This is an inference from:

- the current measured 2-node MLX DP throughput on this cluster
- the way data parallel scaling usually behaves once communication overhead becomes noticeable
- the official MLX distributed communication constraints for JACCL over Thunderbolt

## What a Third Node Helps With

The main benefit of a third node is faster `dp` experimentation.

That means:

- shorter wall-clock time for `d10`, `d12`, `d16` style training runs
- faster iteration on scaling-law style checks
- faster time to determine whether larger Apple Silicon training is viable

What it does **not** directly solve:

- fitting much larger single models
- reducing the need for `tp` once model size becomes the real bottleneck

So the third node is most valuable for accelerating the research loop, not for eliminating the eventual need for tensor parallelism.

## Current Baseline

What we know from the current cluster:

- 2-node MLX DP over JACCL is working end to end
- resume, export, sync, and distributed inference are working
- `cluster-mlx-d8` completed a long run successfully
- the current `cluster-mlx-d10` measured run is slower than `d8`, as expected

The current 2-node `d10` throughput is in the rough range of `4k-5k tok/s`, while the mature `d8` run was closer to `6k tok/s`.

That means we are already seeing the expected scaling pressure from model size on Apple Silicon, even before moving toward `d24-d26`.

## Hardware and Topology Constraints

For a 3-node JACCL setup, the most important finding is:

- the MLX JACCL backend supports only fully connected topologies
- that means every Mac must be directly connected to every other Mac with Thunderbolt

For `3` nodes, that means `3` cables total:

- node A <-> node B
- node A <-> node C
- node B <-> node C

This is physically plausible with the `M4 Pro` Mac mini because Apple documents `three Thunderbolt 5 (USB-C) ports` on the back of the `M4 Pro` model.

Implication:

- a 3-node JACCL cluster is feasible in principle
- but only if we build a proper Thunderbolt mesh, not a star through Ethernet or a partial cable layout

## MLX Requirements That Matter

The current MLX docs imply these operational requirements for a 3-node JACCL cluster:

- passwordless SSH must work among the launch/controller setup
- the same Python path and project path must exist on all nodes
- RDMA over Thunderbolt must be enabled on all nodes
- Thunderbolt Bridge must still be disabled
- isolated local networks still need to be configured for each Thunderbolt connection
- the JACCL hostfile must include the RDMA device mapping for each node-to-node connection

For 3 nodes this becomes materially more complex than the 2-node setup, because each node participates in multiple Thunderbolt links and the hostfile needs to describe all pairwise RDMA paths.

## Timeline Impact

The realistic benefit is probably meaningful but not dramatic.

Reasoning:

- idealized DP scaling from 2 nodes to 3 nodes would be `1.5x`
- in practice, collective communication and launch overhead reduce that gain
- larger models increase compute per step, which helps scaling somewhat
- but Apple Silicon + Thunderbolt + JACCL on a small cluster is still not the same as NVLink-heavy H100 infrastructure

Best rough planning range:

- pessimistic: `1.2x-1.3x`
- likely: `1.3x-1.7x`
- optimistic: near `1.8x` on the right model size and batch regime

That is enough to matter for the project, especially if the goal is to reduce iteration latency on measured runs.

## Recommendation

Recommendation today:

- yes, a third `M4 Pro / 64 GB` node is worth adding
- but it should be treated as an experiment-acceleration investment, not a magic jump to GPT-2 capability

Recommended decision rule:

- if the 2-node `d10` and follow-up `d12/d16` data shows stable validation behavior and no memory pressure, then adding a third node is a good next infrastructure move
- if the data suggests we are already running into model-size limits rather than wall-clock limits, then `tp` work may be a better next investment than pure cluster-size growth

In short:

- add node 3 if the next few measured DP runs look promising
- do not assume node 3 replaces the need for eventual tensor parallel work

## What To Revisit Once More 2-Node Data Is In

Before making the 3-node jump, revisit these questions:

- does `d10` show better validation behavior than `d8` on the new eval-aware trainer?
- what is the true tokens/sec trend over the whole `d10` run, not just short windows?
- do longer `d10` or `d12` runs stay off swap on both nodes?
- does the quality improvement justify the slower throughput?
- are we bottlenecked by wall-clock training time or by model size?

If the answer is "wall-clock time", a third node is attractive.

If the answer is "model size", `tp` becomes more attractive.

## Suggested Future Work Order

When we are ready to act on this:

1. finish evaluating the current 2-node measured `d10` run
2. compare `d10` against the completed `d8` baseline
3. if results are encouraging, plan the 3-node physical topology and RDMA/JACCL hostfile strategy
4. validate 3-node `ring` first
5. validate 3-node `jaccl`
6. then rerun the measured DP ladder on 3 nodes

## Sources

Primary references used for this note:

- MLX distributed communication docs: https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- MLX distributed launching docs: https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html
- Apple Mac mini (2024) specs: https://support.apple.com/en-us/121555
- Upstream `nanochat` README: https://github.com/karpathy/nanochat/blob/master/README.md

Key source points:

- MLX states that the JACCL backend supports only fully connected Thunderbolt topologies.
- MLX documents Thunderbolt RDMA availability starting with macOS `26.2`.
- Apple documents that the `M4 Pro` Mac mini has `three Thunderbolt 5` ports.
- Upstream `nanochat` currently frames GPT-2-grade capability as roughly `d24-d26`.
