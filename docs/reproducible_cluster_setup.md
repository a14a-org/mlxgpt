# Reproducible 2-Node Thunderbolt MLX Setup

> **Customize for your setup:** Replace `<node-0>`, `<node-1>`, `<your-username>`,
> interface names (`en4`/`en2`), RDMA devices (`rdma_en4`/`rdma_en2`), and
> Thunderbolt IPs (`192.168.0.1`/`192.168.0.2`) with the values that match your
> cluster. The shell scripts in `scripts/` accept these as environment variables
> (e.g. `NODE0`, `NODE1`, `TB_IFACE_NODE0`, `RDMA_DEV_NODE0`, `TB_IP_NODE0`).

This is the full from-scratch runbook for reproducing the current `auto-mlx` cluster setup on two Apple Silicon Macs.

It covers:

- local-only SSH setup
- MLX and `nanochat` environment bootstrap
- enabling RDMA over Thunderbolt
- discovering the active Thunderbolt port pair
- bringing up JACCL
- validating both distributed smoke and distributed inference

This document is intentionally longer than the short smoke-stack guide in [cluster_smoke_stack.md](./cluster_smoke_stack.md). Use this one when starting from zero.

## Official references

These are the primary references that informed the setup here:

- MLX distributed communication docs:
  - https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- MLX launching/config docs:
  - https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html
- MLX releases:
  - https://github.com/ml-explore/mlx/releases

Key points confirmed from the official docs as of March 14, 2026:

- JACCL requires macOS `26.2+`
- RDMA over Thunderbolt must be enabled in Recovery with `rdma_ctl enable`
- `mlx.distributed_config --over thunderbolt --dot` is the recommended way to inspect cable topology
- even for JACCL, Thunderbolt point-to-point IP setup is still required
- disabling the Thunderbolt Bridge is part of the expected setup

## Scope and assumptions

This runbook assumes:

- two Apple Silicon Macs
- both connected with a Thunderbolt 5 cable
- both on the same local network
- both reachable by `.local` hostnames
- this workspace lives at the same path on both nodes:
  - `~/Documents/auto-mlx`

Current validated cluster names:

- controller and launcher: `<node-0>`
- worker: `<node-1>`

Current validated working JACCL mapping for this specific cluster:

- `<node-0>`: `en4` / `rdma_en4`
- `<node-1>`: `en2` / `rdma_en2`

Important: if the cable is moved, the active port pair can change. Always re-check with `mlx.distributed_config --dot`.

## 1. Prepare the Macs

Make the software baseline match first.

Checklist:

- both Macs on the same macOS version
- both Macs on macOS `26.2+`
- `Remote Login` enabled on both
- both unlocked after reboot before expecting SSH to work normally

Why matching versions matters:

- it is not explicitly required by the MLX docs, but in practice we aligned both nodes before the final successful JACCL bring-up

## 2. Enable passwordless SSH

Do this from your controller machine first.

If you do not already have a dedicated local cluster key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_auto_mlx_local -C "auto-mlx-local"
```

Copy the key to both nodes:

```bash
cat ~/.ssh/id_ed25519_auto_mlx_local.pub | ssh <your-username>@<node-0> 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
cat ~/.ssh/id_ed25519_auto_mlx_local.pub | ssh <your-username>@<node-1> 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

Add a local SSH config on the controller:

```sshconfig
Host <node-0> <node-1>
  User <your-username>
  IdentityFile ~/.ssh/id_ed25519_auto_mlx_local
  IdentitiesOnly yes
  PreferredAuthentications publickey
```

Validate:

```bash
ssh <node-0> hostname
ssh <node-1> hostname
```

Then make sure `<node-0>` can SSH to `<node-1>`, because `mlx.launch` will use SSH between nodes:

On `<node-0>`:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_auto_mlx_cluster -C "auto-mlx-cluster"
cat ~/.ssh/id_ed25519_auto_mlx_cluster.pub | ssh <your-username>@<node-1> 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

On `<node-0>`, add:

```sshconfig
Host <node-0> <node-1>
  User <your-username>
  IdentityFile ~/.ssh/id_ed25519_auto_mlx_cluster
  IdentitiesOnly yes
  PreferredAuthentications publickey
  StrictHostKeyChecking accept-new
```

Validate from `<node-0>`:

```bash
ssh <node-1> hostname
```

## 3. Create the workspace and mirror the project

On both nodes:

```bash
mkdir -p ~/Documents/auto-mlx
```

Sync this workspace to both nodes at the same path. The exact tool is flexible, but the important constraints are:

- same path on both nodes
- same scripts on both nodes
- same Python environment locations on both nodes

If using `rsync --delete`, exclude generated artifacts unless you intentionally want to wipe them:

- `.nanochat-cache/`
- `converted/`

## 4. Bootstrap the environments

On both nodes:

```bash
cd ~/Documents/auto-mlx
./scripts/setup_mlx_env.sh
./scripts/setup_nanochat_mps.sh
```

Validate the MLX bridge locally on each node:

```bash
cd ~/Documents/auto-mlx
. .venv/bin/activate
python scripts/smoke_test_mlx.py
```

## 5. Enable RDMA over Thunderbolt

This part follows the official MLX docs and must be done in Recovery.

On each Mac:

1. Boot into Recovery
2. Open `Utilities -> Terminal`
3. Run:

```bash
rdma_ctl enable
```

4. Reboot

After reboot, verify on each node:

```bash
rdma_ctl status
ibv_devices
```

Expected:

- `rdma_ctl status` prints `enabled`
- `ibv_devices` lists Thunderbolt RDMA devices such as `rdma_en2`, `rdma_en3`, `rdma_en4`

## 6. Verify the simple network path first

Before touching JACCL, validate ordinary distributed MLX over the LAN.

From `<node-0>`:

```bash
cd ~/Documents/auto-mlx
. .venv/bin/activate
./scripts/cluster_run_smoke.sh ring
```

Expected outputs:

- `build/cluster/smoke-ring/rank_0.json` on `<node-0>`
- `build/cluster/smoke-ring/rank_1.json` on `<node-1>`

This confirms:

- host reachability
- `mlx.launch`
- hostfile creation
- script paths
- Python paths

## 7. Discover the active Thunderbolt cable mapping

From `<node-0>`:

```bash
cd ~/Documents/auto-mlx
. .venv/bin/activate
mlx.distributed_config --verbose --hosts <node-0>,<node-1> --over thunderbolt --dot
```

This prints a GraphViz-style graph such as:

```text
graph G {
  node [shape=rectangle];
  a [label="<node-0>"];
  b [label="<node-1>"];
  a -- b [label="en4/en2"]
}
```

That means:

- the active Thunderbolt port on `<node-0>` is `en4`
- the active Thunderbolt port on `<node-1>` is `en2`

If the graph changes after moving the cable, update the setup commands accordingly.

## 8. Configure the Thunderbolt point-to-point IPs

For the currently validated cable mapping on this cluster:

On `<node-0>`:

```bash
sudo ifconfig bridge0 down
sudo ifconfig en4 inet 192.168.0.1 netmask 255.255.255.252
sudo route change 192.168.0.2 -interface en4
```

On `<node-1>`:

```bash
sudo ifconfig bridge0 down
sudo ifconfig en2 inet 192.168.0.2 netmask 255.255.255.252
sudo route change 192.168.0.1 -interface en2
```

## 9. Remove the active Thunderbolt interfaces from `bridge0`

This turned out to be the missing practical step on this cluster.

Without this, the RDMA device could exist and even have the correct IP, but still remain `PORT_DOWN`.

On `<node-0>`:

```bash
sudo ifconfig bridge0 deletem en4
```

On `<node-1>`:

```bash
sudo ifconfig bridge0 deletem en2
```

Useful interpretation:

- if `deletem` succeeds, the interface was still a member of `bridge0`
- if it says `No such file or directory`, that usually means it is already not a member

## 10. Validate JACCL readiness

From `<node-0>`:

```bash
cd ~/Documents/auto-mlx
./scripts/check_jaccl_ready.sh
```

Expected on this cluster:

```text
== <node-0> ==
rdma_ctl: enabled
rdma_device_present: yes
rdma_port_state: PORT_ACTIVE
iface_ipv4: 192.168.0.1
iface_status: active

== <node-1> ==
rdma_ctl: enabled
rdma_device_present: yes
rdma_port_state: PORT_ACTIVE
iface_ipv4: 192.168.0.2
iface_status: active

JACCL readiness: READY
```

If it still shows `PORT_DOWN`:

- re-run the `--dot` topology command
- confirm you configured the correct interfaces
- confirm the active interfaces were removed from `bridge0`
- if needed, reboot and reapply the Thunderbolt IP + `deletem` steps

## 11. Validate JACCL itself

From `<node-0>`:

```bash
cd ~/Documents/auto-mlx
. .venv/bin/activate
./scripts/cluster_run_smoke.sh jaccl
```

Expected:

- both ranks complete
- `all_sum` equals `3.0` for all entries
- `all_gather` equals `[0, 1]`

This writes:

- `build/cluster/smoke-jaccl/rank_0.json`
- `build/cluster/smoke-jaccl/rank_1.json`

## 12. Validate the converted model over JACCL

The repo already includes the bridge and smoke scripts for converted `nanochat` checkpoints.

From `<node-0>`:

```bash
cd ~/Documents/auto-mlx
. .venv/bin/activate
./scripts/cluster_run_inference.sh jaccl
```

This validates:

- model load on both nodes
- tokenizer load on both nodes
- coordinated distributed execution
- deterministic matching outputs via checksum comparison

Outputs:

- `build/cluster/inference-jaccl/rank_0.json`
- `build/cluster/inference-jaccl/rank_1.json`

## 13. Train a real small model and convert it

This is the current workflow for producing a real model artifact:

1. train upstream `nanochat` on one node
2. convert it to MLX
3. sync the converted artifact
4. validate distributed inference separately

Example validated run on `<node-0>`:

```bash
export NANOCHAT_BASE_DIR=$HOME/Documents/auto-mlx/.nanochat-cache
cd ~/Documents/auto-mlx/nanochat
. .venv/bin/activate
python -m scripts.base_train \
  --device-type=mps \
  --depth=6 \
  --head-dim=64 \
  --window-pattern=L \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --total-batch-size=2048 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --num-iterations=30 \
  --run=dummy \
  --model-tag=cluster-d6
```

Convert it on `<node-0>`:

```bash
cd ~/Documents/auto-mlx
. .venv/bin/activate
python scripts/convert_nanochat_to_mlx.py \
  --checkpoint-dir "$HOME/Documents/auto-mlx/.nanochat-cache/base_checkpoints/cluster-d6" \
  --output-dir "$HOME/Documents/auto-mlx/converted/cluster-d6" \
  --tokenizer-dir "$HOME/Documents/auto-mlx/.nanochat-cache/tokenizer"
```

Sync it to `<node-1>`:

```bash
cd ~/Documents/auto-mlx
./scripts/cluster_sync_converted_model.sh cluster-d6
```

Validate it over JACCL:

```bash
cd ~/Documents/auto-mlx
. .venv/bin/activate
CLUSTER_MODEL_DIR=~/Documents/auto-mlx/converted/cluster-d6 \
CLUSTER_TOKENIZER_DIR=~/Documents/auto-mlx/converted/cluster-d6/tokenizer \
./scripts/cluster_run_inference.sh jaccl
```

## 14. Current validated results

These have been validated on this cluster:

- 2-node `ring` smoke
- 2-node `jaccl` smoke
- 2-node `ring` converted-model inference
- 2-node `jaccl` converted-model inference
- larger real converted checkpoint `cluster-d6` over JACCL

Current artifact examples:

- `cluster-smoke` converted model
- `cluster-d6` converted model

## 15. What is not automated yet

These parts still need manual handling:

- enabling RDMA in Recovery
- reapplying Thunderbolt point-to-point IP configuration after reboot if the interfaces reset
- removing the active Thunderbolt interfaces from `bridge0`
- re-discovering the active port mapping if the cable is moved

## 16. Known operational findings

- `ring` works reliably with raw IPs in the hostfile
- JACCL requires the correct RDMA device mapping in the hostfile
- the active Thunderbolt interfaces may not be the same after moving the cable
- `check_jaccl_ready.sh` is the fastest way to tell whether the setup is really usable
- `PORT_ACTIVE` is the key success signal at the RDMA layer
- `PORT_DOWN` means the hostfile/launcher may be fine, but the Thunderbolt/RDMA path is still not correctly configured
- after reboot, macOS may be reachable by SSH but still locked; a local login may be required before SSH works normally

## 17. Fast recovery checklist

If JACCL was working before and stopped working after a reboot or cable move:

1. confirm both Macs are unlocked and reachable over SSH
2. run:

```bash
ssh <node-0> 'cd ~/Documents/auto-mlx && . .venv/bin/activate && mlx.distributed_config --verbose --hosts <node-0>,<node-1> --over thunderbolt --dot'
```

3. identify the active interface pair
4. assign the `192.168.0.1/30` and `192.168.0.2/30` point-to-point IPs to those interfaces
5. remove those interfaces from `bridge0` with `ifconfig bridge0 deletem ...`
6. run:

```bash
ssh <node-0> 'cd ~/Documents/auto-mlx && ./scripts/check_jaccl_ready.sh'
```

7. once it shows `READY`, run:

```bash
ssh <node-0> 'cd ~/Documents/auto-mlx && . .venv/bin/activate && ./scripts/cluster_run_smoke.sh jaccl'
```
