# Research: Rust/C/C++ Optimization Potential for nanochat_mlx

**Date:** 2026-03-17
**Status:** Investigation complete
**Authors:** Expert agent team (MLX Internals, Systems Performance, Ecosystem Research)

---

## Executive Summary

The core finding is counterintuitive: **replacing Python with Rust/C/C++ would provide near-zero speedup for compute-bound operations.** MLX is a C++ framework with Metal GPU backends — Python is only a thin graph-construction layer. The GPU does all the real work regardless of host language.

However, there are **significant pure-Python optimization opportunities** that don't require a language change, and **one specific module** (data loading / bin packing) where Rust could help if it becomes a bottleneck.

---

## Table of Contents

1. [How MLX Actually Works Under the Hood](#1-how-mlx-actually-works-under-the-hood)
2. [Where the Time Actually Goes](#2-where-the-time-actually-goes)
3. [Python Overhead Analysis](#3-python-overhead-analysis)
4. [Quick Wins Without Any Language Change](#4-quick-wins-without-any-language-change)
5. [Rust/C/C++ Replacement Candidates](#5-rustcc-replacement-candidates)
6. [MLX Ecosystem Assessment](#6-mlx-ecosystem-assessment)
7. [Risk Assessment](#7-risk-assessment)
8. [Recommended Action Plan](#8-recommended-action-plan)
9. [Appendix: Detailed Benchmark Estimates](#9-appendix-detailed-benchmark-estimates)

---

## 1. How MLX Actually Works Under the Hood

### Architecture

MLX is a C++ framework (65% of its codebase) with Python bindings via Nanobind. Every operation called from Python — `mx.add`, `nn.Linear`, `mx.matmul` — is a thin wrapper that records a node in a **lazy computation graph** implemented in C++. No actual computation happens in Python.

When `mx.eval()` is called, the C++ backend walks the graph, dispatches work to Metal, and blocks until results are ready. Python is idle during GPU execution.

### Hand-Written Metal Kernels

MLX ships with dedicated, hand-optimized Metal shaders for all core operations:

| Metal Kernel | Used By (in nanochat_mlx) |
|---|---|
| `matmul` | `nn.Linear` — Q/K/V projections, MLP layers |
| `scaled_dot_product_attention` | `mx.fast.scaled_dot_product_attention` (fused Flash Attention) |
| `rope` | `mx.fast.rope` (rotary positional embeddings) |
| `rms_norm` | `mx.fast.rms_norm` (root mean square normalization) |
| `softmax` | Loss computation |
| Binary/unary/ternary ops | Element-wise arithmetic, activations |
| `reduce` | Sum, mean, max, min reductions |
| `quantized` | Quantized matmul |

### Compilation via `mx.compile`

`mx.compile()` traces a Python function, builds a fused computation graph, and merges element-wise operations into single Metal kernels. Apple reports **5x speedups** on element-wise chains (e.g., GELU). Recompilation is triggered by shape/dtype changes; a `shapeless=True` mode exists for variable-length inputs.

The compilation pipeline runs three passes:
1. **Simplification** — scalar consolidation, no-op elimination, common subexpression elimination
2. **Dependency tracking** — topological ordering via DFS
3. **Kernel fusion** — merges compatible element-wise operations (depth limit: 11, input limit: 24 arrays)

---

## 2. Where the Time Actually Goes

### Operation-Level Breakdown

| Component | Runs In | Bottleneck? | Python Overhead |
|---|---|---|---|
| Matrix multiplications (Q/K/V, MLP) | Metal GPU | **Yes — dominates** | Zero |
| `mx.fast.scaled_dot_product_attention` | Metal GPU (fused kernel) | **Yes** | Zero |
| RMS norm (`model.py:13`) | Metal GPU (multiple kernels) | Minor | Zero |
| RoPE (`model.py:21-34`) | Metal GPU (multiple kernels) | Minor | Zero |
| `nn.value_and_grad` | C++ autograd + Metal | **Yes** | Zero |
| `mx.distributed.all_sum` | Metal + interconnect | Moderate | Zero |
| Best-fit packing (`data.py:200-229`) | **Pure Python** | Potentially | **Significant** |
| Tokenization | Rust (tiktoken) | No | Zero |
| Parquet I/O | C++ (PyArrow) | No | Minimal |

### Key Insight

For a transformer with `n_embd=768`, `n_layer=12`, the forward pass is dominated by matrix multiplications (`O(batch * seq * n_embd^2)` per layer) and attention (`O(batch * heads * seq^2 * head_dim)`). These run on Metal regardless of the host language.

---

## 3. Python Overhead Analysis

### 3.1 Training Loop Inner Loop (training.py:466-490)

```python
for _ in range(grad_accum_steps):
    batch_inputs, batch_targets, loader_state = self.batch_iterator.next_batch()
    inputs = mx.array(batch_inputs, dtype=mx.int32)
    targets = mx.array(batch_targets, dtype=mx.int32)
    loss, grads = self.loss_and_grad(self.model, inputs, targets)
    accumulated_loss = accumulated_loss + loss
    accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)
```

**Python overhead: negligible.** Each `loss_and_grad` call enqueues an entire forward+backward pass onto the computation graph. The `tree_map` for gradient accumulation is O(num_parameter_tensors) in count (roughly 50-100 leaf tensors for a 12-layer model), not O(num_elements). Each `a + b` just enqueues a Metal add node.

### 3.2 Data Loading — Best-Fit Packing (data.py:200-229)

This is the **most significant Python overhead** in the codebase:

```python
for idx, doc in enumerate(self.doc_buffer):   # buffer_size=1000
    doc_len = len(doc)
    if doc_len <= remaining and doc_len > best_len:
        best_idx = idx
        best_len = doc_len
```

This is an O(buffer_size) linear scan per document placement, executed for each document packed into each row. With `buffer_size=1000` and `batch_size * docs_per_row` iterations, this is a hot loop entirely in the Python interpreter.

**However:** this runs between GPU steps. If data preparation takes less time than the GPU forward+backward pass, it is fully hidden. Only if data preparation exceeds GPU step time does this become a wall-clock bottleneck.

### 3.3 Tokenization (tokenizer.py)

Already optimized — `tiktoken`'s `encode_ordinary_batch()` with `num_threads=4` dispatches to Rust-native multithreaded tokenization. The HuggingFace tokenizer path lacks batching but the inner encoder is also Rust-based.

### 3.4 Parquet Reading

PyArrow's Parquet reading is C++ under the hood. The `read_row_group()` call and `.to_pylist()` conversion involve some Python object creation overhead, but this is I/O-dominated.

### 3.5 Gradient Accumulation Loop

The `tree_map` for gradient accumulation is Python iteration over the parameter tree, but this is O(num_parameter_tensors) in count — roughly 50-100 leaf tensors. Each `a + b` just enqueues a Metal add node. Not a bottleneck.

### 3.6 The `mx.eval()` Sync Point

At `training.py:490`:
```python
mx.eval(self.model.parameters(), self.optimizer.state, mean_loss)
```

This is the synchronization barrier where all lazy computation materializes. Python is blocked waiting for the GPU. This is by design — it **is** the actual compute happening.

---

## 4. Quick Wins Without Any Language Change

### 4.1 Replace Custom `rms_norm()` with `mx.fast.rms_norm`

**Current** (`model.py:13`):
```python
def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
```

This decomposes into **4 separate Metal kernel dispatches** (`square`, `mean`, `rsqrt`, `multiply`). `mx.fast.rms_norm` is a **single fused Metal kernel** that does the same thing with ~3-4x less memory bandwidth.

### 4.2 Replace Custom `apply_rotary_emb()` with `mx.fast.rope`

**Current** (`model.py:21-34`):
```python
def apply_rotary_emb(x: mx.array, offset: int, base: float) -> mx.array:
    # 8+ separate operations: arange, division, power, multiply, cos, sin, slice, concatenate
```

`mx.fast.rope` is a single fused Metal kernel that handles all of this in one pass.

### 4.3 Use `mx.compile()` on the Loss Function

Wrapping the loss computation with `mx.compile()` enables MLX's graph optimizer to fuse element-wise operations (residual connections, lambda scaling, squared ReLU) into fewer kernel launches. Apple benchmarks show 5x speedup on element-wise chains.

### 4.4 Async Data Prefetching

Pipeline `next_batch()` with GPU compute using `mx.async_eval` or a background thread to fully hide data loading latency.

### 4.5 Estimated Combined Impact

**5-15% training throughput improvement with zero code complexity increase.**

---

## 5. Rust/C/C++ Replacement Candidates

### 5.1 Rust Data Loader (Best Candidate)

**Target:** The bin-packing loop in `data.py:200-229`

A Rust PyO3 module could:
- Replace the O(N) linear scan with a BTreeMap keyed by length for O(log N) lookups
- Eliminate Python list manipulation overhead (pop, enumerate, len)
- Prefetch and pipeline Parquet I/O with packing in a separate thread

**Speedup estimate:** 5-20x for the packing logic itself. Overall training speedup: **1.0-1.3x** depending on whether packing is on the critical path.

**But first try the Python fix:** replace the linear scan with `bisect` or `sortedcontainers.SortedList` for O(log N) lookups. This is a 1-day change that may make Rust unnecessary.

### 5.2 Rust Training Loop (Not Recommended)

Rewriting the training orchestration in Rust would require:
- Rust bindings to MLX's C++ API (via mlx-rs)
- Re-implementing `nn.value_and_grad`, `tree_map`, optimizer logic
- Handling all checkpoint/logging/distributed logic

**Speedup estimate: 1.0-1.05x.** The Python overhead in the loop is microseconds of dict traversal and function dispatch per step. The GPU compute is milliseconds to seconds.

**Risk: extremely high.** Loss of MLX's automatic differentiation.

### 5.3 C/C++ Custom Metal Kernels

MLX supports two paths for custom Metal shaders:

**Path 1: `mx.fast.metal_kernel()` (Python-accessible, JIT)**
```python
kernel = mx.fast.metal_kernel(
    name="my_kernel",
    input_names=["inp"],
    output_names=["out"],
    source='uint elem = thread_position_in_grid.x; out[elem] = metal::exp(inp[elem]);'
)
```

**Path 2: C++ Extension with `.metal` files (compiled)**
Full control: subclass `Primitive`, implement forward and backward, compile `.metal` shaders via CMake, bind to Python via Nanobind.

**Potential use cases:**
- Fused RMSNorm + RoPE + QKNorm in attention (avoid intermediate memory writes)
- Fused softcap + cross-entropy loss

**However:** simply switching to `mx.fast.rms_norm` and `mx.fast.rope` captures most of this benefit with zero custom kernel work.

### 5.4 Full Model Rewrite in C++/Rust (Not Recommended)

This would mean writing the entire transformer forward/backward pass in a lower-level language, managing memory allocation, and reimplementing autograd.

**Speedup estimate: 1.0-1.2x** over MLX, which is already a well-optimized Metal-native framework.

**Risk: catastrophic.** This is essentially writing a new ML framework. Multi-year effort for marginal gains.

---

## 6. MLX Ecosystem Assessment

### 6.1 mlx-rs (Rust Bindings)

| Property | Status |
|---|---|
| Repository | [oxiglade/mlx-rs](https://github.com/oxiglade/mlx-rs) |
| Version | 0.25.3 (December 2025) |
| Stars | 278 |
| Training/Autograd | Yes, but awkward — closures cause issues, API "needs improvement" |
| Production users | None confirmed |
| Verdict | **Not ready for production training workloads** |

mlx-rs calls the same C++/Metal backend as Python MLX. **Same Metal kernels = same GPU performance.** The only potential benefit is reduced host-language overhead, which is negligible for this workload.

### 6.2 MLX C++ API

Fully featured and mirrors the Python API. You can build pure C++ MLX applications using CMake. This is the actual core of MLX — Python is the binding layer on top.

**Verdict:** Viable for production native applications, but offers no meaningful speedup for training workloads where GPU compute dominates.

### 6.3 mlx-c (C Bindings)

Official Apple project ([ml-explore/mlx-c](https://github.com/ml-explore/mlx-c)). Foundation for mlx-swift and mlx-rs. Actively maintained (last commit: March 2026).

### 6.4 mlx-swift (Apple's Recommended Native Path)

| Property | Status |
|---|---|
| Repository | [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) |
| Version | 0.30.6 (February 2026) |
| Stars | 1,600 |
| API coverage | "All capabilities in MLX (Python) should be available" |
| Training | Full support |
| Platforms | macOS, iOS, iPadOS, visionOS, Linux |
| Verdict | **Apple's recommended path for production native apps** |

WWDC25 recommendation: "Use Swift for shipping production apps; Python for prototyping."

### 6.5 ZMLX (Triton-style Toolkit for MLX)

[Hmbown/ZMLX](https://github.com/Hmbown/ZMLX) provides Python-first Metal kernel authoring:
```python
elementwise("x * tanh(log(1 + exp(x)))")
```
Compiles to Metal. Results: +12% decode speed on LFM2-8B, +7% on LFM2-24B. Works with stock MLX.

### 6.6 Summary Table

| Approach | Maturity | Training | Speedup | Effort | Risk |
|---|---|---|---|---|---|
| **Python + `mx.fast.*` + `mx.compile`** | Production | Full | 1.05-1.15x | Days | None |
| **Python + async data prefetch** | Production | Full | 1.1-1.3x | Days | None |
| **Rust data loader (PyO3)** | Viable | N/A (data only) | 1.0-1.3x | 1-2 weeks | Low |
| **mlx-rs (full Rust)** | Early (v0.25) | Limited | 1.0-1.05x | Months | High |
| **MLX C++ API** | Mature | Full | 1.0-1.05x | Months | Medium |
| **mlx-swift** | Mature (Apple) | Full | 1.0-1.05x | Months | Medium |
| **Custom Metal kernels** | Supported | With vjp/jvp | 1.05-1.15x | 1-2 months | Medium |
| **ZMLX fused kernels** | Emerging | Inference focus | 1.05-1.12x | Weeks | Low-Medium |
| **Full Metal rewrite** | Theoretical | Manual autograd | 1.0-1.2x | 6-12 months | Catastrophic |

---

## 7. Risk Assessment

### 7.1 Losing MLX's Automatic Differentiation

`nn.value_and_grad` (`training.py:240`) is the foundation of training. MLX's autograd traces Python-defined forward passes and compiles backward passes automatically. Any rewrite to C++/Rust would need to either:
- Reimplement autograd (multi-month effort, high bug risk)
- Use MLX's C++ API directly (less documented, less stable ABI)

**Risk: very high.** This is the single biggest reason NOT to rewrite the model/training loop.

### 7.2 Distributed Training Complexity

`tensor_parallel.py` uses `mx.distributed.all_gather`, `mx.distributed.all_sum` — MLX primitives tightly integrated with the computation graph. Reimplementing these requires custom MPI or communication layers.

**Risk: high.** Distributed correctness is hard to test and debug.

### 7.3 FFI Overhead

Each FFI boundary crossing costs ~100-500ns. For a Rust data loader returning numpy arrays once per batch, this is negligible. For frequent callbacks between Python and native code, it could add up.

**Risk: low** for data loading; **moderate** for anything requiring frequent Python/native interop.

### 7.4 Maintenance Burden

The current codebase is ~800 lines of Python — highly readable, easy to modify. MLX is evolving rapidly; Python API changes are easy to track, C++ ABI changes require recompilation and may break compatibility.

**Risk: low** for isolated Rust data loader; **high** for anything touching the model or training loop.

---

## 8. Recommended Action Plan

### Phase 1: Pure Python Optimizations (Days, Zero Risk)

1. **Switch `rms_norm()` to `mx.fast.rms_norm`** — single fused Metal kernel vs. 4 separate dispatches
2. **Switch `apply_rotary_emb()` to `mx.fast.rope`** — single fused kernel vs. 8+ operations
3. **Add `mx.compile()` to the loss/forward function** — enable kernel fusion for element-wise ops
4. **Profile:** measure wall-clock time in `next_batch()` vs. `mx.eval()` to determine if data loading is on the critical path

**Expected impact: 5-15% training throughput improvement.**

### Phase 2: Data Pipeline Optimization (Days, Low Risk)

5. **Replace O(N) packing scan with sorted data structure** — use `bisect` or `sortedcontainers.SortedList` for O(log N) best-fit lookups (Python change only)
6. **Add async batch prefetching** — pipeline data preparation with GPU compute using threading or `mx.async_eval`

**Expected impact: hides data loading latency entirely if GPU compute dominates.**

### Phase 3: Targeted Native Code (Weeks, Low Risk)

7. **Rust PyO3 module for bin packing** — only if Phase 2 profiling shows packing is still a bottleneck after the sorted data structure optimization
8. **Explore ZMLX-style fused Metal kernels** — for any remaining custom operations not covered by `mx.fast.*`

**Expected impact: 1.0-1.3x depending on data loading bottleneck severity.**

### Do Not Pursue

- Full Rust/C++ rewrite of model or training loop (1.0-1.05x speedup, months of effort, loss of autograd)
- Custom Metal replacement of MLX's built-in kernels (MLX team already optimizes these)
- mlx-rs for training (autograd API not ergonomic enough, no production users)
- Full Metal compute pipeline rewrite (essentially building a new ML framework)

---

## 9. Appendix: Detailed Benchmark Estimates

| Candidate | Speedup (Component) | Speedup (Overall Training) | Confidence | Effort | Risk |
|---|---|---|---|---|---|
| `mx.fast.rms_norm` replacement | 3-4x for norm ops | 1.02-1.05x | High | Hours | None |
| `mx.fast.rope` replacement | 3-4x for RoPE ops | 1.02-1.05x | High | Hours | None |
| `mx.compile()` on loss function | 2-5x for fused element-wise | 1.03-1.08x | Medium | Hours | None |
| Sorted packing (Python) | 5-10x for packing | 1.0-1.2x | Medium | 1 day | None |
| Async data prefetch | Hides latency | 1.0-1.3x | Medium | 1-2 days | None |
| Rust data loader (PyO3) | 5-20x for packing | 1.0-1.3x | Medium | 1-2 weeks | Low |
| Rust training loop | 1.0-1.05x | 1.0-1.05x | High | 2-3 months | Very High |
| Rust checkpoint I/O | 1.0x | 1.0x | High | 1 week | Low |
| Custom fused Metal kernels | 1.5-3x per fused op | 1.05-1.15x | Low | 1-2 months | Medium |
| C++ full Metal replacement | 1.0-1.2x | 1.0-1.2x | Low | 6-12 months | Catastrophic |

---

## Conclusion

**Python is just the remote control — optimizing the remote doesn't make the TV faster.**

The highest-impact changes are all achievable without leaving Python:
1. Use MLX's built-in fused kernels (`mx.fast.*`)
2. Enable graph compilation (`mx.compile`)
3. Optimize the data pipeline algorithm and add async prefetching

Only the bin-packing loop in `data.py` is a viable Rust candidate, and even that should be tried first as a Python algorithmic fix. The model and training loop should remain in Python where MLX's autograd, lazy evaluation, and rapid iteration provide maximum value.
