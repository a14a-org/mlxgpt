# mlxgpt

Python project for training GPT-2 scale language models on Apple Silicon using MLX framework.

## Stack

- Python with MLX framework
- Distributed training (JACCL/RDMA over Thunderbolt)
- Static site in `site/` for mlxgpt.com

## Commands

- Training: `python -m nanochat_mlx.train`
- Tests: `python -m pytest tests/`

## Structure

- `nanochat_mlx/` - core training code
- `site/` - landing page
- `scripts/` - utility scripts
- `docs/` - documentation
