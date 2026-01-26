# 📦 Installation

This document describes all dependencies required to run **DataArc SynData Toolkit** and provides detail installation guide.

## 1. Hardware Requirements

This project requires GPU environment. We recommend following settings:
  - Linux (Ubuntu 22.04+) or Windows 10/11
  - CUDA 12.8+
  - GPU memory ≥ 24 GB (for 7B–13B models)

## 2. Core Python Dependencies

| Category | Library | Version | Notes |
|---------|----------|---------|--------|
| **Core** | Python | 3.11.13 | Fixed version required |
| **LLM Engine** | vLLM | ≥0.11.0 | Inference engine with CUDA 12.8 |
| **Deep Learning** | PyTorch | 2.8.0 | Fixed version with CUDA 12.8 |
|  | torchvision | 0.23.0 | Vision utilities |
|  | torchaudio | 2.8.0 | Audio processing |
| **Training** | flash-attn | ≥2.8.3 | Optimized attention |
|  | peft | ≥0.18.0 | Parameter-efficient fine-tuning |
|  | tensordict | ≥0.10.0 | RL data structures |
|  | ray[default] | ≥0.1.0 | Distributed training framework |
| **Model loading** | sentence-transformers | ≥5.1.2 | Embedding models |
| **Tokenizer** | tiktoken | ≥0.12.0 | Fast tokenization |
| **Data Processing** | datasets | ≥4.4.1 | HuggingFace datasets |
|  | pandas | ≥2.3.3 | Data manipulation |
|  | pyarrow | ≥22.0.0 | Columnar data format |
|  | torchdata | ≥0.11.0 | Data loading pipelines |
| **Document Processing** | mineru[core] | ≥2.6.4 | PDF → JSONL pipeline |
|  | pymupdf | ≥1.26.5 | PDF parsing |
|  | rank-bm25 | ≥0.2.2 | Passage retrieval |
| **Web API** | fastapi | ≥0.124.2 | REST API framework |
| **Configuration** | hydra-core | ≥1.3.2 | Config management |
|  | pyyaml | ≥6.0.3 | YAML parsing |
|  | python-dotenv | ≥1.1.1 | Environment variables |
| **Monitoring** | wandb | ≥0.23.1 | Experiment tracking |
|  | codetiming | ≥1.4.0 | Performance profiling |
|  | tqdm | ≥4.67.1 | Progress bars |
| **Build Tools** | hatchling | ≥1.28.0 | Build backend |
|  | editables | ≥0.5 | Editable installs |
| **API Clients** | openai | ≥2.6.0 | OpenAI API support |

## 3. Installation guide

If you encounter problem building dependencies with ``uv sync``, please follow this installation guide.

Firstly, specified the python version that you want and delete the cuda dependent package in [pyproject.toml](../pyproject.toml)

```shell
requires-python = "==3.11.13"  ## specified python version here
dependencies = [
    # Core SDG dependencies
    "datasets>=4.4.1",
    "fastapi>=0.124.2",
    "mineru[core]>=2.6.4",
    "openai>=2.6.0",
    "pymupdf>=1.26.5",
    "python-dotenv>=1.1.1",
    "pyyaml>=6.0.3",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=5.1.2",
    "tiktoken>=0.12.0",
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "torchvision==0.23.0",
    "tqdm>=4.67.1",
    "vllm>=0.11.0",            # delete this line since it's cuda dependent

    # Training dependencies
    "pandas>=2.3.3",
    "codetiming>=1.4.0",
    "flash-attn>=2.8.3",       # delete this line since it's cuda dependent
    "hydra-core>=1.3.2",
    "peft>=0.18.0",
    "pyarrow>=22.0.0",
    "ray[default]>=0.1.0",
    "tensordict>=0.10.0",
    "torchdata>=0.11.0",
    "wandb>=0.23.1",

    # Evaluation dependencies
    "deepeval>=3.8.0",

    # Build tools
    "editables>=0.5",
    "hatchling>=1.28.0",
]
```

#### Step 1 — Check Your CUDA Version

Before installing PyTorch or vLLM, confirm your CUDA verison, run command ``nvidia-smi``.

Change the cuda version specified in [pyproject.toml](../pyproject.toml) to your cuda version.

``` shell
[[tool.uv.index]]
name = "pytorch-cu128"  ## e.g. if you have cuda 12.6, change cu128 -> cu126, all of them below as well
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }
```

#### Step 2 — Install Matching PyTorch Version

Install the correct pytorch version that match your cuda version. See details in [Pytorch Previous Versions](https://pytorch.org/get-started/previous-versions/).

``` shell
# for example, if you are using cuda12.6, you can install torch 2.7.0 using this uv command
uv add torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
```

> [!Note]
> If you want to experience the model training module, you should have CUDA>=12.8

#### Step 3 — Install CUDA Dpendent Dependencies (Must Be After PyTorch)

vLLM and flash-attention depends on your existing PyTorch installation and CUDA runtime. Install the correct version of vLLM and flash-attention with following command.

```shell
uv add flash-attn vllm
```

#### Step 4 - Install General Dependencies (CUDA-Independent)

```shell
uv sync
```