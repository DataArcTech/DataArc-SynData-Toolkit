# 📦 安装指南（中文版）

本文档描述运行 **DataArc SynData Toolkit** 所需的全部依赖，并提供详细的安装步骤。

## 1. 硬件要求

本项目需要 GPU 环境，推荐配置如下：
  - Linux（Ubuntu 22.04+）或 Windows 10/11
  - CUDA 12.8+
  - GPU 显存 ≥ 24 GB（适用于 7B–13B 模型）

## 2. 核心 Python 依赖

| 分类 | 库 | 版本 | 说明 |
|---------|----------|---------|--------|
| **核心** | Python | 3.11.13 | 固定版本要求 |
| **LLM 引擎** | vLLM | ≥0.11.0 | 推理引擎（CUDA 12.8） |
| **深度学习** | PyTorch | 2.8.0 | 固定版本（CUDA 12.8） |
|  | torchvision | 0.23.0 | 视觉工具 |
|  | torchaudio | 2.8.0 | 音频处理 |
| **训练** | flash-attn | ≥2.8.3 | 优化注意力机制 |
|  | peft | ≥0.18.0 | 参数高效微调 |
|  | tensordict | ≥0.10.0 | 强化学习数据结构 |
|  | ray[default] | ≥0.1.0 | 分布式训练框架 |
| **模型加载** | sentence-transformers | ≥5.1.2 | 嵌入模型 |
| **分词器** | tiktoken | ≥0.12.0 | 快速分词 |
| **数据处理** | datasets | ≥4.4.1 | HuggingFace 数据集 |
|  | pandas | ≥2.3.3 | 数据处理 |
|  | pyarrow | ≥22.0.0 | 列式数据格式 |
|  | torchdata | ≥0.11.0 | 数据加载管道 |
| **文档处理** | mineru[core] | ≥2.6.4 | PDF → JSONL 处理流程 |
|  | pymupdf | ≥1.26.5 | PDF 解析 |
|  | rank-bm25 | ≥0.2.2 | 文本检索 |
| **Web API** | fastapi | ≥0.124.2 | REST API 框架 |
| **配置管理** | hydra-core | ≥1.3.2 | 配置管理 |
|  | pyyaml | ≥6.0.3 | YAML 解析 |
|  | python-dotenv | ≥1.1.1 | 环境变量 |
| **监控** | wandb | ≥0.23.1 | 实验跟踪 |
|  | codetiming | ≥1.4.0 | 性能分析 |
|  | tqdm | ≥4.67.1 | 进度条 |
| **构建工具** | hatchling | ≥1.28.0 | 构建后端 |
|  | editables | ≥0.5 | 可编辑安装 |
| **API 客户端** | openai | ≥2.6.0 | OpenAI API 支持 |

## 3. 安装指南

如果在使用 `uv sync` 安装依赖时遇到构建问题，请按照以下步骤进行。

首先，确认你要使用的 Python 版本，并删除 [pyproject.toml](../pyproject.toml) 中所有与 CUDA 相关的依赖。

```shell
requires-python = "==3.11.13"  ## 在此指定 Python 版本
dependencies = [
    # 核心 SDG 依赖
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
    "vllm>=0.11.0",            # 删除这一行，因为它依赖 CUDA

    # 训练依赖
    "pandas>=2.3.3",
    "codetiming>=1.4.0",
    "flash-attn>=2.8.3",       # 删除这一行，因为它依赖 CUDA
    "hydra-core>=1.3.2",
    "peft>=0.18.0",
    "pyarrow>=22.0.0",
    "ray[default]>=0.1.0",
    "tensordict>=0.10.0",
    "torchdata>=0.11.0",
    "wandb>=0.23.1",

    # 构建工具
    "editables>=0.5",
    "hatchling>=1.28.0",
]
```

#### 第一步 — 检查 CUDA 版本

在安装 PyTorch 或 vLLM 之前，请使用以下命令确认你的 CUDA 版本：

```shell
nvidia-smi
```

然后在 [pyproject.toml](../pyproject.toml) 中修改为你对应的 CUDA 版本。

``` shell
[[tool.uv.index]]
name = "pytorch-cu128"  ## 例如你的环境为 CUDA 12.6，则将下方所有 cu128 都改为 cu126
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }
```

#### 第二步 — 安装匹配的 PyTorch 版本

安装与你的 CUDA 版本匹配的 PyTorch。详情请参考 [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)。

``` shell
# 例如，如果你使用 CUDA 12.6，可以使用以下 uv 命令安装 torch 2.7.0
uv add torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
```

> [!Note]
> 如果你想尝试模型训练模块，你应该使用 CUDA>=12.8

#### 第三步 — 安装 CUDA 依赖库（必须在 PyTorch 之后安装）

vLLM 和 flash-attention 依赖于已安装的 PyTorch 和 CUDA 运行时。使用以下命令安装正确版本的 vLLM 和 flash-attention。

```shell
uv add flash-attn --no-build-isolation
uv add vllm
```

#### 第四步 - 安装通用依赖（与 CUDA 无关）

```shell
uv sync
```
