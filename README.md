# DataArc SynData Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: uv](https://img.shields.io/badge/Package_Manager-uv-42b983.svg)](https://github.com/astral-sh/uv)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-ffa000.svg)](https://docs.pydantic.dev/)

*A modular, highly user-friendly synthetic data generation toolkit supporting multi-source, multi-language data synthesis.*

### Easily synthesize training data for LLMs with zero-code [CLI](#rocket-quick-start) and [GUI](#desktop_computer-run-with-gui) !

:book: [ **English** | [中文](./README_zh.md) ]

## :dart: Project Overview

**DataArc SynData Toolkit** is a synthetic data generation toolkit developed and open-sourced by DataArcTech (https://www.dataarctech.com/) and International Digital Economy Academy (https://www.idea.edu.cn/). It enables users to generate customized training data in one step through simple configuration files based on their requirements.

## :bulb: Key Features

- **Extremely Simple Usage**: Synthesize data with [a single command](#3-synthesize-data) and a configuration file. [GUI](#desktop_computer-run-with-gui) is also provided for easy operations.
- **Support for Multi-Source Synthetic Data**:
  - **Local Synthesis**: Support for generating data based on local corpora.
  - **Huggingface Integration**: Automatically crawl and filter data from Huggingface.
  - **Model Distillation**: Enable synthetic data generation through model distillation.
- **Integrated Post-Training Module**: End-to-end model training workflows powered by verl, supporting SFT and GRPO.
- **Multilingual Support**: Supports English and various low-resource languages.
- **Multi-Provider Model Support**: Works with local deployment, OpenAI APIs, and more.
- **Highly Extensible**: The entire synthetic data workflow is modular, allowing developers to flexibly customize them.

## :movie_camera: Demo

Watch our 2-minute demo to experience how **DataArc SynData Toolkit** works in practice.

https://github.com/user-attachments/assets/4b4d5ae4-d274-4971-a3cb-e9f07e841374

## :microscope: Performance

| Model                       | Medical | Finance | Law    |
|-----------------------------|---------|---------|--------|
| Qwen-2.5-7B-Instruct        | 42.34%  | 52.91%  | 19.80% |
| Trained with Synthetic Data | 64.57%  | 73.93%  | 42.80% |

A few lines of code deliver over 20% performance improvements.

## :notebook: Changelog

[25/11/17] 🎉We open-sourced our synthetic data platform.  
[25/11/27] We added **parallel processing module** to significantly accelerate the synthetic data generation pipeline.  
[25/11/28] We added **intermediate result saving**, allowing users to resume from the last successful stage** instead of restarting the entire pipeline — a major **token saver**.  
[25/12/25] 🔥Major upgrade:
- **Frontend–Backend Separation**: **DataArc SynData Toolkit** now adopts a fully frontend–backend separated architecture, featuring a **FastAPI backend** (REST APIs + SSE streaming for real-time progress) and a standalone **React** frontend for improved visualization, usability, and scalability.
- **Post-Training Support via verl**: Introduced an integrated post-training module powered by **verl**, enabling end-to-end model training workflows including **SFT** and **GRPO** on synthesized data.
- **Multilingual Expansion**: Added support for generating **Arabic** datasets, leveraging an Arabic translation model to produce fully localized synthetic data outputs.
[26/01/xx] 🖼️ **Image Modality Support**:
- **Image Local Task**: Generate VQA (Visual Question Answering) data from local images or PDF-extracted figures using VLMs. Supports automatic context extraction from surrounding document text.
- **Image Web Task**: Automatically search and retrieve image-text datasets from HuggingFace Hub, with intelligent field mapping and quality scoring based on task instructions.
- **Multi-source Image Input**: Support for user-uploaded images, PDF figure extraction via MinerU, and HuggingFace dataset streaming.

> [!TIP]
>
> If you cannot use the latest feature, please pull the latest code.

## :factory: DataArc SynData Toolkit Pipeline

**DataArc SynData Toolkit** is designed to synthesize data in a modular pipeline, allowing users to customize the strategies and implementation methods of each step. The main components include:

- **Synthetic Data Generation**: Generate data through methods such as local synthesis, Huggingface dataset retrieval, and model distillation.
  - Developers can inherit [BaseTaskConfig](./sdgsystem/configs/sdg.py) and [BaseTaskExecutor](./sdgsystem/tasks/base.py) to customize the generation task.
- **Data Filtering and Rewriting**: Filter and rewrite initially synthesized data according to the target model's requirements.
  - Developers can inherit [BaseRewriteConfig](./sdgsystem/configs/sdg.py) and [BaseRewriter](./sdgsystem/generation/rewriter.py) to customize the rewrite method for synthetic data (or no rewriting).

![dataarc-sdg_pipeline](assets/dataarc-syndata-toolkit_pipeline.png)

By decoupling modules, developers can achieve flexible customization of functional modules based on specific needs.

## :jigsaw: Use Cases

We provide [three different use cases](docs/USE_CASES.md) that sythesize data through **DataArc SynData Toolkit**.

## :file_folder: Project Structure

```
DataArc-SynData-Toolkit/
├── configs/                        # YAML configuration examples
│   ├── sdg.yaml                    # SDG pipeline config
│   ├── sft.yaml                    # SFT training config
│   └── grpo.yaml                   # GRPO training config
│
├── sdgsystem/                      # Core System
│   ├── app/                        # FastAPI backend (REST + SSE)
│   ├── generation/                 # Data generation
│   ├── documents/                  # File parsing & retrieval
│   ├── huggingface/                # HF dataset integration
│   ├── distillation/               # Model distillation synthesis
│   ├── tasks/                      # SDG execution tasks
│   ├── evaluation/                 # Quality evaluation
│   ├── models/                     # Unified LLM interface & postprocess
│   ├── trainer/                    # Post-training (verl: SFT + GRPO)
│   ├── translation/                # Multilingual support
│   ├── webui/                      # React frontend
│   ├── pipeline.py                 # Core SDG pipeline
│   └── cli.py                      # CLI entry
│
├── verl/                           # Integrated verl framework
├── docs/                           # Documentation
├── pyproject.toml
└── README.md
```

## :rocket: Quick Start

### 1. Install DataArc SynData Toolkit

```shell
# 1. Clone the repository
git clone https://github.com/DataArcTech/DataArc-SynData-Toolkit.git
cd DataArc-SynData-Toolkit

# 2. Install uv if not already installed
pip install uv

# 3. Install dependencies 
uv sync
```

For hardware requirements and dependencies detail, please refer to [dependency and installation guide](/docs/DEPENDENCIES.md).

### 2. Configuration

Please refer to the [example configuration file](./configs/sdg.yaml) and modify the configuration based on your requirements.

### 3. Synthesize Data

Run through CLI: 

Create a .env file and specified the following fields.

```shell
API_KEY=sk-xxx   # your api key
BASE_URL=https://api.openai.com/v1  # Optional: your base url
```

And run following command.

```shell
uv run sdg generate configs/sdg.yaml  # or change to your .yaml file
```

## :twisted_rightwards_arrows: Training with Synthesized Data

**DataArc SynData Toolkit** integrates an end-to-end model training module powered by [verl](https://github.com/volcengine/verl), enabling you to train models directly on your synthesized data. We support two training methods: **SFT (Supervised Fine-Tuning)** and **GRPO (Group Relative Policy Optimization)**

### Quick Start with CLI

#### 1. Prepare Your Configuration

Create a training configuration file based on the [SFT Configuration Example](./configs/sft.yaml) or [GRPO Configuration Example](./configs/grpo.yaml).

#### 2. Run Training

```shell
# SFT training
uv run sdg train configs/sft.yaml

# GRPO training
uv run sdg train configs/grpo.yaml
```

For detailed configuration options, refer to the example YAML files.

## :desktop_computer: Run with GUI

Start FastAPI server with following command.

```shell
uv run fastapi dev sdgsystem/app/main.py
```

Open another terminal and build frontend with following command.

```shell
cd sdgsystem/webui

# Install dependencies
pnpm install

# Start development server
pnpm dev
```

If you have any doubt about regrading our Web UI, check our [Web UI document](/sdgsystem/webui/README.md).

## :date: Schedule for the Next Release

- **Synthetic Data Evaluating**: Support generated dataset evaluation in real-time.

## :handshake: Contributing

We welcome contributions!
