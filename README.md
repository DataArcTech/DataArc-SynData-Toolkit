# DataArc SynData Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: uv](https://img.shields.io/badge/Package_Manager-uv-42b983.svg)](https://github.com/astral-sh/uv)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-ffa000.svg)](https://docs.pydantic.dev/)

*A modular, highly user-friendly synthetic data generation toolkit supporting multi-source, multi-language data synthesis.*

### Easily synthesize training data for LLMs with zero-code [CLI](#:rocket:-Quick-Start) and [GUI](#:desktop_computer:-Synthesizing-Data-with-GUI) !

:book: [ **English** | [中文](./README_zh.md) ]

## :dart: Project Overview

**DataArc SynData Toolkit** is a synthetic data generation toolkit developed and open-sourced by DataArcTech (https://www.dataarctech.com/) and International Digital Economy Academy (https://www.idea.edu.cn/). It enables users to generate customized training data in one step through simple configuration files based on their requirements.

## :bulb: Key Features

- **Extremely Simple Usage**: Synthesize data with [a single command](#3-Synthesize-Data) and a configuration file. [GUI](##:desktop_computer:-Synthesizing-Data-with-GUI) is also provided for easy operations.
- **Support for Multi-Source Synthetic Data**:
  - **Local Synthesis**: Support for generating data based on local corpora.
  - **Huggingface Integration**: Automatically screens and retrieves data from Huggingface.
  - **Model Distillation**: Enable synthetic data generation through model distillation.
- **Integrated Post-Training Module**: End-to-end model training workflows powered by verl, supporting SFT and GRPO.
- **Multilingual Support**: Supports English and various low-resource languages.
- **Multi-Provider Model Support**: Works with local deployment, OpenAI APIs, and more.
- **Highly Extensible**: The entire synthetic data workflow is modular, allowing developers to flexibly customize them.

## :movie_camera: Demo

We provide a highly user friendly GUI for everything. Watch a two-minute demo to understand **DataArc SynData Toolkit**.

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
[25/12/xx] 🔥Major upgrade:
- **Frontend–Backend Separation**: **DataArc SynData Toolkit** now adopts a fully frontend–backend separated architecture, featuring a **FastAPI backend** (REST APIs + SSE streaming for real-time progress) and a standalone **React** frontend for improved visualization, usability, and scalability.
- **Post-Training Support via verl**: Introduced an integrated post-training module powered by **verl**, enabling end-to-end model training workflows including **SFT** and **GRPO** on synthesized data.
- **Multilingual Expansion**: Added support for generating **Arabic** datasets, leveraging an Arabic translation model to produce fully localized synthetic data outputs.

> [!TIP]
>
> If you cannot use the latest feature, please pull the latest code.

## :factory: DataArc SynData Toolkit Pipeline

**DataArc SynData Toolkit** is designed to synthesize data in a modular pipeline, allowing users to customize the strategies and implementation methods of each step. The main components include:

- **Synthetic Data Generation**: Generate data through methods such as local synthesis, Huggingface dataset retrieval, and model distillation.
  - Developers can inherit [BaseTaskConfig](./sdgsystem/configs/config.py) and [BaseTaskExecutor](./sdgsystem/tasks/base.py) to customize the generation task.
- **Data Filtering and Rewriting**: Filter and rewrite initially synthesized data according to the target model's requirements.
  - Developers can inherit [BaseRewriteConfig](./sdgsystem/configs/config.py) and [BaseRewriter](./sdgsystem/generation/rewriter.py) to customize the rewrite method for synthetic data (or no rewriting).

![dataarc-sdg_pipeline](assets/dataarc-syndata-toolkit_pipeline.png)

By decoupling modules, developers can achieve flexible customization of functional modules based on specific needs.

## :jigsaw: Use Cases

We provide [three different use cases](docs/USE_CASES.md) that sythesize data through **DataArc SynData Toolkit**.

## :file_folder: Project Structure

```
DataArc-SynData-Toolkit/
├── configs/                        # Configuration Examples
│   ├── example.yaml                # SDG configuration example
│   ├── sft_example.yaml            # SFT training configuration
│   └── grpo_example.yaml           # GRPO training configuration
│
├── sdgsystem/                      # Core Implementation
│   ├── app/                        # FastAPI Backend (REST + SSE)
│   │   ├── api/                    # API endpoints
│   │   │   ├── jobs.py             # job management endpoints
│   │   │   ├── schemas.py          # Pydantic schemas
│   │   │   └── router.py           # API router
│   │   ├── core/                   # Core backend components
│   │   │   ├── job_manager.py      # job lifecycle management
│   │   │   ├── progress.py         # progress reporter for SSE
│   │   │   └── sse.py              # Server-Sent Events utilities
│   │   ├── services/               # Business logic services
│   │   │   └── sdg_service.py      # SDG pipeline service wrapper
│   │   └── main.py                 # FastAPI application entry
│   │
│   ├── configs/                    # Configuration Module
│   │   ├── config.py               # configuration parsing
│   │   └── constants.py            # default arguments
│   │
│   ├── dataset/                    # Dataset Module
│   │   ├── dataset.py              # dataset class
│   │   └── process.py              # quality control and formatting
│   │
│   ├── distillation/               # Model Distillation
│   │   ├── base.py                 # base distillation class
│   │   ├── sdg_distill.py          # SDG distillation implementation
│   │   ├── self_instruct.py        # self-instruct method
│   │   └── evol_instruct.py        # evol-instruct method
│   │
│   ├── documents/                  # Document Processing
│   │   ├── load.py                 # document loading
│   │   ├── parse.py                # document parsing
│   │   ├── chunk.py                # text chunking
│   │   └── retrieve.py             # passage retrieval (BM25)
│   │
│   ├── evaluation/                 # Evaluation Module
│   │   ├── answer_comparison.py    # answer comparison methods
│   │   └── evaluator.py            # sample evaluator
│   │
│   ├── generation/                 # Generation Module
│   │   ├── base.py                 # base generator with validation
│   │   ├── generator.py            # data generator
│   │   └── rewriter.py             # data rewriter
│   │
│   ├── huggingface/                # HuggingFace Integration
│   │   └── crawl.py                # dataset crawling from HF
│   │
│   ├── models/                     # Model Interaction Module
│   │   ├── postprocess/            # response postprocessing
│   │   │   ├── majority_voting.py  # majority voting implementation
│   │   │   └── processor.py        # postprocessor orchestration
│   │   ├── answer_extraction.py    # answer extraction from responses
│   │   ├── client.py               # unified model client
│   │   ├── models.py               # model deployment adapters
│   │   ├── processor_arguments.py  # postprocessor arguments
│   │   └── usage_counter.py        # token/time usage tracking
│   │
│   ├── tasks/                      # Task Execution Module
│   │   ├── base.py                 # base executor class
│   │   ├── local.py                # local document-based task
│   │   ├── web.py                  # HuggingFace web task
│   │   ├── distill.py              # distillation task
│   │   └── task_executor.py        # unified task executor
│   │
│   ├── trainer/                    # Model Training Module (verl)
│   │   ├── methods/                # training method implementations
│   │   │   ├── sft.py              # SFT training method
│   │   │   └── grpo.py             # GRPO training method
│   │   ├── config.py               # training configuration
│   │   ├── data_preprocessing.py   # training data preprocessing
│   │   └── launcher.py             # training job launcher
│   │
│   ├── translation/                # Multilingual Support
│   │   └── translator.py           # translation utilities
│   │
│   ├── webui/                      # React Frontend
│   │
│   ├── buffer.py                   # checkpoint/buffer management
│   ├── cli.py                      # CLI entry point
│   ├── parallel.py                 # parallel processing utilities
│   ├── pipeline.py                 # main SDG pipeline
│   ├── prompts.py                  # LLM prompts
│   └── utils.py                    # utility functions
│
├── verl/                           # verl Training Framework
│
├── docs/                           # Documentation
│
├── pyproject.toml                  # project dependencies
└── README.md                       # project documentation
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

Please refer to the [example configuration file](./configs/example.yaml) and modify the configuration based on your requirements.

### 3. Synthesize Data

Run through CLI: 

Create a .env file and specified the following fields.

```shell
API_KEY=sk-xxx   # your api key
BASE_URL=https://api.openai.com/v1  # Optional: your base url
```

And run following command.

```shell
uv run sdg generate configs/example.yaml  # or change to your .yaml file
```

## :twisted_rightwards_arrows: Training with Synthesized Data

**DataArc SynData Toolkit** integrates an end-to-end model training module powered by [verl](https://github.com/volcengine/verl), enabling you to train models directly on your synthesized data. We support two training methods: **SFT (Supervised Fine-Tuning)** and **GRPO (Group Relative Policy Optimization)**

### Quick Start with CLI

#### 1. Prepare Your Configuration

Create a training configuration file based on the [SFT Configuration Example](./configs/sft_example.yaml) or [GRPO Configuration Example](./configs/grpo_example.yaml).

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

- **Multi-modal Dataset Synthesizing**: Support synthesize data through image.

## :handshake: Contributing

We welcome contributions!
