# DataArc SynData Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Framework: uv](https://img.shields.io/badge/Package_Manager-uv-42b983.svg)](https://github.com/astral-sh/uv) [![Gradio UI](https://img.shields.io/badge/GUI-Gradio-ff6f00.svg)](https://github.com/gradio-app/gradio) [![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-ffa000.svg)](https://docs.pydantic.dev/)

*A modular, highly user-friendly synthetic data generation toolkit supporting multi-source, multi-language data synthesis.*

### Easily synthesize training data for LLMs with zero-code [CLI](#:rocket:-Quick-Start) and [GUI](#:desktop_computer:-Synthesizing-Data-with-GUI) !

:book: [ **English** | [中文](./README_zh.md) ]

  

## :dart: Project Overview

**DataArc SynData Toolkit** is a synthetic data generation toolkit developed and open-sourced by DataArc. It enables users to generate customized training data in one step through simple configuration files based on their requirements.



## :bulb: Key Features

- **Extremely Simple Usage**: Synthesize data with [a single command](#3-Synthesize-Data) and a configuration file. [Gradio UI](##:desktop_computer:-Synthesizing-Data-with-GUI) is also provided for easy operations.
- **Support for Multi-Source Synthetic Data**:
  - **Local Synthesis**: Support for generating data based on local corpora.
  - **Huggingface Integration**: Automatically screens and retrieves data from Huggingface.
  - **Model Distillation**: Enable synthetic data generation through model distillation.
- **Multilingual Support**: Supports English and various low-resource languages.
- **Multi-Provider Model Support**: Works with local deployment, OpenAI APIs, and more.
- **Highly Extensible**: The entire synthetic data workflow is modular, allowing developers to flexibly customize them.


## 🔬 Performance

| Model                       | Medical | Finance | Law   |
|----------------------------|---------|---------|-------|
| Qwen-2.5-7B-Instruct       | 42.34%  | 52.91%   | 19.80% |
| Trained with Synthetic Data | 64.57%  | 73.93%  | 42.80% |

A few lines of code deliver over 20% performance improvements.

## :notebook: Changelog

[25/11/17] We open-sourced our synthetic data platform.

[25/11/27] We added **parallel processing** to significantly accelerate the synthetic data generation pipeline.

[25/11/28] We added **intermediate result saving**, allowing users to resume from the last successful stage instead of restarting the entire pipeline — a major **token saver**.

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

We provide [three different use cases](examples/README.md) that sythesize data through **DataArc SynData Toolkit**.



## :file_folder: Project Structure

```
dataarc-sdg/
├── configs/						# Configuration Examples
│   ├── example.yaml				# example YAML file
|
├── sdgsystem/						# Implementation of Functions
│   ├── configs/					# Configuration Module
|	|	├── config.py				# configuration parsing
|	|	└── constants.py			# default arguments
|
│   ├── dataset/					# Dataset Module
|	|	├── dataset.py				# dataset class
|	|	└── process.py				# quality control and formatting
|
│   ├── huggingface/				# Huggingface Crawling
│   ├── documents/					# Retrieve/Parsing/Chunk of Local Corpora
│   ├── distillation/				# Model Distillation
|
│   ├── evaluation/					# Evaluation Module
|	|	├── answer_comparison.py	# answer comparison
|	|	├── evaluator.py			# evaluator
|
│   ├── generation/					# Generation Module
|	|	├── base.py					# base class of generation
|	|	├── generator.py			# data generator
|	|	├── rewriter.py				# data rewriter
|
│   ├── models/						# Model Interaction Module
|	|	├── postprocess/			# postprocess of model responses (e.g. majority voting)
|	|	├── answer_extraction.py	# answer extraction from responses
|	|	├── models.py				# model deployment and chatting
|	|	├── processor_arguments.py	# arguments of post-processor
|	|	├── client.py				# client for interacting with models
|
│   ├── tasks/						# Generation Task Execution Module
|	|	├── base.py					# base class of executor
|	|	├── (local/web/distill).py	# executor for different sources/route
|	|	├── total_executor.py		# total executor
|
│   ├── translation/				# Support for Low-Resource Languages
|
│   ├── cli.py						# API for project functions
│   ├── pipeline.py					# pipeline of data synthesis
│   ├── prompts.py					# prompts used in project
│   ├── token_counter.py			# token usage estimation
│   └── utils.py					# other function utils
|
├── train/
│   ├── datasets/					# training datasets
|	|	├── examples/				  # training examples
|	|	├── scripts/				  # scripts
|	|	└── verl/			        # training module by verl
|
├── tests/							# Test Suite
|
├── app.py							# gradio UI
├── pyproject.toml					# project dependencies
└── README.md						# project documentation
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

For hardware requirements and dependencies detail, please refer to [dependency and installation guide](DEPENDENCIES.md).

### 2. Configuration

Please refer to the [example configuration file](./configs/example.yaml) and modify the configuration based on your requirements.

### 3. Synthesize Data

Run through CLI: 

Create a .env file and specified the following fields.

```shell
OPENAI_API_KEY=sk-xxx   # your api key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: your base url
```

And run following command.

```shell
uv run sdg configs/example.yaml  # or change to your .yaml file
```

## :twisted_rightwards_arrows: Training with Synthesized Data

Prepare your synthesized data at datasets/. Here is an example of LoRA fine-tuning with gsm8k dataset on Qwen2.5-0.5B:

```shell
cd train
bash examples/sft/gsm8k/run_qwen_05_peft.sh
```

## :desktop_computer: Synthesizing Data with GUI

The UI is powered by [Gradio](https://github.com/gradio-app/gradio). Build with following command.

```shell
uv run python app.py
```

![](./assets/frontend.png)



## :wrench: Configuration System

DataArc-SDG is configured using a flexible YAML file, please check our provided [example yaml file](configs/example.yaml).



## :date: Schedule for the Next Release

- **Arabic Support**: Support for generating Arabic synthetic data.
- **Custom Data Sources**: Support for custom addition of data sources and corresponding protocol file conversion.
- **Model Fine-tuning Module**: Support fine-tuning models using synthetic data within the pipeline.



## :handshake: Contributing

We welcome contributions!
