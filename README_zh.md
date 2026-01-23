<div align="center">

# DataArc SynData Toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: uv](https://img.shields.io/badge/Package_Manager-uv-42b983.svg)](https://github.com/astral-sh/uv)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-ffa000.svg)](https://docs.pydantic.dev/)

<p>
  <a href="https://discord.gg/u48SJ9HEbd">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&message=Community&color=7289da&logo=discord&logoColor=white&label=Discord&labelColor=1a1a2e">
  </a>
  <a href="https://github.com/DataArcTech/DataArc-SynData-Toolkit/issues/2">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&message=Group&color=07c160&logo=wechat&logoColor=white&label=WeChat&labelColor=1a1a2e">
  </a>
  <a href="https://x.com/DataArcTech">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&message=Home&color=000000&logo=x&logoColor=white&label=&labelColor=1a1a2e">
  </a>
  <a href="https://www.linkedin.com/company/dataarctech/">
    <img src="https://img.shields.io/static/v1?style=for-the-badge&message=Home&color=0077B5&logo=linkedin&logoColor=white&label=LinkedIn&labelColor=1a1a2e">
  </a>
</p>

*一个模块化、高度易用的合成数据生成工具集，支持多来源、多语言的数据合成。*

### 使用零代码[命令行](#rocket-快速开始)与[可视化界面](#desktop_computer-可视化界面)，轻松合成大模型训练数据！

:book:[ [English](./README.md) | **中文** ]

</div>

## :dart: 项目概述

**DataArc SynData Toolkit**是由[数创弧光](https://www.dataarctech.com/)与[粤港澳大湾区数字经济研究院](https://www.idea.edu.cn/)联合开发并开源的合成数据工具集，能够根据使用者需求，通过简单配置文件一步到位合成所需训练数据。

## :bulb: 项目特色

- **极简使用**：通过一个[简单指令](#3-合成数据)和一个配置文件合成数据。也提供[可视化界面](#desktop_computer-可视化界面)进行操作。
- **支持多来源获取数据**
  - **本地合成**：支持基于本地语料合成数据。
  - **huggingface集成**：支持基于需求自动爬取筛查huggingface数据集。
  - **模型蒸馏**：支持基于模型蒸馏合成数据。
- **集成模型后训练模块**：基于[verl](https://github.com/volcengine/verl)框架的端到端模型训练工作流，支持在合成数据上进行SFT和GRPO训练。
- **后训练模型评估**：使用[DeepEval](https://github.com/confident-ai/deepeval)框架评估训练后的模型。
- **支持多语言**：支持英语以及各类小语种。
- **支持多源模型接入**：支持通过本地部署、OpenAI接口等多种形式接入模型。
- **高度可扩展**：合成数据全流程模块化，开发者可灵活基于模块定制化策略和方法实现。

## :movie_camera: 演示

观看两分钟演示视频快速了解**DataArc SynData Toolkit**。

https://github.com/user-attachments/assets/4b4d5ae4-d274-4971-a3cb-e9f07e841374

## :microscope: 性能表现

| 模型                       | Medical | Finance | Law   |
|----------------------------|---------|---------|-------|
| Qwen-2.5-7B-Instruct       | 42.34%  | 52.91%   | 19.80% |
| Trained with Synthetic Data | 64.57%  | 73.93%  | 42.80% |

仅需少量代码即可带来超过20%的性能提升。

## :notebook: 更新日志

[25/11/17] 🎉我们开源了合成数据平台。  
[25/11/27] 增加了**并行处理模块**，可以大幅度降低合成数据所需时间。  
[25/11/28] 新增合成数据中间结果保存功能，支持**断点续跑**，无需从头重新生成，节省Token消耗。  
[25/12/25] 🔥重要更新：
- **前后端分离架构**：**DataArc SynData Toolkit**现采用完全前后端分离的架构，配备**FastAPI后端**（REST API + SSE实时进度流式推送）和独立的**React**前端，提升可视化、易用性和可扩展性。
- **基于verl的后训练支持**：引入集成的后训练模块，基于**verl**框架，支持在合成数据上进行**SFT**和**GRPO**的端到端模型训练工作流。
- **多语言扩展**：新增**阿拉伯语**数据集生成支持，利用阿拉伯语翻译模型生成完全本地化的合成数据输出。

[26/01/12] 🖼️ 新增图像模态支持：
- **图像模态本地任务**：使用VLM从本地图像或PDF提取的图表生成VQA（视觉问答）数据。
- **图像模态网络任务**：自动从HuggingFace搜索和获取图文数据集。

[26/01/xx] 📊 新增后训练模型评估：
- **DeepEval集成**：新增基于**DeepEval**框架的模型评估模块。
- **三大评估指标**：
  - **答案正确性**：将模型输出与标准答案进行比较，支持自定义评分标准。
  - **格式合规性**：评估模型输出是否遵循指定的输出格式要求。
  - **成对偏好比较**：比较后训练模型与基础模型，衡量训练效果提升。

> [!TIP]
>
> 如果您无法使用最新的功能，请尝试重新拉取代码

## :factory: DataArc SynData Toolkit 数据合成流程

**DataArc SynData Toolkit**的设计旨在以模块化方式运行数据合成流程，允许用户自定义各模块的策略和方法实现。主要组件包括：

- **数据合成**：通过本地合成、huggingface爬取、数据蒸馏等方法合成数据。
  - 开发者可以继承[BaseTaskConfig](./sdgsystem/configs/config.py)和[BaseTaskExecutor](./sdgsystem/tasks/base.py)定制化合成数据的方法
- **数据筛选与改写**：对初步合成的数据，针对待训练模型进行筛选和改写。
  - 开发者可以继承[BaseRewriteConfig](./sdgsystem/configs/config.py)和[BaseRewriter](./sdgsystem/generation/rewriter.py)定制化数据改写方法（或不改写）

![dataarc-sdg_pipeline](assets/dataarc-syndata-toolkit_pipeline.png)

通过解耦模块，开发者可以基于特定需求实现各功能模块的灵活定制化。

## :jigsaw: 使用场景

我们提供三个不同的使用**DataArc SynData Toolkit**进行数据合成的[使用场景](docs/USE_CASES_zh.md)。

## :file_folder: 项目结构

```
DataArc-SynData-Toolkit/
├── configs/                        # YAML配置样例
│   ├── sdg.yaml                    # SDG流程配置
│   ├── sft.yaml                    # SFT训练配置
│   ├── grpo.yaml                   # GRPO训练配置
│   └── eval.yaml                   # 模型评估配置
│
├── sdgsystem/                      # 核心系统
│   ├── app/                        # FastAPI后端 (REST + SSE)
│   ├── generation/                 # 数据生成
│   ├── documents/                  # 文本解析与检索
│   ├── huggingface/                # HuggingFace数据集集成
│   ├── distillation/               # 模型蒸馏合成
│   ├── tasks/                      # SDG执行任务
│   ├── evaluation/                 # 数据质量评估
│   ├── deepeval/                   # 后训练模型评估
│   ├── models/                     # 统一LLM接口与后处理
│   ├── trainer/                    # 模型后训练 (verl: SFT + GRPO)
│   ├── translation/                # 多语言支持
│   ├── webui/                      # React前端
│   ├── pipeline.py                 # 核心SDG流程
│   └── cli.py                      # 命令行入口
│
├── verl/                           # 集成的verl训练框架
├── docs/                           # 文档
├── pyproject.toml
└── README.md
```

## :rocket: 快速开始

### 1. 安装DataArc SynData Toolkit

```shell
# 1. 克隆项目仓库
git clone https://github.com/DataArcTech/DataArc-SynData-Toolkit.git
cd DataArc-SynData-Toolkit

# 2. 安装uv（如果尚未安装）
pip install uv

# 3. 安装依赖 
uv sync
```

具体的硬件要求以及环境配置请详见[配置文档](docs/DEPENDENCIES_zh.md)。

### 2. 配置

请参照[样例配置文件](./configs/sdg.yaml)，根据您的需求修改配置。

### 3. 合成数据

通过命令行：

创建.env文件，填写如下字段。
```shell
API_KEY=sk-xxx   # 你的api key
BASE_URL=https://api.openai.com/v1  # 可选：指定的base url
```

并执行如下命令。

```shell
uv run sdg generate configs/sdg.yaml  # 可以更改为你的.yaml文件
```

## :twisted_rightwards_arrows: 使用合成数据训练模型

**DataArc SynData Toolkit**集成了基于[verl](https://github.com/volcengine/verl)的端到端模型训练模块，支持直接在合成数据上训练模型。我们支持两种训练方法：**SFT（监督微调）**和**GRPO（群组相对策略优化）**。

### 通过命令行快速开始

#### 1. 准备配置文件

基于[SFT配置样例](./configs/sft.yaml)或[GRPO配置样例](./configs/grpo.yaml)创建训练配置文件。

#### 2. 运行训练

```shell
# SFT训练
uv run sdg train configs/sft.yaml

# GRPO训练
uv run sdg train configs/grpo.yaml
```

详细配置选项请参考样例YAML文件。

## :bar_chart: 评估后训练模型

**DataArc SynData Toolkit**提供了基于[DeepEval](https://github.com/confident-ai/deepeval)的模型评估模块，支持使用LLM-as-a-Judge（G-Eval）评估后训练模型。我们支持三种评估指标：**答案正确性**、**格式合规性**和**成对偏好比较**。

### 通过命令行快速开始

#### 1. 准备配置文件

基于[评估配置样例](./configs/eval.yaml)创建评估配置文件。

在.env文件中添加API密钥。

```shell
OPENAI_API_KEY=sk-xxx   # 你的OpenAI API密钥
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选：你的OpenAI base url
CONFIDENT_API_KEY=confident_us_xxx  # 你的Confident AI API密钥（用于访问DeepEval，注册后可免费创建）
```

#### 2. 运行评估

```shell
uv run sdg eval configs/eval.yaml
```

评估结果可在Confident AI提供的云端可视化界面查看，同时也会保存到配置的输出目录。

## :desktop_computer: 可视化界面

使用如下命令启动FastAPI后端。

```shell
uv run fastapi dev sdgsystem/app/main.py
```

打开另一个终端，输入如下命令启动前端。

```shell
cd sdgsystem/webui

# 安装依赖
pnpm install

# 启动开发服务器
pnpm dev
```

如果您对前端有任何疑问，可以查看我们的[前端文档](/sdgsystem/webui/README_zh.md)。

## :date: 下一步发布的计划

- **加密合成数据生成**：生成对隐私信息进行加密后的数据。

## :handshake: 欢迎贡献

我们欢迎对**DataArc SynData Toolkit**进行改进贡献！

