# Third-Party Notices

DataArc SynData Toolkit is licensed under the [Apache License, Version 2.0](./LICENSE).
This file lists third-party software that is redistributed as part of this
repository, together with its license and attribution notices, as required by
Section 4 of the Apache License.

## Bundled source code

### verl (`verl/`)

- **Project:** verl: Volcano Engine Reinforcement Learning for LLMs
- **Upstream:** https://github.com/volcengine/verl
- **Bundled version:** `0.7.0.dev` (see `verl/version/version`)
- **License:** Apache License, Version 2.0 (identical to the [LICENSE](./LICENSE) file of this repository)
- **Upstream NOTICE:** `Copyright 2023-2024 Bytedance Ltd. and/or its affiliates`

The `verl/` directory is a vendored copy of the verl post-training framework,
used to provide the integrated SFT and GRPO training module. The copy may
include local modifications required for integration with DataArc SynData
Toolkit. All original per-file copyright and license headers have been
retained. Those headers attribute portions of the code to, among others:

- Bytedance Ltd. and/or its affiliates
- SGLang Team
- ModelBest Inc. and/or its affiliates
- EleutherAI and the HuggingFace Inc. team
- PRIME team and/or its affiliates
- Kakao Brain
- The vLLM team
- Amazon.com Inc and/or its affiliates
- Meituan Ltd. and/or its affiliates
- z.ai
- The SwissAI Initiative
- The Qwen Team and The HuggingFace Inc. team
- Search-R1 Contributors
- Individual contributors named in the respective files

All of the above portions are licensed under the Apache License, Version 2.0
unless a file header states otherwise.

## Dependencies installed at build or run time

Python packages declared in `pyproject.toml` (for example vLLM, PyTorch,
Transformers, DeepEval, MinerU, FastAPI) and JavaScript packages declared in
`sdgsystem/webui/package.json` are **not** redistributed in this repository.
They are downloaded from their respective package registries at install time
and remain subject to their own licenses.
