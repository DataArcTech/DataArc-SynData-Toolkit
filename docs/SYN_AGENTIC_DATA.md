# Syn Agentic Data

Syn Agentic Data is an additive sidecar workflow for verifier-backed synthetic data. It reuses the toolkit's existing synthesis generators and adds a Loong-style executable verification wrapper.

## What It Adds

The workflow keeps the existing DataArc generation path intact:

- `few_shot`: uses `sdgsystem.distillation.sdg_distill.SDGDistillation`.
- `self_instruct`: uses `sdgsystem.distillation.self_instruct.SelfInstructDistillation`.
- `evol_instruct`: uses `sdgsystem.distillation.evol_instruct.EvolInstructDistillation`.

Those generators first produce regular DataArc-style candidate samples with `input` and `output`. A separate agentic completion step then converts each candidate into:

- `question`
- executable Python `rationale`
- `final_answer`
- `metadata`

The verifier executes `rationale` and accepts the sample only when the final printed line matches `final_answer`.

## Current Scope

The initial example is intentionally small and covers only three subsets:

- `finance`
- `programming`
- `mathematical_programming`

Each strategy generates one candidate per subset by default, so the smoke run produces 9 attempted records.

## Run

Set API credentials through environment variables. Do not place API keys in config files.

```bash
export API_KEY="<your-api-key>"
export BASE_URL="https://api.deepseek.com"
.venv/bin/python examples/syn_agentic_data/run.py configs/syn_agentic_data.yaml
```

The default model in `configs/syn_agentic_data.yaml` is `deepseek-v4-pro`.
`model_request_timeout` applies to every sidecar model call, including the reused few-shot, self-instruct, and evol-instruct generators.

## Outputs

The runner writes artifacts under `output_dir`:

- `candidate_results.json`: DataArc-style candidates from the existing generators.
- `baseline_results.json`: all agentic completion attempts.
- `accepted_results.json`: accepted attempts with verification metadata.
- `accepted_agentic_items.json`: accepted Loong-style records.
- `dataarc_train.jsonl`: accepted records converted back to DataArc `input` / `output`.
- `summary.json`: run counts and domain/strategy coverage.

## Design Notes

This contribution does not modify the main `sdg generate` CLI, pipeline, prompt constants, web UI, or existing distillation classes. After the sidecar path is reviewed, it can be wired into the main pipeline as an optional stage.
