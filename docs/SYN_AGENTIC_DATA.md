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
Environment-backed records can opt into a different verifier by setting metadata. For example,
`metadata.verifier: terminal_bench_harness` uses a sanitized external harness observation instead of executing
`rationale` as Python.

For `evol_instruct`, the sidecar explicitly generates both Evol-Instruct directions:

- `in_depth`
- `in_breadth`

The direction is recorded in candidate metadata and accepted agentic item metadata as `evol_direction`.
If a directional Evol candidate cannot be parsed, the sidecar keeps the run complete by falling back to the seed item for that direction and records `candidate_source: seed_fallback`.

## Current Scope

The default smoke config is intentionally small and covers only three subsets:

- `finance`
- `programming`
- `mathematical_programming`

`few_shot` and `self_instruct` each generate one candidate per subset by default. `evol_instruct` generates one `in_depth` candidate and one `in_breadth` candidate per subset, so the smoke run produces 12 attempted records.

## Run

Set API credentials through environment variables. Do not place API keys in config files.

```bash
export API_KEY="<your-api-key>"
export BASE_URL="https://api.deepseek.com"
.venv/bin/python examples/syn_agentic_data/run.py configs/syn_agentic_data.yaml
```

The default model in `configs/syn_agentic_data.yaml` is `deepseek-v4-pro`.
`model_request_timeout` applies to every sidecar model call, including the reused few-shot, self-instruct, and evol-instruct generators.

For the terminal-bench demo, provide the task path and Harbor credentials at runtime:

```bash
export TERMINAL_BENCH_QUERY_OPTIMIZE_PATH="/Volumes/PSSD/code/My-Agent/Agent-Benchmark/terminal-bench-2/query-optimize"
export HARBOR_OPENAI_BASE_URL="<your-openai-compatible-base-url>"
export HARBOR_API_KEY="<your-api-key>"
```

## Outputs

The runner writes artifacts under `output_dir`:

- `candidate_results.json`: DataArc-style candidates from the existing generators.
- `baseline_results.json`: all agentic completion attempts.
- `accepted_results.json`: accepted attempts with verification metadata.
- `accepted_agentic_items.json`: accepted Loong-style records.
- `dataarc_train.jsonl`: accepted records exported as multi-turn tool-call JSONL. Each line has `messages` and `tools`. `messages` uses typed content blocks such as `text`, `reasoning`, `function_call`, and `observation`. Python-rationale records use `execute_python`; environment-backed records can provide their own tool schema, such as `run_codex_terminal_task`.
- `summary.json`: run counts and domain/strategy coverage.

## Design Notes

This contribution does not modify the main `sdg generate` CLI, pipeline, prompt constants, web UI, or existing distillation classes. After the sidecar path is reviewed, it can be wired into the main pipeline as an optional stage.

## Terminal-Bench Task Synthesis

Terminal-bench-style synthesis is handled by a separate sidecar runner because each sample is a task directory, not a
single executable rationale. Seed tasks live under `examples/syn_agentic_data/terminal-bench-seed` and include:

- `portfolio-optimization`
- `cancel-async-tasks`
- `constraints-scheduling`

The terminal-bench sidecar asks the model to generate a coherent task artifact bundle:

- `instruction.md`: rewritten user-facing task description.
- `solution/solve.sh`: updated reference solution.
- `tests/test_outputs.py`: updated verifier tests.
- optional environment or fixture files when the variant needs them.

The sidecar copies the seed directory, strips terminal-bench canary comments from model context and generated artifacts,
applies the generated files, runs static validation, and exports `dataarc_train.jsonl` as `messages`/`tools` rows. The
training rows include task instructions, assistant reasoning, a `run_harbor_terminal_bench` tool call, and a sanitized
verification observation. They do not include solution files or verifier test source.

The same run also writes the generated task directories in terminal-bench organization under `tasks/<synthetic-task>/`
and a `terminal_bench_manifest.json` file. The manifest maps each accepted synthetic sample to its terminal-bench task
directory and key relative files (`instruction.md`, `task.toml`, `environment/Dockerfile`, `solution/solve.sh`, and
`tests/test_outputs.py`) without embedding solution or verifier source in the training rows.

Run a small synthesis job:

```bash
.venv/bin/python examples/syn_agentic_data/run_terminal_bench.py configs/syn_agentic_terminal_bench.yaml
```

For Harbor verification, use the terminal-bench Science environment and provide credentials through environment
variables:

```bash
export HARBOR_OPENAI_BASE_URL="<your-openai-compatible-base-url>"
export HARBOR_API_KEY="<your-api-key>"
python examples/syn_agentic_data/run_terminal_bench_harbor_smoke.py \
  outputs/syn_agentic_terminal_bench/terminal_bench_records.json \
  --index 0
```

The smoke script defaults to
`/Volumes/PSSD/code/My-Agent/Agent-Benchmark/terminal-bench-science/.venv/bin/harbor`, agent `codex`, model `gpt-5.5`,
Docker, and `--force-build` so generated tasks build from their local `environment/Dockerfile` instead of inheriting a
seed task's remote prebuilt image. It returns `status: skipped` when required environment variables are missing.

The equivalent manual Harbor command is:

```bash
/Volumes/PSSD/code/My-Agent/Agent-Benchmark/terminal-bench-science/.venv/bin/harbor run \
  --path <generated-task-path> \
  --jobs-dir <output-dir>/harbor_jobs \
  --agent codex \
  --model gpt-5.5 \
  --env docker \
  --force-build \
  --n-concurrent 1 \
  --n-tasks 1 \
  --yes \
  --agent-env OPENAI_BASE_URL="${HARBOR_OPENAI_BASE_URL}" \
  --agent-env OPENAI_API_KEY="${HARBOR_API_KEY}"
```
