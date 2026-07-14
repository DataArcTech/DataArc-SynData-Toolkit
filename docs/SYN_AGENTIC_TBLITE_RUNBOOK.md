# Syn Agentic TBLite Runbook

This runbook documents the OpenThoughts-TBLite agentic synthesis and Terminus-2 evaluation workflow used for large-scale
experiments. It is intentionally a sidecar recipe: it does not change the main `sdg generate` CLI.

## Scope

The workflow covers five task sets:

- Original OpenThoughts-TBLite tasks.
- `few_shot` synthetic terminal-bench tasks.
- `self_instruct` synthetic terminal-bench tasks.
- `evol_instruct` in-depth synthetic terminal-bench tasks.
- `evol_instruct` in-breadth synthetic terminal-bench tasks.

For each synthetic method, the target is 100 verifier-backed accepted tasks. Accepted means the generated task bundle
passes gold-solution validation.

## Environment

Use the repository virtual environment for DataArc code and the terminal-bench/Harbor environment for Terminus-2:

```bash
cd /path/to/DataArc-SynData-Toolkit
uv sync

export TBLITE_ROOT="/path/to/OpenThoughts-TBLite"
export SYN_TBLITE_ROOT="/path/to/Syn-TBLite-agentic-v3"
export HARBOR_BIN="/path/to/terminal-bench-science/.venv/bin/harbor"
export HARBOR_PYTHON="/path/to/terminal-bench-science/.venv/bin/python"
export API_KEY="<your-openai-compatible-api-key>"
export OPENAI_API_KEY="${API_KEY}"
export BASE_URL="https://api.deepseek.com"
```

Do not commit API keys. Prefer `read -s API_KEY` in interactive shells for long runs.

## 1. Synthesize Terminal-Bench Tasks

The underlying synthesis entry point is:

```bash
python "${TBLITE_ROOT}/scripts/synthesize_syn_tblite.py" \
  --source-root "${TBLITE_ROOT}" \
  --output-root "${SYN_TBLITE_ROOT}" \
  --dataarc-root "$(pwd)" \
  --model deepseek-v4-flash \
  --base-url "${BASE_URL}" \
  --completion-max-tokens 20000 \
  --request-timeout 600 \
  --api-retries 10 \
  --api-retry-delay 30 \
  --sample-attempts 4 \
  --repair-attempts 4 \
  --gold-validation-retries 2 \
  --gold-validation-retry-delay 30 \
  --strategies few_shot self_instruct evol_instruct \
  --evol-directions in_depth in_breadth
```

For unattended runs, use a small supervisor that repeatedly launches the same command until each method has at least 100
accepted records. The supervisor should count accepted records from:

```text
${SYN_TBLITE_ROOT}/<method>/terminal_bench_accepted.json
```

and should not treat failed records as completed. The synthesis script only skips records whose seed/method/direction has
already passed the configured validation mode, so failed attempts can be retried in later rounds.

Recommended background pattern:

```bash
mkdir -p "${SYN_TBLITE_ROOT}/logs"
nohup python "${SYN_TBLITE_ROOT}/resume_synthesis_to_100.py" \
  > "${SYN_TBLITE_ROOT}/logs/resume_synthesis_supervisor_$(date +%Y%m%d_%H%M%S).log" \
  2>&1 &
```

The experiment supervisor used four workers:

- `few_shot`
- `self_instruct`
- `evol_instruct_in_depth`
- `evol_instruct_in_breadth`

Each method writes:

- `terminal_bench_records.json`: all generated attempts.
- `terminal_bench_accepted.json`: accepted verifier-backed records.
- `tasks/`: generated terminal-bench task directories, including retry/repair intermediates.

Progress check:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SYN_TBLITE_ROOT"])
for method in ["few_shot", "self_instruct", "evol_instruct_in_depth", "evol_instruct_in_breadth"]:
    method_dir = root / method
    records = json.loads((method_dir / "terminal_bench_records.json").read_text())
    accepted = json.loads((method_dir / "terminal_bench_accepted.json").read_text())
    verified = sum(bool((row.get("verification") or {}).get("matched")) for row in records)
    gold = sum(
        bool((row.get("verification") or {}).get("matched"))
        and (row.get("verification") or {}).get("status") == "gold_solution_validated"
        for row in records
    )
    task_dirs = sum(1 for path in (method_dir / "tasks").iterdir() if path.is_dir())
    print(method, "records", len(records), "verified", verified, "gold", gold, "accepted", len(accepted), "task_dirs", task_dirs)
PY
```

## 2. Prepare Accepted-Only Evaluation Roots

Terminus-2 evaluation should run only on accepted records. Do not point Harbor at the raw `tasks/` directory, because it
contains failed and intermediate repair attempts.

Create one evaluation root per method by symlinking the first 100 accepted task paths:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

syn_root = Path(os.environ["SYN_TBLITE_ROOT"])
methods = ["few_shot", "self_instruct", "evol_instruct_in_depth", "evol_instruct_in_breadth"]

for method in methods:
    accepted = json.loads((syn_root / method / "terminal_bench_accepted.json").read_text())
    if len(accepted) < 100:
        raise SystemExit(f"{method} has only {len(accepted)} accepted records")

    eval_root = syn_root / "eval/synthetic_verified100_task_roots" / method
    eval_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(accepted[:100], start=1):
        task_path = Path(record["task_path"])
        link_path = eval_root / f"{index:03d}-{task_path.name}"
        target = os.path.relpath(task_path, start=eval_root)
        if not link_path.exists():
            link_path.symlink_to(target, target_is_directory=True)
        manifest.append({"index": index, "eval_task_dir": str(link_path), "source_task_path": str(task_path)})

    (eval_root / "synthetic_verified100_manifest.json").write_text(
        json.dumps({"method": method, "num_tasks": len(manifest), "tasks": manifest}, indent=2),
        encoding="utf-8",
    )
PY
```

## 3. Run Terminus-2 on Synthetic Tasks

Run each accepted-only root with Harbor. Long jobs should use `nohup`.

```bash
for method in few_shot self_instruct evol_instruct_in_depth evol_instruct_in_breadth; do
  task_root="${SYN_TBLITE_ROOT}/eval/synthetic_verified100_task_roots/${method}"
  jobs_dir="${SYN_TBLITE_ROOT}/eval/synthetic_verified100_terminus2_deepseek_flash/${method}/jobs"
  logs_dir="${SYN_TBLITE_ROOT}/eval/synthetic_verified100_terminus2_deepseek_flash/${method}/logs"
  mkdir -p "${jobs_dir}" "${logs_dir}"

  nohup "${HARBOR_PYTHON}" "${HARBOR_BIN}" run \
    --path "${task_root}" \
    --jobs-dir "${jobs_dir}" \
    --agent terminus-2 \
    --model openai/deepseek-v4-flash \
    --agent-kwarg api_base="${BASE_URL}" \
    --agent-kwarg collect_rollout_details=True \
    --agent-kwarg store_all_messages=True \
    --agent-include-logs "**/*" \
    --env docker \
    --force-build \
    --n-concurrent 1 \
    --n-tasks 100 \
    --max-retries 2 \
    --yes \
    > "${logs_dir}/${method}_terminus2_$(date +%Y%m%d_%H%M%S).log" \
    2>&1 &
done
```

Each trial writes its ATIF trajectory to:

```text
<jobs-dir>/<job-timestamp>/<trial-name>/agent/trajectory.json
```

The verifier reward is:

```text
<jobs-dir>/<job-timestamp>/<trial-name>/verifier/reward.txt
```

## 4. Run Terminus-2 on Original Tasks

The original 100 tasks can be evaluated directly with Harbor:

```bash
nohup "${HARBOR_PYTHON}" "${HARBOR_BIN}" run \
  --path "${TBLITE_PREPARED_ROOT}" \
  --jobs-dir "${SYN_TBLITE_ROOT}/eval/original_t100_terminus2_deepseek_flash/jobs" \
  --agent terminus-2 \
  --model openai/deepseek-v4-flash \
  --agent-kwarg api_base="${BASE_URL}" \
  --agent-kwarg collect_rollout_details=True \
  --agent-kwarg store_all_messages=True \
  --agent-include-logs "**/*" \
  --env docker \
  --force-build \
  --n-concurrent 1 \
  --n-tasks 100 \
  --max-retries 2 \
  --yes \
  > "${SYN_TBLITE_ROOT}/eval/original_t100_terminus2_deepseek_flash/original_t100.log" \
  2>&1 &
```

If a subset fails because fixtures are Git LFS pointers, fix the prepared copy before rerunning the missing subset:

```bash
git -C "${TBLITE_ROOT}" lfs smudge < pointer-file > real-file
```

For the recorded run, `playing-card-recognition` required this treatment because its PNG files were LFS pointers in the
prepared task copy.

## 5. Inspect Job Status

Harbor writes aggregate status to `result.json`:

```bash
python - <<'PY'
import json
from pathlib import Path

job = Path("<jobs-dir>/<job-timestamp>")
stats = json.loads((job / "result.json").read_text()).get("stats")
print(json.dumps(stats, indent=2, ensure_ascii=False))
PY
```

Summarize rewards for a completed job:

```bash
python - <<'PY'
import json
import re
from collections import Counter
from pathlib import Path

job = Path("<jobs-dir>/<job-timestamp>")

def reward(trial):
    path = trial / "verifier/reward.txt"
    return None if not path.exists() else float(path.read_text().strip())

def exception_type(trial):
    path = trial / "exception.txt"
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|CancelledError))", text)
    return match.group(1) if match else "Exception"

trials = [path for path in job.iterdir() if path.is_dir() and (path / "config.json").exists()]
rewards = [reward(trial) for trial in trials]
exceptions = Counter(exception_type(trial) for trial in trials if exception_type(trial))
print(
    {
        "n": len(trials),
        "reward_1": sum(value == 1.0 for value in rewards),
        "partial": sum(value not in (None, 0.0, 1.0) for value in rewards),
        "reward_0": sum(value == 0.0 for value in rewards),
        "missing_reward": sum(value is None for value in rewards),
        "strict_mean": sum(value or 0.0 for value in rewards) / len(trials),
        "exceptions": dict(exceptions),
    }
)
PY
```

## 6. Recorded Run Results

The run that produced the current artifacts used:

- `deepseek-v4-flash` for synthesis.
- Harbor Terminus-2 with `openai/deepseek-v4-flash`.
- `n_tasks=100`, `n_concurrent=1`, `max_retries=2`.

Final synthesis counts:

| Method | Records | Accepted / Verified / Gold | Task dirs |
|---|---:|---:|---:|
| `few_shot` | 121 | 100 | 210 |
| `self_instruct` | 113 | 100 | 194 |
| `evol_instruct_in_depth` | 116 | 100 | 215 |
| `evol_instruct_in_breadth` | 107 | 100 | 174 |

Final Terminus-2 results:

| Task set | N | reward=1 | partial | reward=0 | missing reward | strict mean | clean pass | errored | trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 100 | 62 | 4 | 29 | 5 | 0.6495 | 56 | 15 | 98 |
| `few_shot` | 100 | 71 | 1 | 11 | 17 | 0.7160 | 68 | 22 | 86 |
| `self_instruct` | 100 | 67 | 1 | 16 | 16 | 0.6770 | 65 | 22 | 89 |
| `evol_instruct_in_depth` | 100 | 65 | 0 | 21 | 14 | 0.6500 | 61 | 21 | 88 |
| `evol_instruct_in_breadth` | 100 | 65 | 1 | 18 | 16 | 0.6567 | 61 | 23 | 88 |
