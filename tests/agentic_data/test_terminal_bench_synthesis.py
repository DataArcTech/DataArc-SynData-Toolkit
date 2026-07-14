import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sdgsystem.agentic_data.terminal_bench import (
    TerminalBenchSynthesisConfig,
    build_terminal_bench_repair_prompt,
    build_harbor_command,
    load_seed_task,
    materialize_artifact,
    run_harbor_gold_solution_verification,
    run_harbor_verification,
    run_terminal_bench_synthesis,
    summarize_terminal_bench_failure,
    terminal_bench_record_to_dataarc,
    validate_terminal_bench_artifact,
)


SEED_ROOT = Path("examples/syn_agentic_data/terminal-bench-seed")


class FakeArtifactModelClient:
    def __init__(self) -> None:
        self.prompts = []
        self.calls = 0

    def generate(self, prompt, n=1, **kwargs):
        self.prompts.append(prompt)
        self.calls += 1
        return json.dumps(
            {
                "task_name": f"synthetic-cancel-async-{self.calls}",
                "instruction_md": f"Implement a synthetic async cleanup task variant {self.calls}.",
                "files": {
                    "instruction.md": f"Implement a synthetic async cleanup task variant {self.calls}.",
                    "solution/solve.sh": "#!/usr/bin/env bash\ncat > /app/solution.py <<'PY'\nprint('ok')\nPY\n",
                    "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                    "tests/test_outputs.py": (
                        "from pathlib import Path\n\n"
                        "def test_solution_exists():\n"
                        "    assert Path('/app/solution.py').exists()\n"
                    ),
                    "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\nRUN pip install pytest==8.4.1\n",
                },
                "metadata": {"difficulty": "smoke"},
            }
        )


class DuplicateNameArtifactModelClient(FakeArtifactModelClient):
    def generate(self, prompt, n=1, **kwargs):
        self.prompts.append(prompt)
        self.calls += 1
        return json.dumps(
            {
                "task_name": "duplicate-terminal-task",
                "instruction_md": f"Implement duplicate task variant {self.calls}.",
                "files": {
                    "instruction.md": f"Implement duplicate task variant {self.calls}.",
                    "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                    "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                    "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                    "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\nRUN pip install pytest==8.4.1\n",
                },
                "metadata": {"difficulty": "smoke"},
            }
        )


class TerminalBenchSynthesisTests(unittest.TestCase):
    def test_config_from_yaml_resolves_relative_seed_root_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seeds").mkdir()
            config_path = root / "terminal.yaml"
            config_path.write_text(
                """
seed_root: seeds
output_dir: out
task_names: [cancel-async-tasks]
strategies: [few_shot]
num_samples_per_strategy: 1
model:
  provider: openai
  model: deepseek-v4-pro
harbor:
  agent: codex
  model: gpt-5.5
""",
                encoding="utf-8",
            )

            config = TerminalBenchSynthesisConfig.from_yaml(str(config_path))

            self.assertEqual(Path(config.seed_root), (root / "seeds").resolve())
            self.assertEqual(Path(config.output_dir), (root / "out").resolve())
            self.assertEqual(config.harbor["model"], "gpt-5.5")

    def test_smoke_config_covers_one_seed_with_three_strategies(self):
        config = TerminalBenchSynthesisConfig.from_yaml("configs/syn_agentic_terminal_bench_smoke.yaml")

        self.assertEqual(config.task_names, ["cancel-async-tasks"])
        self.assertEqual(config.strategies, ["few_shot", "self_instruct", "evol_instruct"])
        self.assertEqual(config.evol_directions, ["in_depth", "in_breadth"])
        self.assertIn("terminal-bench-seed", config.seed_root)
        self.assertEqual(
            Path(config.output_dir),
            Path("outputs/syn_agentic_terminal_bench_smoke_real").resolve(),
        )

    def test_terminal_bench_full_config_writes_to_repo_outputs_dir(self):
        config = TerminalBenchSynthesisConfig.from_yaml("configs/syn_agentic_terminal_bench.yaml")

        self.assertEqual(
            Path(config.output_dir),
            Path("outputs/syn_agentic_terminal_bench").resolve(),
        )

    def test_terminal_bench_example_runner_exists(self):
        self.assertTrue(Path("examples/syn_agentic_data/run_terminal_bench.py").exists())

    def test_terminal_bench_runner_avoids_full_model_client_dependency(self):
        source = Path("examples/syn_agentic_data/run_terminal_bench.py").read_text(encoding="utf-8")

        self.assertNotIn("from sdgsystem.models import ModelClient", source)
        self.assertIn("OpenAICompatibleModelClient", source)

    def test_load_seed_task_sanitizes_canary_lines_for_model_context(self):
        seed = load_seed_task(SEED_ROOT / "cancel-async-tasks")

        self.assertEqual(seed.name, "cancel-async-tasks")
        self.assertIn("instruction.md", seed.files)
        self.assertIn("solution/solve.sh", seed.files)
        self.assertIn("tests/test_outputs.py", seed.files)
        self.assertNotIn("terminal-bench-canary", seed.files["solution/solve.sh"])
        self.assertNotIn("Terminal-Bench Canary String", seed.files["solution/solve.sh"])
        self.assertNotIn("BENCHMARK DATA SHOULD NEVER APPEAR", seed.files["solution/solve.sh"])

    def test_artifact_prompt_includes_seed_helpers_and_environment_fixtures(self):
        seed = load_seed_task(SEED_ROOT / "portfolio-optimization")

        prompt = build_terminal_bench_repair_prompt(
            seed=seed,
            artifact={
                "task_name": "synthetic-cancel-async",
                "instruction_md": "Do the task.",
                "files": {
                    "instruction.md": "Do the task.",
                    "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                    "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                    "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                    "environment/Dockerfile": "FROM python:3.13-slim-bookworm\n",
                },
                "metadata": {},
            },
            failure_summary="reward 0.0",
            strategy="few_shot",
            evol_direction=None,
            repair_index=1,
            model="fake",
        )

        self.assertIn("environment/benchmark.py", prompt)
        self.assertIn("tests/portfolio_baseline_protected.py", prompt)
        self.assertIn("tests/test_outputs.py", prompt)
        self.assertIn("tests/test.sh", prompt)
        self.assertIn("environment/Dockerfile", prompt)

    def test_artifact_prompt_allows_domestic_package_mirrors(self):
        seed = load_seed_task(SEED_ROOT / "cancel-async-tasks")

        prompt = build_terminal_bench_repair_prompt(
            seed=seed,
            artifact={
                "task_name": "synthetic-cancel-async",
                "instruction_md": "Do the task.",
                "files": {
                    "instruction.md": "Do the task.",
                    "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                    "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                    "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                    "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nRUN apt-get update\n",
                },
                "metadata": {},
            },
            failure_summary="apt-get timed out",
            strategy="self_instruct",
            evol_direction=None,
            repair_index=1,
            model="fake",
        )

        self.assertIn("mirrors.tuna.tsinghua.edu.cn", prompt)
        self.assertIn("pypi.tuna.tsinghua.edu.cn", prompt)
        self.assertIn("registry.npmmirror.com", prompt)

    def test_validate_artifact_requires_task_description_solution_and_tests(self):
        valid = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\nRUN pip install pytest==8.4.1\n",
            },
            "metadata": {},
        }

        self.assertEqual(validate_terminal_bench_artifact(valid), [])

        invalid = {**valid, "files": {"../escape.txt": "bad"}}
        errors = validate_terminal_bench_artifact(invalid)

        self.assertIn("missing generated file: instruction.md", errors)
        self.assertIn("unsafe generated file path: ../escape.txt", errors)

    def test_validate_artifact_rejects_dockerfile_copying_task_root_files(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": (
                    "FROM python:3.13-slim-bookworm\n"
                    "COPY tests/test_outputs.py /tests/test_outputs.py\n"
                ),
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn(
            "environment/Dockerfile must not copy task-root path: COPY tests/test_outputs.py /tests/test_outputs.py",
            errors,
        )

    def test_validate_artifact_allows_copying_environment_context_data(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": (
                    "FROM python:3.13-slim-bookworm\n"
                    "WORKDIR /app\n"
                    "COPY data/input.txt /app/input.txt\n"
                ),
                "environment/data/input.txt": "fixture\n",
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertNotIn(
            "environment/Dockerfile must not copy task-root path: COPY data/input.txt /app/input.txt",
            errors,
        )

    def test_validate_artifact_rejects_external_python_bootstrap_urls(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\nwget https://bootstrap.pypa.io/get-pip.py\npython get-pip.py\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\n",
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("generated file must not depend on external bootstrap URL: solution/solve.sh", errors)

    def test_validate_artifact_rejects_external_uv_installer_in_tests(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\ncurl -LsSf https://astral.sh/uv/install.sh | sh\nuvx pytest\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\n",
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("tests/test.sh must not install uv from external release scripts; install needed tools in environment/Dockerfile with configured mirrors", errors)

    def test_validate_artifact_rejects_test_runner_that_can_skip_reward_under_set_e(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": (
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "pytest /tests/test_outputs.py -rA\n"
                    "if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n"
                ),
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\nRUN pip install pytest==8.4.1\n",
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("tests/test.sh must write /logs/verifier/reward.txt even when pytest fails", errors)

    def test_validate_artifact_allows_test_runner_that_captures_reward_without_set_e(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": (
                    "#!/usr/bin/env bash\n"
                    "pytest /tests/test_outputs.py -rA\n"
                    "status=$?\n"
                    "if [ $status -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n"
                    "exit $status\n"
                ),
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\nRUN pip install pytest==8.4.1\n",
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertNotIn("tests/test.sh must write /logs/verifier/reward.txt even when pytest fails", errors)

    def test_validate_artifact_rejects_apt_python_pip_repair_on_official_python_image(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\napt-get update\napt-get install -y python3-pip\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\n",
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("do not repair official Python image pip with apt python3-pip", errors)

    def test_validate_artifact_rejects_system_pip_install_on_debian_ubuntu_without_venv(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npython3 -m pytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": (
                    "FROM ubuntu:24.04\n"
                    "RUN apt-get update && apt-get install -y python3 python3-pip\n"
                    "RUN pip3 install pytest==8.4.1\n"
                ),
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn(
            "environment/Dockerfile must not use system pip install on Debian/Ubuntu without a venv or --break-system-packages",
            errors,
        )

    def test_validate_artifact_rejects_incomplete_ensurepip_repair_with_removed_pip_package(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\npython -m ensurepip --upgrade\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": (
                    "FROM python:3.13-slim-bookworm\n"
                    "RUN rm -rf /usr/local/lib/python3.13/site-packages/pip\n"
                ),
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("ensurepip repair must remove stale /usr/local/bin/pip entrypoints and use --default-pip", errors)

    def test_validate_artifact_rejects_unstable_or_reserved_dockerfile_patterns(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": (
                    "FROM debian:latest\n"
                    "RUN useradd -m nobody\n"
                ),
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("environment/Dockerfile must pin base image tag instead of using :latest", errors)
        self.assertIn("environment/Dockerfile must not create reserved user: nobody", errors)

    def test_validate_artifact_rejects_unguarded_apt_sources_list_sed(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": (
                    "FROM python:3.13-slim-bookworm\n"
                    "RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list\n"
                ),
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("environment/Dockerfile must not sed /etc/apt/sources.list without an existence guard", errors)

    def test_validate_artifact_allows_useradd_with_bin_bash_shell(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": (
                    "FROM ubuntu:22.04\n"
                    "RUN useradd -m -s /bin/bash alice\n"
                ),
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertNotIn("environment/Dockerfile must not create reserved user: bin", errors)

    def test_validate_artifact_rejects_terminal_bench_canary_variants(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\n# Terminal-Bench Canary String, DO NOT REMOVE:\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\nRUN pip install pytest==8.4.1\n",
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("generated file contains terminal-bench canary text: solution/solve.sh", errors)

    def test_validate_artifact_rejects_mismatched_instruction_fields(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the updated task.",
            "files": {
                "instruction.md": "Do the original task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\nRUN pip install pytest==8.4.1\n",
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("instruction_md must match files['instruction.md']", errors)

    def test_repair_prompt_includes_complete_artifact_and_failure_details(self):
        seed = load_seed_task(SEED_ROOT / "cancel-async-tasks")
        artifact = {
            "task_name": "synthetic-async-cleanup",
            "instruction_md": "Fix cancellation cleanup.",
            "files": {
                "instruction.md": "Fix cancellation cleanup.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM ubuntu:22.04\nRUN apt-get update\n",
            },
            "metadata": {},
        }
        failure_summary = summarize_terminal_bench_failure(
            {
                "parse_error": None,
                "schema_errors": [],
                "verification": {
                    "status": "gold_solution_failed",
                    "matched": False,
                    "gold_solution": {
                        "reward": 0.0,
                        "returncode": 0,
                        "harbor_result": {
                            "exception_type": "RuntimeError",
                            "exception_message_tail": "Docker build failed during apt-get update",
                        },
                    },
                },
            }
        )

        prompt = build_terminal_bench_repair_prompt(
            seed=seed,
            artifact=artifact,
            failure_summary=failure_summary,
            strategy="few_shot",
            evol_direction=None,
            repair_index=1,
            model="deepseek-v4-flash",
        )

        self.assertIn("Return exactly one complete JSON object", prompt)
        self.assertIn("Docker build failed during apt-get update", prompt)
        self.assertIn('"task_name": "synthetic-async-cleanup"', prompt)
        self.assertIn("solution/solve.sh", prompt)
        self.assertIn("tests/test_outputs.py", prompt)
        self.assertIn("reward 1", prompt)

    def test_materialize_artifact_copies_environment_and_applies_generated_files(self):
        seed = load_seed_task(SEED_ROOT / "cancel-async-tasks")
        artifact = {
            "task_name": "synthetic-cancel-async",
            "instruction_md": "New task text.",
            "files": {
                "instruction.md": "New task text.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\nRUN pip install pytest==8.4.1\n",
            },
            "metadata": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = materialize_artifact(seed, artifact, Path(tmp))

            self.assertTrue((task_dir / "environment" / "Dockerfile").exists())
            self.assertEqual((task_dir / "instruction.md").read_text(encoding="utf-8"), "New task text.")
            self.assertIn("true", (task_dir / "solution" / "solve.sh").read_text(encoding="utf-8"))
            self.assertNotIn(
                "Terminal-Bench Canary String",
                (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8"),
            )
            self.assertNotIn("terminal-bench-canary", (task_dir / "tests" / "test_outputs.py").read_text(encoding="utf-8"))

    def test_materialize_artifact_injects_package_mirrors_into_networked_dockerfile(self):
        seed = load_seed_task(SEED_ROOT / "cancel-async-tasks")
        artifact = {
            "task_name": "synthetic-cancel-async",
            "instruction_md": "New task text.",
            "files": {
                "instruction.md": "New task text.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": (
                    "FROM python:3.13-slim-bookworm\n"
                    "WORKDIR /app\n"
                    "RUN apt-get update && apt-get install -y git\n"
                    "RUN pip install pytest==8.4.1\n"
                    "RUN npm install left-pad\n"
                ),
            },
            "metadata": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = materialize_artifact(seed, artifact, Path(tmp))

            dockerfile = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")

            self.assertIn("mirrors.tuna.tsinghua.edu.cn", dockerfile)
            self.assertIn("http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports", dockerfile)
            self.assertNotIn("https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports", dockerfile)
            self.assertIn("PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple", dockerfile)
            self.assertIn("NPM_CONFIG_REGISTRY=https://registry.npmmirror.com", dockerfile)
            self.assertIn("apt-get update && apt-get install -y git", dockerfile)

    def test_materialize_artifact_injects_package_mirrors_for_solution_network_commands(self):
        seed = load_seed_task(SEED_ROOT / "cancel-async-tasks")
        artifact = {
            "task_name": "synthetic-cancel-async",
            "instruction_md": "New task text.",
            "files": {
                "instruction.md": "New task text.",
                "solution/solve.sh": "#!/usr/bin/env bash\npip install virtualenv\n",
                "tests/test.sh": "#!/usr/bin/env bash\napt-get update\nnpm install left-pad\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\n",
            },
            "metadata": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = materialize_artifact(seed, artifact, Path(tmp))

            dockerfile = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")

            self.assertIn("mirrors.tuna.tsinghua.edu.cn", dockerfile)
            self.assertIn("PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple", dockerfile)
            self.assertIn("NPM_CONFIG_REGISTRY=https://registry.npmmirror.com", dockerfile)

    def test_materialize_artifact_optionally_rewrites_official_base_images_to_docker_mirror(self):
        seed = load_seed_task(SEED_ROOT / "cancel-async-tasks")
        artifact = {
            "task_name": "synthetic-cancel-async",
            "instruction_md": "New task text.",
            "files": {
                "instruction.md": "New task text.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test.sh": "#!/usr/bin/env bash\npytest /tests/test_outputs.py -rA\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
                "environment/Dockerfile": "FROM python:3.13-slim-bookworm\nWORKDIR /app\n",
            },
            "metadata": {},
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"SYNDATA_DOCKER_REGISTRY_MIRROR": "docker.mirror.local"},
        ):
            task_dir = materialize_artifact(seed, artifact, Path(tmp))

            dockerfile = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")

            self.assertIn("FROM docker.mirror.local/library/python:3.13-slim-bookworm", dockerfile)

    def test_run_terminal_bench_synthesis_generates_strategy_task_directories_and_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = TerminalBenchSynthesisConfig(
                seed_root=str(SEED_ROOT),
                output_dir=str(Path(tmp) / "out"),
                task_names=["cancel-async-tasks"],
                strategies=["few_shot", "self_instruct", "evol_instruct"],
                num_samples_per_strategy=1,
                evol_directions=["in_depth", "in_breadth"],
                validation_mode="static",
                model={"model": "fake-terminal-bench-model"},
            )

            result = run_terminal_bench_synthesis(config, FakeArtifactModelClient())

            self.assertEqual(result["summary"]["num_records"], 4)
            self.assertEqual(result["summary"]["num_accepted"], 4)
            self.assertEqual(
                [record["strategy"] for record in result["records"]],
                ["few_shot", "self_instruct", "evol_instruct", "evol_instruct"],
            )
            self.assertIn("harbor_command", result["records"][0])
            self.assertIn("gpt-5.5", result["records"][0]["harbor_command"])
            output_dir = Path(config.output_dir)
            self.assertTrue(Path(result["records"][0]["task_path"]).exists())
            self.assertTrue(str(result["records"][0]["task_path"]).startswith(str(output_dir / "tasks")))
            exported = json.loads((output_dir / "dataarc_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(exported["messages"][2]["content"][1]["type"], "function_call")
            self.assertEqual(json.loads(exported["tools"])[0]["function"]["name"], "run_harbor_terminal_bench")
            manifest = json.loads((output_dir / "terminal_bench_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest), 4)
            self.assertEqual(manifest[0]["task_path"], result["records"][0]["task_path"])
            self.assertEqual(
                manifest[0]["terminal_bench_files"],
                {
                    "instruction": "instruction.md",
                    "task_config": "task.toml",
                    "environment": "environment/Dockerfile",
                    "solution": "solution/solve.sh",
                    "tests": "tests/test_outputs.py",
                },
            )
            self.assertTrue(Path(manifest[0]["task_path"], manifest[0]["terminal_bench_files"]["instruction"]).exists())

    def test_run_terminal_bench_synthesis_keeps_duplicate_task_names_in_distinct_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = TerminalBenchSynthesisConfig(
                seed_root=str(SEED_ROOT),
                output_dir=str(Path(tmp) / "out"),
                task_names=["cancel-async-tasks"],
                strategies=["few_shot", "self_instruct", "evol_instruct"],
                num_samples_per_strategy=1,
                evol_directions=["in_depth", "in_breadth"],
                validation_mode="static",
                model={"model": "fake-terminal-bench-model"},
            )

            result = run_terminal_bench_synthesis(config, DuplicateNameArtifactModelClient())
            task_paths = [record["task_path"] for record in result["records"]]

            self.assertEqual(len(task_paths), 4)
            self.assertEqual(len(set(task_paths)), 4)
            for task_path in task_paths:
                self.assertTrue(Path(task_path).exists(), task_path)

    def test_gold_solution_validation_failure_prevents_accepting_generated_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = TerminalBenchSynthesisConfig(
                seed_root=str(SEED_ROOT),
                output_dir=str(Path(tmp) / "out"),
                task_names=["cancel-async-tasks"],
                strategies=["few_shot"],
                num_samples_per_strategy=1,
                model={"model": "fake-terminal-bench-model"},
            )
            failed_gold = {
                "matched": False,
                "status": "gold_solution_failed",
                "returncode": 0,
                "reward": 0,
                "stdout_tail": "pytest failed",
                "stderr_tail": "",
                "command": ["harbor", "run", "--agent", "oracle"],
            }

            with patch(
                "sdgsystem.agentic_data.terminal_bench.run_harbor_gold_solution_verification",
                return_value=failed_gold,
            ):
                result = run_terminal_bench_synthesis(config, FakeArtifactModelClient())

            self.assertEqual(result["summary"]["num_records"], 1)
            self.assertEqual(result["summary"]["num_accepted"], 0)
            self.assertEqual(result["records"][0]["verification"]["status"], "gold_solution_failed")

    def test_terminal_bench_record_export_omits_solution_tests_and_canary_text(self):
        record = {
            "task_name": "synthetic-task",
            "task_path": "/tmp/synthetic-task",
            "strategy": "few_shot",
            "instruction": "Solve this generated terminal-bench task.",
            "verification": {"matched": True, "status": "static_validated"},
            "harbor": {"agent": "codex", "model": "gpt-5.5"},
        }

        row = terminal_bench_record_to_dataarc(record)
        blob = json.dumps(row, ensure_ascii=False).lower()
        call = json.loads(row["messages"][2]["content"][1]["value"])

        self.assertIn("messages", row)
        self.assertEqual(call["name"], "run_harbor_terminal_bench")
        self.assertEqual(call["arguments"]["model"], "gpt-5.5")
        self.assertNotIn("solution/solve.sh", blob)
        self.assertNotIn("tests/test_outputs.py", blob)
        self.assertNotIn("terminal-bench-canary", blob)

    def test_build_harbor_command_uses_codex_gpt55_and_env_names_without_secret_values(self):
        command = build_harbor_command(
            task_path=Path("/tmp/synthetic-task"),
            jobs_dir=Path("/tmp/jobs"),
            harbor_bin="/venv/bin/harbor",
            agent="codex",
            model="gpt-5.5",
        )
        rendered = " ".join(command)

        self.assertIn("/venv/bin/harbor", command[0])
        self.assertIn("codex", command)
        self.assertIn("gpt-5.5", command)
        self.assertIn("OPENAI_BASE_URL=${HARBOR_OPENAI_BASE_URL}", rendered)
        self.assertIn("OPENAI_API_KEY=${HARBOR_API_KEY}", rendered)
        self.assertIn("--force-build", command)
        self.assertNotIn("sk-", rendered)

    def test_build_harbor_command_omits_model_for_oracle_gold_solution_validation(self):
        command = build_harbor_command(
            task_path=Path("/tmp/synthetic-task"),
            jobs_dir=Path("/tmp/jobs"),
            harbor_bin="/venv/bin/harbor",
            agent="oracle",
            model=None,
        )

        self.assertIn("oracle", command)
        self.assertNotIn("--model", command)
        self.assertNotIn("OPENAI_API_KEY=${HARBOR_API_KEY}", " ".join(command))

    def test_run_harbor_verification_skips_when_required_env_is_missing(self):
        result = run_harbor_verification(
            task_path=Path("/tmp/synthetic-task"),
            jobs_dir=Path("/tmp/jobs"),
            harbor_bin="/venv/bin/harbor",
            env={},
        )

        self.assertFalse(result["matched"])
        self.assertEqual(result["status"], "skipped")
        self.assertIn("HARBOR_API_KEY", result["missing_env"])

    def test_run_harbor_verification_executes_and_reads_reward_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_dir = root / "jobs"
            reward_path = jobs_dir / "job" / "trial" / "logs" / "verifier" / "reward.txt"
            reward_path.parent.mkdir(parents=True)
            reward_path.write_text("1.0\n", encoding="utf-8")

            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "harbor ok",
                    "stderr": "",
                },
            )()

            with patch("sdgsystem.agentic_data.terminal_bench.subprocess.run", return_value=completed) as run_mock:
                result = run_harbor_verification(
                    task_path=root / "task",
                    jobs_dir=jobs_dir,
                    harbor_bin="/venv/bin/harbor",
                    env={"HARBOR_API_KEY": "secret", "HARBOR_OPENAI_BASE_URL": "http://example/v1"},
                )

            self.assertTrue(result["matched"])
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["reward"], 1.0)
            called_env = run_mock.call_args.kwargs["env"]
            self.assertEqual(called_env["OPENAI_API_KEY"], "secret")
            self.assertEqual(called_env["OPENAI_BASE_URL"], "http://example/v1")

    def test_run_harbor_gold_solution_verification_uses_oracle_and_needs_no_api_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_dir = root / "jobs"
            reward_path = jobs_dir / "job" / "trial" / "logs" / "verifier" / "reward.txt"
            reward_path.parent.mkdir(parents=True)
            reward_path.write_text("0\n", encoding="utf-8")

            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "oracle ran",
                    "stderr": "",
                },
            )()

            with patch("sdgsystem.agentic_data.terminal_bench.subprocess.run", return_value=completed) as run_mock:
                result = run_harbor_gold_solution_verification(
                    task_path=root / "task",
                    jobs_dir=jobs_dir,
                    harbor_bin="/venv/bin/harbor",
                    env={},
                )

            self.assertFalse(result["matched"])
            self.assertEqual(result["status"], "gold_solution_failed")
            self.assertEqual(result["reward"], 0)
            command = run_mock.call_args.args[0]
            self.assertIn("oracle", command)
            self.assertNotIn("--model", command)

    def test_run_harbor_gold_solution_verification_summarizes_harbor_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_dir = root / "jobs"
            trial_dir = jobs_dir / "2026-07-02__00-00-00" / "synthetic-trial"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "exception_info": {
                            "exception_type": "RuntimeError",
                            "exception_message": "Docker build failed: pytest command not found",
                        },
                        "verifier_result": None,
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "exception.txt").write_text("Docker build failed: pytest command not found\n", encoding="utf-8")

            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "harbor finished with runtime error",
                    "stderr": "",
                },
            )()

            with patch("sdgsystem.agentic_data.terminal_bench.subprocess.run", return_value=completed):
                result = run_harbor_gold_solution_verification(
                    task_path=root / "task",
                    jobs_dir=jobs_dir,
                    harbor_bin="/venv/bin/harbor",
                    env={},
                )

            self.assertFalse(result["matched"])
            self.assertEqual(result["harbor_result"]["exception_type"], "RuntimeError")
            self.assertIn("pytest command not found", result["harbor_result"]["exception_message_tail"])
            self.assertIn("pytest command not found", result["harbor_result"]["exception_txt_tail"])

    def test_run_harbor_gold_solution_verification_prefers_nested_trial_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_dir = root / "jobs"
            run_dir = jobs_dir / "2026-07-02__00-00-00"
            trial_dir = run_dir / "synthetic-trial"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "exception_info": {
                            "exception_type": "RuntimeError",
                            "exception_message": "sed: can't read /etc/apt/sources.list",
                        },
                        "verifier_result": None,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "result.json").write_text(
                json.dumps({"exception_info": None, "verifier_result": None}),
                encoding="utf-8",
            )

            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "harbor finished with runtime error",
                    "stderr": "",
                },
            )()

            with patch("sdgsystem.agentic_data.terminal_bench.subprocess.run", return_value=completed):
                result = run_harbor_gold_solution_verification(
                    task_path=root / "task",
                    jobs_dir=jobs_dir,
                    harbor_bin="/venv/bin/harbor",
                    env={},
                )

            self.assertEqual(result["harbor_result"]["exception_type"], "RuntimeError")
            self.assertIn("/etc/apt/sources.list", result["harbor_result"]["exception_message_tail"])

    def test_run_harbor_gold_solution_verification_includes_verifier_output_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_dir = root / "jobs"
            trial_dir = jobs_dir / "2026-07-02__00-00-00" / "synthetic-trial"
            verifier_dir = trial_dir / "verifier"
            verifier_dir.mkdir(parents=True)
            (verifier_dir / "reward.txt").write_text("0\n", encoding="utf-8")
            (verifier_dir / "test-stdout.txt").write_text("uvx: command not found\n", encoding="utf-8")
            (verifier_dir / "test-stderr.txt").write_text("curl: (18) transfer closed\n", encoding="utf-8")
            (trial_dir / "result.json").write_text(
                json.dumps({"exception_info": None, "verifier_result": {"rewards": {"reward": 0.0}}}),
                encoding="utf-8",
            )

            completed = type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "harbor finished",
                    "stderr": "",
                },
            )()

            with patch("sdgsystem.agentic_data.terminal_bench.subprocess.run", return_value=completed):
                result = run_harbor_gold_solution_verification(
                    task_path=root / "task",
                    jobs_dir=jobs_dir,
                    harbor_bin="/venv/bin/harbor",
                    env={},
                )

            self.assertFalse(result["matched"])
            self.assertIn("uvx: command not found", result["harbor_result"]["verifier_stdout_tail"])
            self.assertIn("curl: (18)", result["harbor_result"]["verifier_stderr_tail"])

    def test_summarize_terminal_bench_failure_includes_verifier_output_tail(self):
        summary = summarize_terminal_bench_failure(
            {
                "verification": {
                    "status": "gold_solution_failed",
                    "matched": False,
                    "gold_solution": {
                        "reward": 0.0,
                        "harbor_result": {
                            "verifier_stdout_tail": "uvx: command not found",
                            "verifier_stderr_tail": "curl: transfer closed",
                        },
                    },
                },
            }
        )

        self.assertIn("uvx: command not found", summary)
        self.assertIn("curl: transfer closed", summary)


if __name__ == "__main__":
    unittest.main()
