import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sdgsystem.agentic_data.terminal_bench import (
    TerminalBenchSynthesisConfig,
    build_harbor_command,
    load_seed_task,
    materialize_artifact,
    run_harbor_verification,
    run_terminal_bench_synthesis,
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
                    "tests/test_outputs.py": (
                        "from pathlib import Path\n\n"
                        "def test_solution_exists():\n"
                        "    assert Path('/app/solution.py').exists()\n"
                    ),
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
                    "tests/test_outputs.py": "def test_ok():\n    assert True\n",
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

    def test_validate_artifact_requires_task_description_solution_and_tests(self):
        valid = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
            },
            "metadata": {},
        }

        self.assertEqual(validate_terminal_bench_artifact(valid), [])

        invalid = {**valid, "files": {"../escape.txt": "bad"}}
        errors = validate_terminal_bench_artifact(invalid)

        self.assertIn("missing generated file: instruction.md", errors)
        self.assertIn("unsafe generated file path: ../escape.txt", errors)

    def test_validate_artifact_rejects_terminal_bench_canary_variants(self):
        artifact = {
            "task_name": "synthetic-task",
            "instruction_md": "Do the task.",
            "files": {
                "instruction.md": "Do the task.",
                "solution/solve.sh": "#!/usr/bin/env bash\n# Terminal-Bench Canary String, DO NOT REMOVE:\ntrue\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
            },
            "metadata": {},
        }

        errors = validate_terminal_bench_artifact(artifact)

        self.assertIn("generated file contains terminal-bench canary text: solution/solve.sh", errors)

    def test_materialize_artifact_copies_environment_and_applies_generated_files(self):
        seed = load_seed_task(SEED_ROOT / "cancel-async-tasks")
        artifact = {
            "task_name": "synthetic-cancel-async",
            "instruction_md": "New task text.",
            "files": {
                "instruction.md": "New task text.",
                "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
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

    def test_run_terminal_bench_synthesis_generates_strategy_task_directories_and_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = TerminalBenchSynthesisConfig(
                seed_root=str(SEED_ROOT),
                output_dir=str(Path(tmp) / "out"),
                task_names=["cancel-async-tasks"],
                strategies=["few_shot", "self_instruct", "evol_instruct"],
                num_samples_per_strategy=1,
                evol_directions=["in_depth", "in_breadth"],
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
                model={"model": "fake-terminal-bench-model"},
            )

            result = run_terminal_bench_synthesis(config, DuplicateNameArtifactModelClient())
            task_paths = [record["task_path"] for record in result["records"]]

            self.assertEqual(len(task_paths), 4)
            self.assertEqual(len(set(task_paths)), 4)
            for task_path in task_paths:
                self.assertTrue(Path(task_path).exists(), task_path)

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
            reward_path.write_text("1\n", encoding="utf-8")

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
            self.assertEqual(result["reward"], 1)
            called_env = run_mock.call_args.kwargs["env"]
            self.assertEqual(called_env["OPENAI_API_KEY"], "secret")
            self.assertEqual(called_env["OPENAI_BASE_URL"], "http://example/v1")


if __name__ == "__main__":
    unittest.main()
