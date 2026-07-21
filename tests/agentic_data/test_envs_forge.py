import math
import tempfile
import unittest
from pathlib import Path

from sdgsystem.agentic_data.envs_forge import (
    EnvsForgePlanItem,
    build_envs_forge_prompt,
    estimate_pass_rate,
    extract_skill_profile,
    learning_frontier_score,
    load_rewards_from_harbor_result,
    plan_envs_forge_batch,
    select_calibrated_projection,
)
from sdgsystem.agentic_data.terminal_bench import TerminalBenchSeedTask


def make_seed(name: str, instruction: str, files: dict[str, str] | None = None) -> TerminalBenchSeedTask:
    root = Path(tempfile.mkdtemp()) / name
    root.mkdir(parents=True)
    all_files = {
        "instruction.md": instruction,
        "task.toml": "name = 'seed'\n",
        "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
        "tests/test.sh": "#!/usr/bin/env bash\ntrue\n",
        "tests/test_outputs.py": "def test_ok():\n    assert True\n",
        "environment/Dockerfile": "FROM python:3.13-slim-bookworm\n",
    }
    if files:
        all_files.update(files)
    return TerminalBenchSeedTask(name=name, path=root, instruction=instruction, task_toml=all_files["task.toml"], files=all_files)


class EnvsForgeTests(unittest.TestCase):
    def test_estimate_pass_rate_uses_beta_smoothing(self):
        self.assertAlmostEqual(estimate_pass_rate([1, 0, 1], alpha=1, beta=1), 3 / 5)
        self.assertAlmostEqual(estimate_pass_rate([], alpha=1, beta=1), 0.5)

    def test_learning_frontier_score_peaks_near_target(self):
        peak = learning_frontier_score(0.5, target=0.5, sigma=0.2)
        easy = learning_frontier_score(0.95, target=0.5, sigma=0.2)
        hard = learning_frontier_score(0.05, target=0.5, sigma=0.2)

        self.assertTrue(math.isclose(peak, 1.0))
        self.assertLess(easy, 0.1)
        self.assertLess(hard, 0.1)

    def test_extract_skill_profile_detects_tools_artifacts_operations_and_constraints(self):
        seed = make_seed(
            "systemd-log-monitoring",
            "Create a systemd service and timer that writes JSON logs, parses journald output, and configures logrotate.",
            {
                "solution/solve.sh": "systemctl enable app.service\njournalctl -u app.service\n",
                "tests/test_outputs.py": "def test_logrotate_and_json_logs(): pass\n",
            },
        )

        profile = extract_skill_profile(seed)

        self.assertIn("system_admin", profile.domains)
        self.assertIn("systemd", profile.tools)
        self.assertIn("logs", profile.artifacts)
        self.assertIn("configure", profile.operations)
        self.assertIn("scheduling", profile.constraints)

    def test_plan_envs_forge_batch_balances_easy_frontier_and_hard_projection(self):
        seeds = [
            make_seed("csv-json-jsonl-merger", "Merge csv json and jsonl files into a deterministic csv."),
            make_seed("bash-log-processor-fix", "Fix a bash log processor with concurrency and atomic writes."),
            make_seed("systemd-log-monitoring", "Configure systemd service, timer, journald log parsing, and logrotate."),
            make_seed("tsl-test-case-generation", "Generate tests from a TSL category partition specification."),
        ]
        rewards = {
            "csv-json-jsonl-merger": [1, 1, 1],
            "bash-log-processor-fix": [1, 0, 0],
            "systemd-log-monitoring": [0, 0, 0],
            "tsl-test-case-generation": [0, 0, 0],
        }

        plan = plan_envs_forge_batch(seeds, rewards_by_seed=rewards, target_count=4)

        self.assertEqual(len(plan), 4)
        self.assertEqual(plan[0].projection, "diversify_frontier")
        self.assertIn("bash-log-processor-fix", {item.seed.name for item in plan})
        self.assertIn("increase_complexity", {item.projection for item in plan})
        self.assertIn("reduce_complexity", {item.projection for item in plan})

    def test_plan_envs_forge_batch_uses_prefix_rewards_for_truncated_harbor_trial_names(self):
        seeds = [
            make_seed("api-endpoint-permission-canonicalizer", "Canonicalize API endpoint permissions."),
            make_seed("systemd-log-monitoring", "Configure systemd service, timer, journald log parsing, and logrotate."),
        ]
        rewards = {
            "api-endpoint-permission-canonica": [1, 1, 1],
            "systemd-log-monitoring": [0, 0, 0],
        }

        plan = plan_envs_forge_batch(seeds, rewards_by_seed=rewards, target_count=2)
        by_seed = {item.seed.name: item for item in plan}

        self.assertEqual(by_seed["api-endpoint-permission-canonicalizer"].band, "easy")
        self.assertEqual(by_seed["systemd-log-monitoring"].band, "hard")

    def test_select_calibrated_projection_chooses_action_closest_to_frontier(self):
        shifts = {
            "increase_complexity": -0.5,
            "reduce_complexity": 0.5,
            "diversify_frontier": 0.0,
        }

        easy_projection, easy_predicted, easy_score = select_calibrated_projection(
            1.0,
            action_shifts=shifts,
        )
        hard_projection, hard_predicted, hard_score = select_calibrated_projection(
            0.0,
            action_shifts=shifts,
        )
        frontier_projection, frontier_predicted, frontier_score = select_calibrated_projection(
            0.5,
            action_shifts=shifts,
        )

        self.assertEqual(easy_projection, "increase_complexity")
        self.assertAlmostEqual(easy_predicted, 0.5)
        self.assertAlmostEqual(easy_score, 1.0)
        self.assertEqual(hard_projection, "reduce_complexity")
        self.assertAlmostEqual(hard_predicted, 0.5)
        self.assertAlmostEqual(hard_score, 1.0)
        self.assertEqual(frontier_projection, "diversify_frontier")
        self.assertAlmostEqual(frontier_predicted, 0.5)
        self.assertAlmostEqual(frontier_score, 1.0)

    def test_plan_envs_forge_batch_can_use_calibrated_transfer_policy(self):
        seeds = [
            make_seed("easy-csv-task", "Merge csv files."),
            make_seed("hard-systemd-task", "Configure systemd service, timer, journald logs, and logrotate."),
            make_seed("frontier-json-task", "Normalize malformed jsonl rows with deterministic sorting."),
        ]
        rewards = {
            "easy-csv-task": [1, 1],
            "hard-systemd-task": [0, 0],
            "frontier-json-task": [1, 0],
        }

        plan = plan_envs_forge_batch(
            seeds,
            rewards_by_seed=rewards,
            target_count=3,
            projection_policy="calibrated_transfer",
            action_shifts={
                "increase_complexity": -0.5,
                "reduce_complexity": 0.5,
                "diversify_frontier": 0.0,
            },
        )
        by_seed = {item.seed.name: item for item in plan}

        self.assertEqual(by_seed["easy-csv-task"].projection, "increase_complexity")
        self.assertEqual(by_seed["hard-systemd-task"].projection, "reduce_complexity")
        self.assertEqual(by_seed["frontier-json-task"].projection, "diversify_frontier")
        self.assertEqual(by_seed["easy-csv-task"].projection_policy, "calibrated_transfer")
        self.assertAlmostEqual(by_seed["easy-csv-task"].predicted_pass_rate, 0.5)
        self.assertAlmostEqual(by_seed["hard-systemd-task"].predicted_pass_rate, 0.5)
        self.assertAlmostEqual(by_seed["frontier-json-task"].predicted_pass_rate, 0.5)

    def test_build_envs_forge_prompt_contains_projection_contract_and_full_artifact_requirements(self):
        seed = make_seed("systemd-log-monitoring", "Configure systemd service, timer, journald logs, and logrotate.")
        item = EnvsForgePlanItem(
            seed=seed,
            profile=extract_skill_profile(seed),
            pass_rate=0.05,
            frontier_score=0.08,
            band="hard",
            projection="reduce_complexity",
            target_difficulty="easy",
            sample_index=0,
        )

        prompt = build_envs_forge_prompt(item, model="deepseek-v4-flash")

        self.assertIn("Envs-FORGE", prompt)
        self.assertIn("reduce compositional complexity", prompt)
        self.assertIn("preserve the target skill subgraph", prompt)
        self.assertIn("solution/solve.sh", prompt)
        self.assertIn("tests/test.sh", prompt)
        self.assertIn("tests/test_outputs.py", prompt)
        self.assertIn("environment/Dockerfile", prompt)
        self.assertIn("Keep the generated artifact small", prompt)
        self.assertIn("Do not modify apt sources", prompt)

    def test_load_rewards_from_harbor_result_maps_trial_names_to_seed_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(
                """
{
  "stats": {
    "evals": {
      "terminus-2__qwen3-8b__tblite": {
        "reward_stats": {
          "reward": {
            "1.0": ["csv-json-jsonl-merger__abc123"],
            "0.0": ["systemd-log-monitoring__def456"],
            "0.5": ["bash-log-processor-fix__ghi789"]
          }
        }
      }
    }
  }
}
""",
                encoding="utf-8",
            )

            rewards = load_rewards_from_harbor_result(path)

        self.assertEqual(rewards["csv-json-jsonl-merger"], [1.0])
        self.assertEqual(rewards["systemd-log-monitoring"], [0.0])
        self.assertEqual(rewards["bash-log-processor-fix"], [0.5])


if __name__ == "__main__":
    unittest.main()
