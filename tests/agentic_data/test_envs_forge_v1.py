import math
import tempfile
import unittest
from pathlib import Path

from sdgsystem.agentic_data.envs_forge_v1 import (
    BAND_SLACK_PENALTY,
    CONSISTENCY_IMPLICATIONS,
    DEFAULT_E_MAX,
    EDIT_COMPONENTS,
    METHOD_VERSION,
    ComponentEditPlan,
    allowed_components,
    build_component_edit_prompt,
    build_component_repair_prompt,
    component_coverage,
    component_of_path,
    estimate_component_costs,
    expand_subset_artifact,
    plan_component_edit,
    plan_envs_forge_v1_batch,
    smoothed_pass_rate,
    validate_subset_artifact,
)
from sdgsystem.agentic_data.terminal_bench import TerminalBenchSeedTask


def make_seed(name: str, instruction: str, files: dict[str, str] | None = None) -> TerminalBenchSeedTask:
    root = Path(tempfile.mkdtemp()) / name
    root.mkdir(parents=True)
    all_files = {
        "instruction.md": instruction,
        "task.toml": "name = 'seed'\n",
        "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
        "tests/test.sh": "#!/usr/bin/env bash\necho 0 > /logs/verifier/reward.txt\n",
        "tests/test_outputs.py": "def test_ok():\n    assert True\n",
        "environment/Dockerfile": "FROM ubuntu:22.04\n",
        "environment/data/input.txt": "1,2,3\n",
    }
    all_files.update(files or {})
    for relative, content in all_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    from sdgsystem.agentic_data.terminal_bench import load_seed_task

    return load_seed_task(root)


def _lp_relaxation_is_fractional(plan_args: dict) -> bool:
    """Re-solve the v1 model with y relaxed and check for a fractional optimum."""
    try:
        from pyscipopt import Model, quicksum  # type: ignore
    except Exception:
        return False
    from sdgsystem.agentic_data.envs_forge_v1 import (
        CONSISTENCY_IMPLICATIONS as impl,
        DEFAULT_SENSITIVITY as alpha,
        DIVERSIFY_ANCHORS as anchors,
        _movement_capacity,
    )

    allowed = plan_args["allowed"]
    z0 = plan_args["z0"]
    band = plan_args["band"]
    e_max = plan_args["e_max"]
    rates = plan_args["fixed_cost"]
    unit = plan_args["linear_cost"]
    model = Model()
    model.hideOutput()
    touch = {c: model.addVar(vtype="C", lb=0.0, ub=1.0, name=f"y_{c}") for c in allowed}
    pos = {c: model.addVar(vtype="C", lb=0.0, ub=e_max, name=f"p_{c}") for c in allowed}
    neg = {c: model.addVar(vtype="C", lb=0.0, ub=e_max, name=f"n_{c}") for c in allowed}
    slack = model.addVar(vtype="C", lb=0.0, name="slack")
    for c in allowed:
        model.addCons(pos[c] <= e_max * touch[c])
        model.addCons(neg[c] <= e_max * touch[c])
    model.addCons(quicksum(touch[c] for c in allowed) >= 1)
    if abs(z0) <= band:
        present = [c for c in anchors if c in allowed]
        if present:
            model.addCons(quicksum(touch[c] for c in present) >= 1)
    for source, target in impl:
        if source in allowed and target in allowed:
            model.addCons(touch[source] <= touch[target])
    movement = quicksum(alpha[c] * (pos[c] - neg[c]) for c in allowed)
    model.addCons(z0 - movement <= band + slack)
    model.addCons(-(z0 - movement) <= band + slack)
    model.setObjective(BAND_SLACK_PENALTY * slack + quicksum(rates[c] * touch[c] + unit[c] * (pos[c] + neg[c]) for c in allowed), "minimize")
    model.optimize()
    if str(model.getStatus()) != "optimal":
        return False
    return any(1e-6 < model.getVal(touch[c]) < 1 - 1e-6 for c in allowed)


class EnvsForgeV1Tests(unittest.TestCase):
    def test_method_version_is_v1(self):
        self.assertEqual(METHOD_VERSION, "envs-forge-v1")

    def test_component_of_path_maps_every_seed_file(self):
        self.assertEqual(component_of_path("instruction.md"), "instruction")
        self.assertEqual(component_of_path("solution/solve.sh"), "solution")
        self.assertEqual(component_of_path("tests/test.sh"), "tests")
        self.assertEqual(component_of_path("tests/test_outputs.py"), "tests")
        self.assertEqual(component_of_path("environment/Dockerfile"), "environment")
        self.assertEqual(component_of_path("environment/data/input.txt"), "data")
        self.assertIsNone(component_of_path("task.toml"))

    def test_allowed_components_exclude_missing_ones(self):
        seed = make_seed("no-data", "Merge csv files.", {"environment/data/input.txt": ""})
        root = seed.path
        (root / "environment/data/input.txt").unlink()
        from sdgsystem.agentic_data.terminal_bench import load_seed_task

        reloaded = load_seed_task(root)
        self.assertNotIn("data", allowed_components(reloaded))
        self.assertIn("tests", allowed_components(reloaded))

    def test_smoothed_pass_rate_keeps_endpoints_finite(self):
        self.assertAlmostEqual(smoothed_pass_rate([]), 0.5)
        self.assertAlmostEqual(smoothed_pass_rate([1.0]), 2 / 3)
        self.assertAlmostEqual(smoothed_pass_rate([0.0]), 1 / 3)
        self.assertAlmostEqual(smoothed_pass_rate([1, 1, 0, 1]), 4 / 6)

    def test_plan_reaches_frontier_band_and_reports_cost(self):
        seed = make_seed("easy-csv", "Merge csv files.")
        plan = plan_component_edit(seed, rewards=[1.0])
        self.assertIsInstance(plan, ComponentEditPlan)
        self.assertTrue(plan.reachable)
        self.assertLessEqual(abs(plan.z_predicted), 0.4 + 1e-6)
        self.assertGreater(plan.edit_strength, 0.0)
        self.assertAlmostEqual(plan.cost, plan.fixed_cost + plan.linear_cost)
        self.assertEqual(plan.projection, "increase_complexity")

    def test_hard_seed_gets_reduce_projection(self):
        seed = make_seed("hard-systemd", "Configure systemd service, timer, and journald logs.")
        plan = plan_component_edit(seed, rewards=[0.0])
        self.assertEqual(plan.projection, "reduce_complexity")
        self.assertTrue(all(value < 0 for value in plan.strengths.values()))

    def test_frontier_seed_keeps_diversify_and_touches_an_anchor(self):
        seed = make_seed("frontier-csv", "Merge csv files with deterministic tie-breaks.")
        plan = plan_component_edit(seed, rewards=[1.0, 0.0])
        self.assertEqual(plan.projection, "diversify_frontier")
        self.assertTrue(set(plan.touched) & {"instruction", "data"})
        self.assertTrue(all(abs(value) < 1e-9 for value in plan.strengths.values()))

    def test_consistency_implications_are_respected(self):
        seed = make_seed("csv", "Merge csv files.")
        for rewards in ([1.0], [0.0], [1.0, 0.0], [1, 1, 0, 1]):
            plan = plan_component_edit(seed, rewards=rewards)
            chosen = set(plan.touched)
            for source, target in CONSISTENCY_IMPLICATIONS:
                if source in chosen:
                    self.assertIn(
                        target,
                        chosen,
                        f"touching {source} without {target} breaks gradeability (rewards={rewards})",
                    )

    def test_solver_and_enumeration_agree(self):
        seed = make_seed("csv", "Merge csv files.")
        cases = ([1.0], [0.0], [1.0, 0.0], [1, 1, 0, 1], [0.75])
        for rewards in cases:
            with_solver = plan_component_edit(seed, rewards=rewards, prefer_solver=True)
            without = plan_component_edit(seed, rewards=rewards, prefer_solver=False)
            self.assertEqual(with_solver.touched, without.touched)
            self.assertAlmostEqual(with_solver.cost, without.cost, places=6)
            self.assertAlmostEqual(with_solver.delta_logit, without.delta_logit, places=6)

    def test_model_is_mixed_integer_programming(self):
        """Binary touch variables and continuous strengths must both be present."""
        seed = make_seed("csv", "Merge csv files.")
        plan = plan_component_edit(seed, rewards=[1.0])
        allowed = allowed_components(seed)
        rates = estimate_component_costs(seed)
        args = {
            "allowed": allowed,
            "z0": plan.z0,
            "band": 0.4,
            "e_max": DEFAULT_E_MAX,
            "fixed_cost": rates,
            "linear_cost": {component: 1.0 for component in EDIT_COMPONENTS},
        }
        self.assertTrue(_lp_relaxation_is_fractional(args), "LP relaxation should be fractional")

    def test_strength_is_interior_not_only_at_bounds(self):
        seed = make_seed("csv", "Merge csv files.")
        plan = plan_component_edit(seed, rewards=[1.0])
        values = [abs(value) for value in plan.strengths.values()]
        self.assertTrue(
            any(1e-6 < value < DEFAULT_E_MAX - 1e-6 for value in values),
            f"continuous strengths should carry real decisions, got {values}",
        )

    def test_unreachable_seed_reports_residual_and_rounds(self):
        seed = make_seed("very-hard", "Do an impossible amount of work.")
        plan = plan_component_edit(seed, rewards=[0.0], e_max=0.01)
        self.assertFalse(plan.reachable)
        self.assertGreater(plan.band_violation, 0.0)
        self.assertGreater(plan.rounds_needed, 1)

    def test_batch_prefers_reachable_plans(self):
        easy = make_seed("easy", "Merge csv files.")
        hard = make_seed("hard", "Configure systemd service, timer, journald logs, logrotate.")
        plans = plan_envs_forge_v1_batch(
            [hard, easy],
            rewards_by_seed={"easy": [1.0], "hard": [0.0]},
            target_count=2,
            e_max=1.0,
        )
        self.assertEqual(len(plans), 2)
        self.assertTrue(all(plan.reachable for plan in plans))
        self.assertEqual({plan.seed_name for plan in plans}, {"easy", "hard"})

    def test_rewards_lookup_does_not_prefix_collide(self):
        seed_a = make_seed("publisher-market-analysis", "Merge csv files.")
        seed_b = make_seed("publisher-market-analysis-v2", "Merge csv files.")
        plans = plan_envs_forge_v1_batch(
            [seed_a, seed_b],
            rewards_by_seed={
                "publisher-market-analysis": [1.0],
                "publisher-market-analysis-v2": [0.0],
            },
            target_count=2,
        )
        by_name = {plan.seed_name: plan for plan in plans}
        self.assertAlmostEqual(by_name["publisher-market-analysis"].pass_rate_raw, 1.0)
        self.assertAlmostEqual(by_name["publisher-market-analysis-v2"].pass_rate_raw, 0.0)

    def test_validate_subset_artifact_rejects_untouched_components(self):
        seed = make_seed("csv", "Merge csv files.")
        plan = plan_component_edit(seed, rewards=[1.0])
        self.assertEqual(plan.touched, ("tests",))
        bad = {"task_name": "x", "files": {"instruction.md": "hello"}}
        errors = validate_subset_artifact(bad, plan.touched)
        self.assertTrue(any("untouched component instruction" in error for error in errors))
        self.assertTrue(any("touched component tests has no generated file" in error for error in errors))

    def test_validate_subset_artifact_accepts_correct_subset(self):
        seed = make_seed("csv", "Merge csv files.")
        plan = plan_component_edit(seed, rewards=[1.0])
        good = {
            "task_name": "x",
            "files": {
                "tests/test.sh": "#!/usr/bin/env bash\necho 1 > /logs/verifier/reward.txt\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
            },
        }
        self.assertEqual(validate_subset_artifact(good, plan.touched), [])

    def test_expand_subset_artifact_inherits_untouched_components(self):
        seed = make_seed("csv", "Merge csv files.")
        plan = plan_component_edit(seed, rewards=[1.0])
        partial = {
            "task_name": "x",
            "files": {
                "tests/test.sh": "#!/usr/bin/env bash\necho 1 > /logs/verifier/reward.txt\n",
                "tests/test_outputs.py": "def test_ok():\n    assert True\n",
            },
            "metadata": {},
        }
        full = expand_subset_artifact(seed, partial, touched=plan.touched)
        self.assertIn("instruction.md", full["files"])
        self.assertIn("environment/Dockerfile", full["files"])
        self.assertIn("solution/solve.sh", full["files"])
        self.assertEqual(full["instruction_md"], seed.files["instruction.md"])
        self.assertIn("tests/test.sh", full["files"])

    def test_component_coverage_counts_generated_files(self):
        artifact = {"files": {"instruction.md": "x", "tests/test.sh": "y", "tests/test_outputs.py": "z"}}
        self.assertEqual(component_coverage(artifact), {"instruction": 1, "tests": 2})

    def test_prompt_only_lists_touched_components_and_forbids_the_rest(self):
        seed = make_seed("csv", "Merge csv files.")
        plan = plan_component_edit(seed, rewards=[1.0])
        prompt = build_component_edit_prompt(plan, seed, model="m")
        self.assertIn(METHOD_VERSION, prompt)
        self.assertIn("Touched", prompt.replace("touched", "Touched"))
        for component in plan.touched:
            self.assertIn(component, prompt)
        for component in EDIT_COMPONENTS:
            if component not in plan.touched:
                self.assertIn(component, prompt, "untouched components must be named so they are left alone")
        self.assertIn("Return ONLY files belonging to the touched components", prompt)

    def test_repair_prompt_keeps_component_set_fixed(self):
        seed = make_seed("csv", "Merge csv files.")
        plan = plan_component_edit(seed, rewards=[1.0])
        prompt = build_component_repair_prompt(
            plan,
            seed,
            previous_files={"tests/test.sh": "old"},
            failure_summary="gold failed",
            model="m",
            repair_index=1,
        )
        self.assertIn("do not add or remove components", prompt)
        self.assertIn("Do not return patch fragments", prompt)

    def test_component_costs_track_dockerfile_installs(self):
        plain = make_seed("plain", "Merge csv files.", {"environment/Dockerfile": "FROM ubuntu:22.04\n"})
        heavy = make_seed(
            "heavy",
            "Merge csv files.",
            {"environment/Dockerfile": "FROM ubuntu:22.04\nRUN apt-get install -y jq\n"},
        )
        self.assertLess(estimate_component_costs(plain)["environment"], estimate_component_costs(heavy)["environment"])

    def test_frontier_band_is_respected_exactly(self):
        seed = make_seed("csv", "Merge csv files.")
        for rewards in ([1.0], [0.0], [1, 0]):
            plan = plan_component_edit(seed, rewards=rewards)
            if plan.reachable:
                self.assertLessEqual(abs(plan.z_predicted), 0.4 + 1e-6)

    def test_z0_matches_smoothed_log_odds(self):
        seed = make_seed("csv", "Merge csv files.")
        plan = plan_component_edit(seed, rewards=[1.0])
        expected = math.log(plan.pass_rate_smoothed / (1 - plan.pass_rate_smoothed))
        self.assertAlmostEqual(plan.z0, expected, places=9)


if __name__ == "__main__":
    unittest.main()
