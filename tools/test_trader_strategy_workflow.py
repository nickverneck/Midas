import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("trader-strategy-workflow.py")
SPEC = importlib.util.spec_from_file_location("trader_strategy_workflow", MODULE_PATH)
workflow = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(workflow)


class AdapterTests(unittest.TestCase):
    def test_replay_is_fixed_and_prevalidated_without_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "trader" / "target" / "release" / "trader"
            binary.parent.mkdir(parents=True)
            binary.write_text("fake binary")
            spec = root / "sweep.json"
            spec.write_text(json.dumps({"schema_version": 1}))
            args = workflow.parser().parse_args([
                "run", "--repo", str(root), "--spec", str(spec),
                "--execution-mode", "batch_cpu", "--timeout", "20",
            ])
            calls = []

            def fake_run(argv, **kwargs):
                calls.append((argv, kwargs))
                class Result:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Result()

            with patch.object(workflow.subprocess, "run", side_effect=fake_run), patch("builtins.print"):
                self.assertEqual(workflow.run(args), 0)
            self.assertEqual(len(calls), 2)
            self.assertIn("validate-replay-sweep", calls[0][0])
            self.assertIn("run-replay-sweep", calls[1][0])
            self.assertIn("--execution-mode", calls[1][0])
            self.assertTrue(all("shell" not in kwargs for _, kwargs in calls))
            self.assertTrue(all(kwargs["timeout"] == 20 for _, kwargs in calls))

    def test_rank_does_not_require_or_run_a_sweep_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "trader" / "target" / "release" / "trader"
            binary.parent.mkdir(parents=True)
            binary.write_text("fake binary")
            summary = root / "sweep-summary.json"
            summary.write_text("{}")
            args = workflow.parser().parse_args([
                "rank", "--repo", str(root), "--summary", str(summary), "--metric", "robustness",
            ])
            with patch.object(workflow.subprocess, "run") as fake_run, patch("builtins.print"):
                fake_run.return_value.returncode = 0
                fake_run.return_value.stdout = "ranked"
                fake_run.return_value.stderr = ""
                self.assertEqual(workflow.run(args), 0)
            command = fake_run.call_args.args[0]
            self.assertIn("rank-replay-sweep", command)
            self.assertNotIn("validate-replay-sweep", command)

    def test_rejects_unknown_rank_metric_and_execution_mode(self):
        self.assertNotIn("anything-goes", workflow.RANK_METRICS)
        self.assertNotIn("anything-goes", workflow.EXECUTION_MODES)


if __name__ == "__main__":
    unittest.main()
