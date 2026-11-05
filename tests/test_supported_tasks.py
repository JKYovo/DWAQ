"""Task cleanup guards; no Isaac, GPU, training, or robot is started."""
import ast
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("supported_elf3_catalog", ROOT / "scripts/elf3_task_catalog.py")
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


class SupportedTaskTests(unittest.TestCase):
    def test_exact_two_public_tasks(self):
        expected = {"elf3_dwaq_feedback_v1", "elf3_dwaq_still_v2_ankle_cost_v1"}
        source = ROOT / "TienKung-Lab/legged_lab/envs/__init__.py"
        tree = ast.parse(source.read_text())
        names = [node.args[0].value for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "register"]
        self.assertEqual(len(names), 2)
        self.assertEqual(set(names), expected)
        self.assertEqual(set(catalog.TASKS), expected)

    def test_registration_uses_unchanged_factories_and_exact_runner_exports(self):
        """Execute actual registry code with only heavyweight imports stubbed."""
        registered, factory_calls, objects = {}, [], {}

        def register(name, environment, cfg, agent):
            registered[name] = (environment, cfg, agent)

        def factory(name):
            def build():
                factory_calls.append(name)
                return objects[name]
            return build

        def module(name, **values):
            result = ModuleType(name)
            result.__dict__.update(values)
            return result

        def identity(name):
            return type(name, (), {})

        prefix = "legged_lab.envs.elf3."
        objects["ankle"] = (object(), object())
        feedback_env = identity("Elf3DwaqFeedbackEnv")
        ankle_env = identity("Elf3DwaqAnkleCostEnv")
        runners = {name: identity(name) for name in (
            "Elf3DwaqFeedbackRunner", "Elf3DwaqAnkleCostRunner")}
        stubs = {
            "legged_lab.utils.task_registry": module("registry", task_registry=SimpleNamespace(register=register)),
            prefix + "elf3_dwaq_env": module("base", Elf3DwaqEnv=identity("Elf3DwaqEnv")),
            prefix + "elf3_feedback_env": module("feedback", Elf3DwaqFeedbackEnv=feedback_env),
            prefix + "elf3_feedback_config": module("config", Elf3DwaqFeedbackEnvCfg=identity("EnvCfg"),
                                                    Elf3DwaqFeedbackAgentCfg=identity("AgentCfg")),
            prefix + "feedback_runner": module("runner", Elf3DwaqFeedbackRunner=runners["Elf3DwaqFeedbackRunner"]),
            prefix + "ankle_cost": module("ankle", Elf3DwaqAnkleCostEnv=ankle_env,
                                          Elf3DwaqAnkleCostRunner=runners["Elf3DwaqAnkleCostRunner"],
                                          make_configs=factory("ankle")),
        }
        namespace = {}
        with patch.dict(sys.modules, stubs):
            source = ROOT / "TienKung-Lab/legged_lab/envs/__init__.py"
            exec(compile(source.read_text(), str(source), "exec"), namespace)
        self.assertEqual(factory_calls, ["ankle"])
        self.assertEqual(registered["elf3_dwaq_feedback_v1"][0], feedback_env)
        self.assertEqual(registered["elf3_dwaq_still_v2_ankle_cost_v1"], (ankle_env, *objects["ankle"]))
        for name, runner in runners.items():
            self.assertIn(name, namespace["__all__"])
            self.assertIs(namespace[name], runner)

    def test_retired_launchers_fail_before_import_or_simulator_start(self):
        for entrypoint in catalog.RETIRED_ENTRYPOINTS:
            with self.subTest(entrypoint=entrypoint), self.assertRaisesRegex(ValueError, "Retired training"):
                catalog.check_entrypoint([str(ROOT / entrypoint), "--headless"])

    def test_retained_entrypoints_export_and_python_checks_are_allowed(self):
        for entrypoint in (*catalog.TASKS.values(), "scripts/export_elf3_dwaq.py"):
            catalog.check_entrypoint([str(ROOT / entrypoint), "--help"])
        catalog.check_entrypoint(["-m", "unittest"])
        catalog.check_entrypoint(["-c", "print('ok')"])
        catalog.check_entrypoint([])

    def test_generic_task_flags_allow_only_supported_tasks(self):
        for entrypoint in ("TienKung-Lab/legged_lab/scripts/train.py", "TienKung-Lab/legged_lab/scripts/play.py"):
            for task in catalog.TASKS:
                for flags in (["--task", task], ["--task=" + task]):
                    catalog.check_entrypoint([str(ROOT / entrypoint), *flags])
            for task in ("elf3_dwaq_still_v2", "elf3_dwaq_consistent_v1", "g1_dwaq", "walk"):
                with self.subTest(task=task), self.assertRaisesRegex(ValueError, "Retired/unknown task"):
                    catalog.check_entrypoint([str(ROOT / entrypoint), "--task=" + task])

    def test_catalog_is_cpu_only_and_does_not_write_or_signal(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(catalog.main([]), 0)
        for task in catalog.TASKS:
            self.assertIn(task, output.getvalue())
        error = io.StringIO()
        with redirect_stderr(error):
            code = catalog.main(["--check-entrypoint", str(ROOT / "scripts/train_elf3_still_v2.py")])
        self.assertEqual(code, 2)
        tree = ast.parse((ROOT / "scripts/elf3_task_catalog.py").read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imports, {"pathlib", "sys"})

    def test_shell_rejects_retired_command_before_python_module_runs(self):
        result = subprocess.run(["bash", str(ROOT / "elf3.sh"), str(ROOT / "scripts/train_elf3_still_v2.py"),
                                 "--help"], capture_output=True, text=True, timeout=15, cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Retired training entrypoint", result.stderr)
        self.assertNotIn("Isaac Sim", result.stdout)


if __name__ == "__main__":
    unittest.main()
