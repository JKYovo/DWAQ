"""CPU schedule/rollout/reset/sampling tests, using actual new environment hooks."""
import ast
from copy import copy, deepcopy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT/"TienKung-Lab/legged_lab/envs/elf3"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schedule = load(DIR/"reward_curriculum_contract.py", "test_reward_curriculum_contract")
peak_rewards = load(DIR/"foot_force_peak_rewards.py", "test_curriculum_peak_rewards")


class PhysicsParent:
    """CPU fake of only physics/sensors; execute real peak and course hooks below."""
    def init_buffers(self):
        self.device, self.num_envs = "cpu", 2
        self.cfg = SimpleNamespace(sim=SimpleNamespace(decimation=4))
        self.contact_sensor = SimpleNamespace(
            find_bodies=lambda names, preserve_order: ([2, 0], tuple(names)),
            data=SimpleNamespace(net_forces_w=torch.zeros(2, 3, 3)))
        self.feedback_calls = 0

    def _post_physics_step(self):
        self.feedback_calls += 1

    def reset(self, env_ids):
        pass

    def step(self, actions):
        for left, right in actions:
            self.contact_sensor.data.net_forces_w[:, 2, 2] = left
            self.contact_sensor.data.net_forces_w[:, 0, 2] = right
            self._post_physics_step()
        return peak_rewards.body_force_control_peak(self, None, threshold=500.)


peak_scope = {"Elf3DwaqControlSmoothnessEnv": PhysicsParent, "torch": torch,
              "FOOT_BODY_NAMES": ("l_ankle_x_link", "r_ankle_x_link"),
              "accumulate_peak": peak_rewards.accumulate_peak}
peak_tree = ast.parse((DIR/"foot_force_peak.py").read_text())
exec(compile(ast.Module(body=[n for n in peak_tree.body if isinstance(n, ast.ClassDef)
                            and n.name == "Elf3DwaqFootForcePeakEnv"], type_ignores=[]),
             str(DIR/"foot_force_peak.py"), "exec"), peak_scope)
peak = SimpleNamespace(rewards=peak_rewards, functions=SimpleNamespace(
    Elf3DwaqFootForcePeakEnv=peak_scope["Elf3DwaqFootForcePeakEnv"]))


class Manager:
    def __init__(self):
        self.original = {k: SimpleNamespace(weight=b, func=k, params={"unchanged": True})
                         for k, (_, b) in schedule.WEIGHT_ENDPOINTS.items()}
        self.terms = deepcopy(self.original)

    def get_term_cfg(self, name):
        return self.terms[name]

    def set_term_cfg(self, name, term):
        self.terms[name] = term


scope = {"Elf3DwaqFootForce500Env": peak.functions.Elf3DwaqFootForcePeakEnv,
         "validate_settings": schedule.validate_settings, "RolloutClock": schedule.RolloutClock,
         "apply_weights": schedule.apply_weights, "curriculum_alpha": schedule.curriculum_alpha}
tree = ast.parse((DIR/"reward_curriculum.py").read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)
                            and n.name == "Elf3DwaqRewardCurriculumEnv"], type_ignores=[]),
             str(DIR/"reward_curriculum.py"), "exec"), scope)
Env = scope["Elf3DwaqRewardCurriculumEnv"]


class RunnerParent:
    def contract(self):
        return {"base_contract": "unchanged"}

    def save(self, path, infos=None):
        self.saved = (path, infos)

    def learn(self, count, random_lengths):
        if getattr(self, "fail_during_learn", False):
            raise RuntimeError("test interruption")
        return "unchanged parent learning loop"


class ModelLoader:
    @staticmethod
    def load(runner, path, load_optimizer=True):
        runner.loader_called = True


feedback = load(DIR/"feedback_contract.py", "test_curriculum_resume_contract")
runner_scope = {"Elf3DwaqFootForce500Runner": RunnerParent, "deepcopy": deepcopy,
                "torch": torch, "DWAQOnPolicyRunner": ModelLoader,
                "validate_checkpoint_contract": feedback.validate_checkpoint_contract,
                "curriculum_contract": schedule.curriculum_contract,
                "curriculum_alpha": schedule.curriculum_alpha,
                "effective_weights": schedule.effective_weights}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)
                            and n.name == "Elf3DwaqRewardCurriculumRunner"], type_ignores=[]),
             str(DIR/"reward_curriculum.py"), "exec"), runner_scope)
Runner = runner_scope["Elf3DwaqRewardCurriculumRunner"]


class RewardCurriculumTests(unittest.TestCase):
    def test_default_and_early_schedule_boundaries(self):
        for start, end in ((20000, 30000), (18500, 28500)):
            settings = schedule.schedule_settings(start, end)
            for iteration, expected in ((0, 0.), (start-1, 0.), (start, 0.),
                                        (start+1, .0001), ((start+end)//2, .5),
                                        (end, 1.), (50000, 1.)):
                with self.subTest(start=start, iteration=iteration):
                    self.assertEqual(schedule.curriculum_alpha(iteration, settings), expected)

    def test_exact_endpoint_and_linear_weights(self):
        settings = schedule.schedule_settings()
        for iteration, alpha in ((20000, 0.), (25000, .5), (30000, 1.)):
            weights = schedule.effective_weights(iteration, settings)
            self.assertEqual(set(weights), set(schedule.WEIGHT_ENDPOINTS))
            for name, (a, b) in schedule.WEIGHT_ENDPOINTS.items():
                self.assertAlmostEqual(weights[name], (1-alpha)*a+alpha*b)
        self.assertEqual(schedule.effective_weights(50000, settings)["dof_acc_l2"], -3e-7)
        self.assertEqual(schedule.effective_weights(0, settings)["feet_force"], -.003)
        self.assertEqual(schedule.effective_weights(50000, settings)["feet_force"], -.005)

    def test_invalid_or_unknown_schedules_rejected(self):
        for a, b in ((0,0), (30000,20000), (-1,30000), (True,30000), (2.5,30000)):
            with self.subTest(a=a,b=b), self.assertRaises(ValueError):
                schedule.schedule_settings(a,b)
        for key, value in (("feet_force_threshold_n",400), ("clock","episode"), ("extra",1)):
            settings = schedule.schedule_settings(); settings[key] = value
            with self.assertRaises(ValueError): schedule.validate_settings(settings)
        for iteration in (-1, True, .5):
            with self.assertRaises(ValueError):
                schedule.curriculum_alpha(iteration, schedule.schedule_settings())

    def test_updates_actual_manager_preserve_config_and_callbacks(self):
        manager = Manager()
        settings = schedule.schedule_settings()
        schedule.apply_weights(manager, 0, settings)
        for name, (start, final) in schedule.WEIGHT_ENDPOINTS.items():
            self.assertEqual(manager.terms[name].weight,start)
            self.assertEqual(manager.original[name].weight,final)
            self.assertEqual(manager.terms[name].func,name)
            self.assertEqual(manager.terms[name].params,{"unchanged":True})
        schedule.apply_weights(manager,30000,settings)
        for name, (_, final) in schedule.WEIGHT_ENDPOINTS.items():
            self.assertEqual(manager.terms[name].weight,final)

    def test_rollout_clock_fixed_for_all24_steps_and_resume_no_restart(self):
        clock = schedule.RolloutClock(); clock.begin(18501,24)
        seen=[]
        for _ in range(48):
            seen.append(clock.iteration); clock.advance()
        self.assertEqual(seen,[18501]*24+[18502]*24)
        clock.end(); before=clock.iteration; clock.advance()
        self.assertEqual(clock.iteration,before)
        clock.begin(25001,24)
        self.assertEqual(clock.iteration,25001)
        self.assertEqual(schedule.curriculum_alpha(clock.iteration,schedule.schedule_settings()),.5001)

    def env(self, iteration=0):
        env = Env(); env.init_buffers(); env.reward_manager=Manager()
        env.configure_reward_curriculum(schedule.schedule_settings(),iteration)
        return env

    def test_real_step_hooks_advance_once_per_rollout_not_substep(self):
        env=self.env(24999); env.reward_curriculum_clock.begin(24999,24)
        seen=[]
        for _ in range(49):
            env.step([(300.,0.)]*4)
            seen.append(env.reward_curriculum_iteration)
        self.assertEqual(seen,[24999]*24+[25000]*24+[25001])
        self.assertEqual(env.feedback_calls,49*4)

    def test_episode_reset_never_resets_global_schedule(self):
        env=self.env(25000); env.reward_curriculum_clock.begin(25000,24)
        for _ in range(7):env.step([(700.,0.)]*4)
        env.reset([0])
        self.assertEqual(env.reward_curriculum_clock.steps,7)
        self.assertEqual(env.reward_curriculum_alpha,.5)
        self.assertEqual(env.reward_curriculum_weights,schedule.effective_weights(25000,schedule.schedule_settings()))
        self.assertEqual(float(env.force_peak_metric[0]),0.)
        self.assertEqual(float(env.force_peak_metric[1]),700.)

    def test_play_steps_do_not_advance_schedule(self):
        env=self.env(25000)
        for _ in range(30):env.step([(300.,0.)]*4)
        self.assertEqual(env.reward_curriculum_iteration,25000)
        self.assertEqual(env.reward_curriculum_alpha,.5)

    def test_all_stages_capture_all_substeps_at500N_and_control_dt_once(self):
        for iteration in (0,20000,25000,30000,50000):
            for pulse in range(4):
                env=self.env(iteration)
                samples=[(200.,0.)]*4; samples[pulse]=(700.,0.)
                env.step(samples)
                raw=peak.rewards.force_excess(env.force_peak_metric,500.,400.)
                torch.testing.assert_close(raw,torch.full((2,),200.))
                actual=raw*env.reward_manager.terms["feet_force"].weight*.02
                expected=200.*schedule.effective_weights(iteration,schedule.schedule_settings())["feet_force"]*.02
                torch.testing.assert_close(actual,torch.full((2,),expected))

    def test_contract_records_global_sampling_endpoints_and_sources(self):
        c=schedule.curriculum_contract(schedule.schedule_settings())
        self.assertEqual(c["feet_force_threshold_n"],500.)
        self.assertEqual(c["feet_force_excess_cap_n"],400.)
        self.assertIn("global all stages",c["force_sampling"])
        self.assertEqual(set(c["sources_sha256"]),{"reward_curriculum.py","reward_curriculum_contract.py"})
        self.assertTrue(all(len(x)==64 for x in c["sources_sha256"].values()))

    def runner(self, iteration=25000):
        runner=object.__new__(Runner)
        runner.reward_curriculum_settings=schedule.schedule_settings()
        runner.env=self.env(iteration)
        runner.current_learning_iteration=iteration
        runner.num_steps_per_env=24
        return runner

    def test_checkpoint_save_records_effective_progress_and_rejects_misalignment(self):
        runner=self.runner(25000)
        runner.save("model_25000.pt")
        state=runner.saved[1]["reward_curriculum_state"]
        self.assertEqual(state["completed_iteration"],25000)
        self.assertEqual(state["next_iteration"],25001)
        self.assertEqual(state["alpha"],.5)
        self.assertEqual(state["weights"]["base_height"],-25.)
        runner.env.set_training_iteration(25001)
        with self.assertRaises(ValueError):runner.save("bad.pt")

    def test_actual_resume_loader_keeps_mid_ramp_progress_and_freezes_play(self):
        runner=self.runner(25000)
        payload={"iter":25000,"infos":{"elf3_contract":runner.contract(),
                  "reward_curriculum_state":runner.curriculum_state(25000)}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model_25000.pt";torch.save(payload,path)
            runner.load(path,load_optimizer=True)
            self.assertEqual(runner.current_learning_iteration,25001)
            self.assertEqual(runner.env.reward_curriculum_alpha,.5001)
            runner.load(path,load_optimizer=False)
            self.assertEqual(runner.env.reward_curriculum_alpha,.5)
            self.assertFalse(runner.env.reward_curriculum_clock.active)

    def test_resume_rejects_tampered_progress_contract_or_initial_snapshot(self):
        runner=self.runner()
        good={"iter":25000,"infos":{"elf3_contract":runner.contract(),
               "reward_curriculum_state":runner.curriculum_state(25000)}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"bad.pt"
            for mutate in (lambda p:p["infos"]["reward_curriculum_state"].update(alpha=0.),
                           lambda p:p["infos"]["reward_curriculum_state"].update(next_iteration=20000),
                           lambda p:p["infos"]["reward_curriculum_state"]["weights"].update(base_height=0.),
                           lambda p:p["infos"]["elf3_contract"].update(extra="different"),
                           lambda p:p["infos"].update(initialization_only=True)):
                payload=deepcopy(good);mutate(payload);torch.save(payload,path)
                with self.assertRaises(ValueError):runner.load(path)

    def test_curriculum_clock_is_disabled_even_if_parent_learning_raises(self):
        runner=self.runner()
        runner.fail_during_learn=True
        with self.assertRaisesRegex(RuntimeError,"test interruption"):
            runner.learn(1)
        self.assertFalse(runner.env.reward_curriculum_clock.active)


if __name__ == "__main__":
    unittest.main()
