"""CPU-only regression checks; no optimization or training run."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from environment import EnvironmentConfig, VectorizedSharkEnv
from policy import CentralizedAttentionPolicy
from ppo import PPOConfig, PPOTrainer
from train import initialize_actor, validation_improves


class AllTenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cuda_guard = patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden"))
        cls.cuda_guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.cuda_guard.stop()

    def environment(self, **overrides):
        config = replace(
            EnvironmentConfig(), progress_reward_scale=0,
            trailing_slow_progress_reward_scale=0, **overrides,
        )
        env = VectorizedSharkEnv(1, "cpu", config, seed=3)
        env.minnow_speeds.fill_(0.2)
        env.minnow_speeds[0, -1] = 1
        return env

    def actions(self):
        return torch.zeros((1, 10, 2))

    def collide(self, env, index):
        env.minnow_positions[0, index] = torch.tensor([1.0, 0.5])
        env.shark_positions[0] = env.minnow_positions[0, index]
        return env.step(self.actions())

    def test_first_slow_death_immediate_and_failure_charged_once(self):
        env = self.environment()
        first = self.collide(env, 0)
        self.assertEqual(first.individual_rewards[0, 0].item(), -1)
        self.assertEqual(first.team_rewards.item(), -11)
        self.assertFalse(first.done.item())
        second = self.collide(env, 1)
        self.assertEqual(second.team_rewards.item(), -1)
        self.assertEqual(env.step(self.actions()).team_rewards.item(), 0)
        env.elapsed_seconds.fill_(env.config.episode_seconds)
        timeout = env.step(self.actions())
        self.assertEqual(timeout.team_rewards.item(), -8)
        self.assertEqual(timeout.individual_rewards[0, :2].sum().item(), 0)
        self.assertEqual(env.step(self.actions()).team_rewards.item(), 0)

    def test_multiple_deaths_only_one_team_failure(self):
        env = self.environment()
        env.minnow_positions[0, :2] = torch.tensor([1.0, 0.5])
        result = self.collide(env, 0)
        self.assertEqual(result.newly_dead.sum().item(), 2)
        self.assertEqual(result.team_rewards.item(), -12)

    def test_timeout_without_death_is_also_failure(self):
        env = self.environment()
        env.elapsed_seconds.fill_(env.config.episode_seconds)
        result = env.step(self.actions())
        self.assertTrue(result.timed_out.item())
        self.assertEqual(result.team_rewards.item(), -20)
        self.assertEqual(env.step(self.actions()).team_rewards.item(), 0)
        env.reset()
        slow_index = int((env.minnow_speeds[0] < env.config.shark_speed).nonzero()[0].item())
        self.assertEqual(self.collide(env, slow_index).team_rewards.item(), -11)

    def test_all_ten_arrivals_and_terminal_bonus_paid_once(self):
        env = self.environment()
        env.minnow_positions[..., 0] = env.config.active_right - 0.001
        actions = self.actions()
        actions[..., 0] = 1
        result = env.step(actions)
        self.assertEqual(result.safe_counts.item(), 10)
        self.assertEqual(result.team_rewards.item(), 30)
        self.assertEqual(env.step(actions).team_rewards.item(), 0)

    def test_nine_saved_sacrifice_return_below_perfect(self):
        env = self.environment()
        total = self.collide(env, 0).team_rewards.item()
        env.minnow_positions[0, 1:, 0] = env.config.active_right - 0.001
        actions = self.actions()
        actions[..., 0] = 1
        result = env.step(actions)
        total += result.team_rewards.item()
        self.assertEqual(result.safe_counts.item(), 9)
        self.assertEqual(total, 7)  # 9 crossings + 9 safe - 1 death - 10 failure

    def test_discounted_potential_telescopes_through_safe_fraction_changes(self):
        config = EnvironmentConfig(progress_reward_scale=0)
        shaped = VectorizedSharkEnv(1, "cpu", config, seed=1)
        plain = VectorizedSharkEnv(
            1, "cpu", replace(config, trailing_slow_progress_reward_scale=0), seed=1,
        )
        for env in (shaped, plain):
            env.minnow_speeds.fill_(0.2)
            env.minnow_positions[..., 0] = env.config.active_right - 0.08
            env.minnow_positions[0, :5, 0] = env.config.active_right - 0.001
            env.shark_positions[0] = torch.tensor([env.config.active_left, 0.5])
        initial = shaped._trailing_slow_potential().item()
        total = 0.0
        actions = self.actions()
        actions[..., 0] = 1
        for step in range(10):
            a, b = shaped.step(actions), plain.step(actions)
            total += config.shaping_discount_factor ** step * (a.team_rewards - b.team_rewards).item()
            if a.done.item():
                break
        self.assertTrue(a.done.item())
        self.assertAlmostEqual(total, -initial, places=4)
        self.assertEqual(shaped._trailing_slow_potential().item(), 0)
        self.assertEqual(shaped.step(actions).team_rewards.item(), 0)

    def test_no_slow_minnows_potential_finite(self):
        env = VectorizedSharkEnv(1, "cpu", EnvironmentConfig(fast_minnow_count=10))
        self.assertEqual(env._trailing_slow_potential().item(), 0)
        self.assertTrue(torch.isfinite(env.step(self.actions()).team_rewards).all())

    def test_perfect_rate_beats_mean_saved(self):
        self.assertTrue(validation_improves(9.6, 0.94, 9.9, 0.90))
        self.assertFalse(validation_improves(9.9, 0.90, 9.6, 0.94))
        self.assertTrue(validation_improves(9.9, 0.94, 9.6, 0.94))

    def test_actor_initialization_keeps_fresh_critics(self):
        old = CentralizedAttentionPolicy()
        new = CentralizedAttentionPolicy()
        fresh = {k: v.clone() for k, v in new.state_dict().items()}
        initialize_actor(new, {"policy_state_dict": old.state_dict()})
        for name, value in new.state_dict().items():
            expected = fresh[name] if "value_head." in name else old.state_dict()[name]
            self.assertTrue(torch.equal(value, expected), name)

    def test_cpu_rollout_finite_and_discount_mismatch_rejected(self):
        env = self.environment(episode_seconds=0.2)
        policy = CentralizedAttentionPolicy()
        with self.assertRaisesRegex(ValueError, "same discount"):
            PPOTrainer(policy, env, PPOConfig(discount_factor=0.9))
        trainer = PPOTrainer(policy, env, PPOConfig(rollout_steps=4))
        batch, metrics = trainer.collect_rollout()
        self.assertGreater(metrics.completed_episodes, 0)
        for value in (batch.mixed_advantages, batch.team_returns, batch.individual_returns):
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(torch.isfinite(value).all())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
