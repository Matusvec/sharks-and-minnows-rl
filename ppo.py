"""PPO training with mixed team and individual advantages."""

from dataclasses import dataclass
import copy
import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from environment import VectorizedSharkEnv
from policy import CentralizedAttentionPolicy, PolicyOutput


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_states: int = 2048
    learning_rate: float = 2e-4
    discount_factor: float = 0.999
    gae_lambda: float = 0.99
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.002
    team_value_coefficient: float = 0.25
    individual_value_coefficient: float = 0.25
    max_gradient_norm: float = 0.5
    team_advantage_weight: float = 1.0
    individual_advantage_weight: float = 0.0
    target_kl: float = 0.02
    rollback_kl: float = 0.05
    minimum_learning_rate: float = 1e-5
    rollback_learning_rate_factor: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.discount_factor <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("discount_factor and gae_lambda must be in [0, 1]")
        if not math.isclose(
            self.team_advantage_weight + self.individual_advantage_weight,
            1.0,
        ):
            raise ValueError("team and individual advantage weights must sum to 1")
        if self.target_kl <= 0:
            raise ValueError("target_kl must be positive")
        if self.rollback_kl <= self.target_kl:
            raise ValueError("rollback_kl must be greater than target_kl")
        if self.minimum_learning_rate <= 0:
            raise ValueError("minimum_learning_rate must be positive")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError(
                "minimum_learning_rate cannot exceed learning_rate"
            )
        if not 0 < self.rollback_learning_rate_factor < 1:
            raise ValueError(
                "rollback_learning_rate_factor must be between zero and one"
            )


@dataclass(frozen=True)
class RolloutBatch:
    minnow_features: Tensor
    global_features: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    active_masks: Tensor
    mixed_advantages: Tensor
    team_returns: Tensor
    individual_returns: Tensor


@dataclass
class RolloutMetrics:
    completed_episodes: int = 0
    total_safe: int = 0
    total_dead: int = 0
    timed_out_episodes: int = 0
    slow_minnows: int = 0
    slow_minnows_safe: int = 0
    fast_minnows: int = 0
    fast_minnows_safe: int = 0
    perfect_games: int = 0
    fast_target_fraction_total: float = 0.0
    fast_decoy_score_total: float = 0.0
    environment_steps: int = 0

    @property
    def mean_safe(self) -> float:
        if self.completed_episodes == 0:
            return float("nan")
        return self.total_safe / self.completed_episodes

    @property
    def slow_survival_rate(self) -> float:
        if self.slow_minnows == 0:
            return float("nan")
        return self.slow_minnows_safe / self.slow_minnows

    @property
    def fast_survival_rate(self) -> float:
        if self.fast_minnows == 0:
            return float("nan")
        return self.fast_minnows_safe / self.fast_minnows

    @property
    def perfect_game_rate(self) -> float:
        if self.completed_episodes == 0:
            return float("nan")
        return self.perfect_games / self.completed_episodes

    @property
    def fast_target_rate(self) -> float:
        if self.environment_steps == 0:
            return float("nan")
        return self.fast_target_fraction_total / self.environment_steps

    @property
    def fast_decoy_quality(self) -> float:
        if self.environment_steps == 0:
            return float("nan")
        return self.fast_decoy_score_total / self.environment_steps


def sample_actions(output: PolicyOutput) -> tuple[Tensor, Tensor]:
    """Sample tanh-bounded actions and return per-minnow log probabilities."""
    distribution = Normal(output.direction_logits, output.direction_log_std.exp())
    unsquashed_actions = distribution.rsample()
    actions = torch.tanh(unsquashed_actions)
    log_probabilities = (
        distribution.log_prob(unsquashed_actions)
        - torch.log(1 - actions.square() + 1e-6)
    ).sum(dim=-1)
    return actions, log_probabilities


def action_log_probabilities(
    output: PolicyOutput, actions: Tensor
) -> tuple[Tensor, Tensor]:
    """Evaluate stored bounded actions under the current policy."""
    bounded_actions = actions.clamp(-1 + 1e-6, 1 - 1e-6)
    unsquashed_actions = torch.atanh(bounded_actions)
    distribution = Normal(output.direction_logits, output.direction_log_std.exp())
    log_probabilities = (
        distribution.log_prob(unsquashed_actions)
        - torch.log(1 - bounded_actions.square() + 1e-6)
    ).sum(dim=-1)
    # Base-normal entropy is a stable exploration proxy for the squashed policy.
    return log_probabilities, distribution.entropy().sum(dim=-1)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    numeric_mask = mask.to(values.dtype)
    return (values * numeric_mask).sum() / numeric_mask.sum().clamp_min(1)


class PPOTrainer:
    def __init__(
        self,
        policy: CentralizedAttentionPolicy,
        environment: VectorizedSharkEnv,
        config: PPOConfig | None = None,
    ) -> None:
        self.policy = policy
        self.environment = environment
        self.config = config or PPOConfig()
        if not math.isclose(
            self.config.discount_factor, environment.config.shaping_discount_factor
        ):
            raise ValueError("PPO and potential shaping must use the same discount")
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )
        self.minnow_features, self.global_features = environment.observe()

    @property
    def learning_rate(self) -> float:
        """Return the optimizer's current learning rate."""
        return float(self.optimizer.param_groups[0]["lr"])

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set every Adam parameter group's learning rate."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def reduce_learning_rate_after_rollback(self) -> tuple[float, float]:
        """Lower the learning rate after restoring a destructive PPO update."""
        before = self.learning_rate
        after = max(
            self.config.minimum_learning_rate,
            before * self.config.rollback_learning_rate_factor,
        )
        self.set_learning_rate(after)
        return before, after

    @torch.no_grad()
    def collect_rollout(self) -> tuple[RolloutBatch, RolloutMetrics]:
        config = self.config
        environment = self.environment
        device = environment.device

        minnow_feature_steps: list[Tensor] = []
        global_feature_steps: list[Tensor] = []
        action_steps: list[Tensor] = []
        log_probability_steps: list[Tensor] = []
        active_mask_steps: list[Tensor] = []
        team_value_steps: list[Tensor] = []
        individual_value_steps: list[Tensor] = []
        team_reward_steps: list[Tensor] = []
        individual_reward_steps: list[Tensor] = []
        done_steps: list[Tensor] = []
        individual_terminal_steps: list[Tensor] = []
        metrics = RolloutMetrics()

        self.policy.eval()
        for _ in range(config.rollout_steps):
            output = self.policy(self.minnow_features, self.global_features)
            actions, log_probabilities = sample_actions(output)
            active_masks = environment.active_mask.clone()

            minnow_feature_steps.append(self.minnow_features)
            global_feature_steps.append(self.global_features)
            action_steps.append(actions)
            log_probability_steps.append(log_probabilities)
            active_mask_steps.append(active_masks)
            team_value_steps.append(output.team_value)
            individual_value_steps.append(output.individual_values)

            result = environment.step(actions)
            metrics.fast_target_fraction_total += float(
                result.fast_target_fraction.sum().item()
            )
            metrics.fast_decoy_score_total += float(
                result.fast_decoy_score.sum().item()
            )
            metrics.environment_steps += environment.environment_count
            team_reward_steps.append(result.team_rewards)
            individual_reward_steps.append(result.individual_rewards)
            done_steps.append(result.done)
            individual_terminal_steps.append(result.individual_terminal)

            completed = result.done
            if completed.any():
                safe = environment.minnow_safe[completed]
                speeds = environment.minnow_speeds[completed]
                metrics.completed_episodes += int(completed.sum().item())
                metrics.total_safe += int(safe.sum().item())
                metrics.total_dead += int(result.dead_counts[completed].sum().item())
                metrics.timed_out_episodes += int(result.timed_out[completed].sum().item())
                slow = speeds < environment.config.shark_speed
                metrics.slow_minnows += int(slow.sum().item())
                metrics.slow_minnows_safe += int((slow & safe).sum().item())
                fast = speeds > environment.config.shark_speed
                metrics.fast_minnows += int(fast.sum().item())
                metrics.fast_minnows_safe += int((fast & safe).sum().item())
                metrics.perfect_games += int(
                    (safe.sum(dim=1) == environment.config.minnow_count)
                    .sum()
                    .item()
                )
                environment.reset(completed)

            self.minnow_features, self.global_features = environment.observe()

        final_output = self.policy(self.minnow_features, self.global_features)

        minnow_features = torch.stack(minnow_feature_steps)
        global_features = torch.stack(global_feature_steps)
        actions = torch.stack(action_steps)
        old_log_probabilities = torch.stack(log_probability_steps)
        active_masks = torch.stack(active_mask_steps)
        team_values = torch.stack(team_value_steps)
        individual_values = torch.stack(individual_value_steps)
        team_rewards = torch.stack(team_reward_steps)
        individual_rewards = torch.stack(individual_reward_steps)
        dones = torch.stack(done_steps)
        individual_terminal = torch.stack(individual_terminal_steps)

        team_advantages = torch.zeros_like(team_rewards, device=device)
        individual_advantages = torch.zeros_like(
            individual_rewards, device=device
        )
        next_team_advantage = torch.zeros(
            environment.environment_count, device=device
        )
        next_individual_advantage = torch.zeros(
            (
                environment.environment_count,
                environment.config.minnow_count,
            ),
            device=device,
        )
        next_team_value = final_output.team_value
        next_individual_values = final_output.individual_values

        for step in reversed(range(config.rollout_steps)):
            team_nonterminal = (~dones[step]).float()
            individual_nonterminal = (~individual_terminal[step]).float()

            team_delta = (
                team_rewards[step]
                + config.discount_factor * next_team_value * team_nonterminal
                - team_values[step]
            )
            next_team_advantage = (
                team_delta
                + config.discount_factor
                * config.gae_lambda
                * team_nonterminal
                * next_team_advantage
            )
            team_advantages[step] = next_team_advantage

            individual_delta = (
                individual_rewards[step]
                + config.discount_factor
                * next_individual_values
                * individual_nonterminal
                - individual_values[step]
            )
            next_individual_advantage = (
                individual_delta
                + config.discount_factor
                * config.gae_lambda
                * individual_nonterminal
                * next_individual_advantage
            )
            individual_advantages[step] = next_individual_advantage

            next_team_value = team_values[step]
            next_individual_values = individual_values[step]

        # Preserve unnormalized GAE returns as critic targets. Advantage
        # normalization below is only for stabilizing the actor update.
        team_returns = team_advantages + team_values
        individual_returns = individual_advantages + individual_values

        team_agent_advantages = team_advantages.unsqueeze(-1).expand_as(
            individual_advantages
        )
        active_team_advantages = team_agent_advantages[active_masks]
        if active_team_advantages.numel() > 1:
            team_agent_advantages = (
                team_agent_advantages - active_team_advantages.mean()
            ) / (active_team_advantages.std(unbiased=False) + 1e-8)

        active_individual_advantages = individual_advantages[active_masks]
        if active_individual_advantages.numel() > 1:
            individual_advantages = (
                individual_advantages - active_individual_advantages.mean()
            ) / (active_individual_advantages.std(unbiased=False) + 1e-8)

        mixed_advantages = (
            config.team_advantage_weight * team_agent_advantages
            + config.individual_advantage_weight * individual_advantages
        )
        active_advantages = mixed_advantages[active_masks]
        if active_advantages.numel() > 1:
            mixed_advantages = (
                mixed_advantages - active_advantages.mean()
            ) / (active_advantages.std(unbiased=False) + 1e-8)

        batch = RolloutBatch(
            minnow_features=minnow_features,
            global_features=global_features,
            actions=actions,
            old_log_probabilities=old_log_probabilities,
            active_masks=active_masks,
            mixed_advantages=mixed_advantages,
            team_returns=team_returns,
            individual_returns=individual_returns,
        )
        return batch, metrics

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        config = self.config
        rollout_steps, environment_count = batch.global_features.shape[:2]
        state_count = rollout_steps * environment_count

        minnow_features = batch.minnow_features.flatten(0, 1)
        global_features = batch.global_features.flatten(0, 1)
        actions = batch.actions.flatten(0, 1)
        old_log_probabilities = batch.old_log_probabilities.flatten(0, 1)
        active_masks = batch.active_masks.flatten(0, 1)
        mixed_advantages = batch.mixed_advantages.flatten(0, 1)
        team_returns = batch.team_returns.flatten(0, 1)
        individual_returns = batch.individual_returns.flatten(0, 1)

        metric_totals = {
            "loss": 0.0,
            "actor_loss": 0.0,
            "team_value_loss": 0.0,
            "individual_value_loss": 0.0,
            "entropy": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
        }
        update_count = 0
        maximum_kl = 0.0
        kl_early_stop = False
        kl_rollback = False
        learning_rate_before_update = self.learning_rate
        # The network is small, so retaining one in-memory safe point is cheap.
        # If a PPO update crosses the hard KL limit, restore both policy and
        # Adam moments rather than allowing one destructive update to erase a
        # learned strategy.
        policy_before_update = copy.deepcopy(self.policy.state_dict())
        optimizer_before_update = copy.deepcopy(self.optimizer.state_dict())
        self.policy.train()

        for _ in range(config.update_epochs):
            permutation = torch.randperm(
                state_count, device=self.environment.device
            )
            for start in range(0, state_count, config.minibatch_states):
                indices = permutation[start : start + config.minibatch_states]
                output = self.policy(
                    minnow_features[indices], global_features[indices]
                )
                new_log_probabilities, entropy = action_log_probabilities(
                    output, actions[indices]
                )
                log_ratio = (
                    new_log_probabilities - old_log_probabilities[indices]
                )
                ratio = log_ratio.exp()
                advantages = mixed_advantages[indices]
                masks = active_masks[indices]

                with torch.no_grad():
                    pre_step_kl = masked_mean(
                        (ratio - 1) - log_ratio, masks
                    )
                pre_step_kl_value = float(pre_step_kl.item())
                maximum_kl = max(maximum_kl, pre_step_kl_value)
                if pre_step_kl_value > config.rollback_kl:
                    kl_rollback = True
                    break
                if pre_step_kl_value > config.target_kl:
                    kl_early_stop = True
                    break

                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * ratio.clamp(
                    1 - config.clip_ratio, 1 + config.clip_ratio
                )
                actor_loss = masked_mean(
                    torch.maximum(unclipped_loss, clipped_loss), masks
                )
                team_value_loss = nn.functional.mse_loss(
                    output.team_value, team_returns[indices]
                )
                # Frozen/dead minnows produce no actor or entropy loss, but
                # their critic values must learn the delayed terminal outcome.
                individual_value_loss = nn.functional.mse_loss(
                    output.individual_values,
                    individual_returns[indices],
                )
                entropy_loss = masked_mean(entropy, masks)

                loss = (
                    actor_loss
                    + config.team_value_coefficient * team_value_loss
                    + config.individual_value_coefficient
                    * individual_value_loss
                    - config.entropy_coefficient * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), config.max_gradient_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    # Measure the policy after the optimizer step. The previous
                    # implementation only reported the pre-step KL, allowing a
                    # dangerous last minibatch to go unnoticed until later.
                    updated_output = self.policy(
                        minnow_features[indices], global_features[indices]
                    )
                    updated_log_probabilities, _ = action_log_probabilities(
                        updated_output, actions[indices]
                    )
                    updated_log_ratio = (
                        updated_log_probabilities
                        - old_log_probabilities[indices]
                    )
                    updated_ratio = updated_log_ratio.exp()
                    approximate_kl = masked_mean(
                        (updated_ratio - 1) - updated_log_ratio, masks
                    )
                    clip_fraction = masked_mean(
                        (
                            (updated_ratio - 1).abs() > config.clip_ratio
                        ).float(),
                        masks,
                    )
                approximate_kl_value = float(approximate_kl.item())
                maximum_kl = max(maximum_kl, approximate_kl_value)

                current_metrics = {
                    "loss": loss,
                    "actor_loss": actor_loss,
                    "team_value_loss": team_value_loss,
                    "individual_value_loss": individual_value_loss,
                    "entropy": entropy_loss,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "gradient_norm": gradient_norm,
                }
                for name, value in current_metrics.items():
                    metric_totals[name] += float(value.detach().item())
                update_count += 1

                if approximate_kl_value > config.rollback_kl:
                    kl_rollback = True
                    break
                if approximate_kl_value > config.target_kl:
                    kl_early_stop = True
                    break

            if kl_early_stop or kl_rollback:
                break

        if kl_rollback:
            self.policy.load_state_dict(policy_before_update)
            self.optimizer.load_state_dict(optimizer_before_update)
            _, learning_rate_after_update = (
                self.reduce_learning_rate_after_rollback()
            )
        else:
            learning_rate_after_update = self.learning_rate

        metrics = {
            name: total / max(update_count, 1)
            for name, total in metric_totals.items()
        }
        metrics.update(
            {
                "maximum_kl": maximum_kl,
                "kl_early_stop": float(kl_early_stop),
                "kl_rollback": float(kl_rollback),
                "optimizer_minibatches": float(update_count),
                "learning_rate_before_update": learning_rate_before_update,
                "learning_rate": learning_rate_after_update,
                "learning_rate_reduced": float(
                    learning_rate_after_update < learning_rate_before_update
                ),
            }
        )
        return metrics
