"""Vectorized Sharks and Minnows reinforcement-learning environment."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EnvironmentConfig:
    minnow_count: int = 10
    episode_seconds: float = 15.0
    decision_seconds: float = 0.1
    physics_substeps: int = 4
    field_width: float = 120 / 53.3
    field_height: float = 1.0
    end_zone_fraction: float = 1 / 12
    minnow_min_speed: float = 0.2
    minnow_max_speed: float = 1.0
    fast_minnow_count: int = 1
    shark_speed: float = 0.5
    minnow_collision_radius: float = 0.014
    shark_collision_radius: float = 0.028
    shark_start_fraction: float = 0.62
    # Break the fixed-row shortcut where the policy permanently designates the
    # lowest starting minnow as expendable while keeping the opening formation.
    minnow_start_y_jitter: float = 0.035
    # Crossing the entire playable field through rightward motion earns +0.5
    # for slow minnows. The fast minnow receives no directional shaping so it
    # can move left to lure the shark without being directly penalized.
    # Crossing gives an immediate arrival reward; being safe at episode end
    # supplies a second terminal survival reward.
    progress_reward_scale: float = 0.5
    crossing_reward_scale: float = 1.0
    # Discount-correct potential shaping for the least-advanced slow minnow.
    # Dead slow minnows remain in the minimum. Terminal potential is zero,
    # so changing the safe-fraction multiplier cannot farm extra return.
    trailing_slow_progress_reward_scale: float = 2.0
    trailing_slow_safe_fraction_bonus: float = 3.0
    shaping_discount_factor: float = 0.999
    # Small dense shaping terms for two observed control failures. Wall
    # pressure penalizes only the fraction of fast-minnow movement physically
    # discarded by the top/bottom boundary. Early finish applies once, when
    # the fast minnow crosses, for each slow minnow still alive and unresolved.
    # It is strongest early and decays toward zero at timeout so the fast
    # minnow still prefers a last-second escape over timing out.
    fast_wall_pressure_penalty_scale: float = 0.003
    # Keep this optional legacy heuristic off: safe arrival alone does not
    # establish that the decoy abandoned a teammate.
    fast_early_finish_penalty_per_slow: float = 0.0
    # Retain the extra cost for losing the team's only fast rescuer.
    fast_death_penalty_scale: float = 2.0
    # A sparse team-only bonus makes the difficult last step from nine saved
    # to a perfect game substantially more valuable than another 9/10 result.
    perfect_team_bonus: float = 10.0
    # Pay once when perfection first becomes impossible (death or timeout).
    # Continue the episode so rescuing the remaining minnows still matters.
    first_team_failure_penalty: float = 10.0
    # Used only to report whether useful decoy behavior emerges; it is not a reward.
    fast_decoy_full_separation: float = 0.5

    @property
    def end_zone_width(self) -> float:
        return self.field_width * self.end_zone_fraction

    @property
    def active_left(self) -> float:
        return self.end_zone_width

    @property
    def active_right(self) -> float:
        return self.field_width - self.end_zone_width


@dataclass(frozen=True)
class StepResult:
    minnow_features: Tensor
    global_features: Tensor
    individual_rewards: Tensor
    team_rewards: Tensor
    done: Tensor
    individual_terminal: Tensor
    newly_safe: Tensor
    newly_dead: Tensor
    timed_out: Tensor
    safe_counts: Tensor
    dead_counts: Tensor
    fast_target_fraction: Tensor
    fast_decoy_score: Tensor


class VectorizedSharkEnv:
    """Run many independent games at once on a CPU or CUDA device."""

    def __init__(
        self,
        environment_count: int,
        device: torch.device | str,
        config: EnvironmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        if environment_count < 1:
            raise ValueError("environment_count must be positive")

        self.config = config or EnvironmentConfig()
        if self.config.minnow_count < 1:
            raise ValueError("minnow_count must be positive")
        if self.config.physics_substeps < 1:
            raise ValueError("physics_substeps must be positive")
        if not 0 <= self.config.fast_minnow_count <= self.config.minnow_count:
            raise ValueError("fast_minnow_count must be between zero and minnow_count")

        self.environment_count = environment_count
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.slow_minnow_speed = self.config.minnow_min_speed

        state_shape = (environment_count, self.config.minnow_count)
        self.minnow_positions = torch.zeros(
            (*state_shape, 2), device=self.device, dtype=torch.float32
        )
        self.minnow_speeds = torch.zeros(
            state_shape, device=self.device, dtype=torch.float32
        )
        self.minnow_alive = torch.ones(
            state_shape, device=self.device, dtype=torch.bool
        )
        self.minnow_safe = torch.zeros(
            state_shape, device=self.device, dtype=torch.bool
        )
        self.shark_positions = torch.zeros(
            (environment_count, 2), device=self.device, dtype=torch.float32
        )
        self.elapsed_seconds = torch.zeros(
            environment_count, device=self.device, dtype=torch.float32
        )
        self.done = torch.zeros(
            environment_count, device=self.device, dtype=torch.bool
        )

        self.reset()

    def set_slow_minnow_speed(self, speed: float) -> None:
        """Set the speed assigned to slow minnows on future episode resets."""
        if not self.config.minnow_min_speed <= speed < self.config.minnow_max_speed:
            raise ValueError(
                "slow minnow speed must be between the configured minimum "
                "and fast-minnow speed"
            )
        self.slow_minnow_speed = float(speed)

    @property
    def active_mask(self) -> Tensor:
        return (
            self.minnow_alive
            & ~self.minnow_safe
            & ~self.done.unsqueeze(-1)
        )

    def reset(self, environment_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Reset all environments or only those selected by a boolean mask."""
        if environment_mask is None:
            environment_mask = torch.ones(
                self.environment_count, device=self.device, dtype=torch.bool
            )
        else:
            environment_mask = environment_mask.to(
                device=self.device, dtype=torch.bool
            )
            if environment_mask.shape != (self.environment_count,):
                raise ValueError(
                    "environment_mask must have shape "
                    f"[{self.environment_count}]"
                )

        reset_count = int(environment_mask.sum().item())
        if reset_count == 0:
            return self.observe()

        config = self.config
        base_y_positions = torch.linspace(
            1 / (config.minnow_count + 1),
            config.minnow_count / (config.minnow_count + 1),
            config.minnow_count,
            device=self.device,
        )
        y_jitter = (
            2
            * torch.rand(
                (reset_count, config.minnow_count),
                generator=self.generator,
                device=self.device,
            )
            - 1
        ) * config.minnow_start_y_jitter
        y_positions = (
            base_y_positions.unsqueeze(0) + y_jitter
        ).clamp(
            config.minnow_collision_radius,
            config.field_height - config.minnow_collision_radius,
        )
        start_x = config.end_zone_width * 0.80
        shark_x = config.active_left + (
            config.active_right - config.active_left
        ) * config.shark_start_fraction

        self.minnow_positions[environment_mask, :, 0] = start_x
        self.minnow_positions[environment_mask, :, 1] = y_positions
        self.minnow_speeds[environment_mask] = self.slow_minnow_speed
        selected_environments = environment_mask.nonzero(as_tuple=False).squeeze(-1)
        if config.fast_minnow_count:
            random_rankings = torch.rand(
                (reset_count, config.minnow_count),
                generator=self.generator,
                device=self.device,
            )
            fast_indices = random_rankings.topk(
                config.fast_minnow_count, dim=1
            ).indices
            self.minnow_speeds[
                selected_environments.unsqueeze(1), fast_indices
            ] = config.minnow_max_speed
        random_shark_y = torch.rand(
            reset_count,
            generator=self.generator,
            device=self.device,
        )
        random_shark_y = (
            config.shark_collision_radius
            + random_shark_y
            * (config.field_height - 2 * config.shark_collision_radius)
        )
        self.minnow_alive[environment_mask] = True
        self.minnow_safe[environment_mask] = False
        self.shark_positions[environment_mask, 0] = shark_x
        self.shark_positions[environment_mask, 1] = random_shark_y
        self.elapsed_seconds[environment_mask] = 0.0
        self.done[environment_mask] = False
        return self.observe()

    def observe(self) -> tuple[Tensor, Tensor]:
        """Return normalized raw minnow and global observations."""
        config = self.config
        active = self.active_mask
        dead = ~self.minnow_alive & ~self.minnow_safe

        normalized_speed = (
            2
            * (self.minnow_speeds - config.minnow_min_speed)
            / (config.minnow_max_speed - config.minnow_min_speed)
            - 1
        )
        minnow_features = torch.stack(
            (
                self.minnow_positions[..., 0] / config.field_width,
                self.minnow_positions[..., 1] / config.field_height,
                normalized_speed,
                active.float(),
                dead.float(),
                self.minnow_safe.float(),
                (
                    self.minnow_positions[..., 1]
                    - config.minnow_collision_radius
                ).clamp(0, config.field_height)
                / config.field_height,
                (
                    config.field_height
                    - config.minnow_collision_radius
                    - self.minnow_positions[..., 1]
                ).clamp(0, config.field_height)
                / config.field_height,
                (
                    (config.active_right - self.minnow_positions[..., 0])
                    / (config.active_right - config.active_left)
                ).clamp(0, 1),
            ),
            dim=-1,
        )

        time_remaining = (
            1 - self.elapsed_seconds / config.episode_seconds
        ).clamp(0, 1)
        slow_mask = self.minnow_speeds < config.shark_speed
        slow_count = slow_mask.sum(dim=1).clamp_min(1)
        unresolved_slow_fraction = (
            (active & slow_mask).sum(dim=1) / slow_count
        )
        global_features = torch.stack(
            (
                self.shark_positions[:, 0] / config.field_width,
                self.shark_positions[:, 1] / config.field_height,
                time_remaining,
                unresolved_slow_fraction,
            ),
            dim=-1,
        )
        return minnow_features, global_features

    def step(self, normalized_actions: Tensor) -> StepResult:
        """Advance every unfinished game by one policy-decision interval."""
        agent_shape = (self.environment_count, self.config.minnow_count)
        expected_shape = (*agent_shape, 2)
        if normalized_actions.shape != expected_shape:
            raise ValueError(
                f"normalized_actions must have shape {expected_shape}, "
                f"received {tuple(normalized_actions.shape)}"
            )

        normalized_actions = normalized_actions.to(
            device=self.device, dtype=torch.float32
        ).clamp(-1, 1)
        progress_start_x = self.minnow_positions[..., 0].clone()
        progress_mask = self.active_mask.clone()
        perfect_possible_start = self.minnow_alive.all(dim=1) & ~self.done
        trailing_potential_start = self._trailing_slow_potential()
        slow_mask = self.minnow_speeds < self.config.shark_speed
        individual_rewards = torch.zeros(
            agent_shape, device=self.device, dtype=torch.float32
        )
        newly_safe_total = torch.zeros(
            agent_shape, device=self.device, dtype=torch.bool
        )
        newly_dead_total = torch.zeros(
            agent_shape, device=self.device, dtype=torch.bool
        )
        fast_target_substeps = torch.zeros(
            self.environment_count, device=self.device, dtype=torch.float32
        )
        fast_wall_pressure = torch.zeros(
            agent_shape, device=self.device, dtype=torch.float32
        )
        fast_mask = self.minnow_speeds > self.config.shark_speed

        physics_seconds = (
            self.config.decision_seconds / self.config.physics_substeps
        )
        for _ in range(self.config.physics_substeps):
            newly_safe, wall_pressure = self._move_minnows(
                normalized_actions, physics_seconds
            )
            individual_rewards += (
                self.config.crossing_reward_scale * newly_safe.float()
            )
            newly_safe_total |= newly_safe
            fast_wall_pressure += wall_pressure

            fast_target_substeps += self._move_sharks(physics_seconds).float()
            newly_dead = self._resolve_collisions()
            # Charge each death on the causal transition, not many seconds
            # later after the victim has stopped taking actions.
            individual_rewards -= newly_dead.float()
            individual_rewards -= (
                self.config.fast_death_penalty_scale
                * (newly_dead & fast_mask).float()
            )
            newly_dead_total |= newly_dead

        normalized_progress = (
            self.minnow_positions[..., 0] - progress_start_x
        ) / (self.config.active_right - self.config.active_left)
        # Dense progress is only for the slow minnows that need help finding
        # the finish. Giving it to the fast minnow made every leftward lure
        # directly costly, even when that maneuver benefited the whole team.
        slow_progress_mask = progress_mask & (
            self.minnow_speeds < self.config.shark_speed
        )
        individual_rewards += (
            self.config.progress_reward_scale
            * normalized_progress
            * slow_progress_mask
        )

        mean_fast_wall_pressure = (
            fast_wall_pressure / self.config.physics_substeps
        ) * fast_mask
        individual_rewards -= (
            self.config.fast_wall_pressure_penalty_scale
            * mean_fast_wall_pressure
        )

        unresolved_slow_count = (
            self.minnow_alive & ~self.minnow_safe & slow_mask
        ).sum(dim=1)
        early_fast_finish = newly_safe_total & fast_mask
        time_remaining_fraction = (
            1 - self.elapsed_seconds / self.config.episode_seconds
        ).clamp(0, 1)
        individual_rewards -= (
            self.config.fast_early_finish_penalty_per_slow
            * unresolved_slow_count.unsqueeze(-1)
            * time_remaining_fraction.unsqueeze(-1)
            * early_fast_finish
        )

        unfinished_environments = ~self.done
        self.elapsed_seconds[unfinished_environments] += self.config.decision_seconds

        unresolved = self.minnow_alive & ~self.minnow_safe
        time_limit_reached = (
            self.elapsed_seconds >= self.config.episode_seconds - 1e-6
        ) & unfinished_environments
        timed_out = time_limit_reached & unresolved.any(dim=1)

        all_resolved = ~unresolved.any(dim=1)
        episode_done = unfinished_environments & (all_resolved | time_limit_reached)
        self.done |= episode_done

        # Safe minnows get the terminal survival reward; unresolved minnows
        # get -1 at timeout. Deaths were already charged at collision.
        terminal_outcomes = torch.where(
            self.minnow_safe,
            torch.ones_like(individual_rewards),
            -self.minnow_alive.to(individual_rewards.dtype),
        )
        individual_rewards += terminal_outcomes * episode_done.unsqueeze(-1)

        fast_target_fraction = (
            fast_target_substeps / self.config.physics_substeps
        )
        unresolved_slow = unresolved & slow_mask
        slow_distances = torch.linalg.vector_norm(
            self.minnow_positions - self.shark_positions.unsqueeze(1),
            dim=-1,
        ).masked_fill(~unresolved_slow, torch.inf)
        nearest_slow_distance = slow_distances.amin(dim=1)
        decoy_opportunity = unresolved_slow.any(dim=1)
        separation_score = (
            nearest_slow_distance / self.config.fast_decoy_full_separation
        ).clamp(0, 1)
        separation_score = torch.where(
            decoy_opportunity,
            separation_score,
            torch.zeros_like(separation_score),
        )
        fast_decoy_score = fast_target_fraction * separation_score

        # Shared return includes all individual outcomes, a first-failure
        # penalty and a perfect-game bonus. Decoy metrics are observational.
        team_rewards = individual_rewards.sum(dim=1)
        team_rewards += (
            self.config.shaping_discount_factor * self._trailing_slow_potential()
            - trailing_potential_start
        )
        first_failure = perfect_possible_start & (
            newly_dead_total.any(dim=1) | timed_out
        )
        team_rewards -= self.config.first_team_failure_penalty * first_failure.float()
        perfect_games = episode_done & self.minnow_safe.all(dim=1)
        team_rewards += self.config.perfect_team_bonus * perfect_games.float()
        # Keep individual returns alive through inactive safe/dead states so
        # their delayed terminal outcomes propagate back to their last action.
        # The PPO actor mask still prevents updates from inactive actions.
        individual_terminal = (
            self.done.unsqueeze(-1).expand_as(unresolved).clone()
        )
        minnow_features, global_features = self.observe()

        return StepResult(
            minnow_features=minnow_features,
            global_features=global_features,
            individual_rewards=individual_rewards,
            team_rewards=team_rewards,
            done=self.done.clone(),
            individual_terminal=individual_terminal,
            newly_safe=newly_safe_total,
            newly_dead=newly_dead_total,
            timed_out=timed_out,
            safe_counts=self.minnow_safe.sum(dim=1),
            dead_counts=(~self.minnow_alive & ~self.minnow_safe).sum(dim=1),
            fast_target_fraction=fast_target_fraction,
            fast_decoy_score=fast_decoy_score,
        )

    def _trailing_slow_potential(self) -> Tensor:
        """State potential, including the multiplier, with absorbing value zero."""
        config = self.config
        slow = self.minnow_speeds < config.shark_speed
        progress = (
            (self.minnow_positions[..., 0] - config.active_left)
            / (config.active_right - config.active_left)
        ).clamp(0, 1)
        trailing = progress.masked_fill(~slow, torch.inf).amin(dim=1)
        trailing = torch.where(slow.any(dim=1), trailing, torch.zeros_like(trailing))
        safe_fraction = (self.minnow_safe & slow).sum(dim=1) / slow.sum(dim=1).clamp_min(1)
        potential = config.trailing_slow_progress_reward_scale * trailing * (
            1 + config.trailing_slow_safe_fraction_bonus * safe_fraction.square()
        )
        return torch.where(self.done, torch.zeros_like(potential), potential)

    def _move_minnows(
        self, normalized_actions: Tensor, seconds: float
    ) -> tuple[Tensor, Tensor]:
        config = self.config
        active = self.active_mask
        magnitudes = torch.linalg.vector_norm(
            normalized_actions, dim=-1, keepdim=True
        )
        unit_directions = normalized_actions / magnitudes.clamp_min(1e-8)
        unit_directions = torch.where(
            magnitudes > 1e-8,
            unit_directions,
            torch.zeros_like(unit_directions),
        )
        throttled_controls = normalized_actions / magnitudes.clamp_min(1.0)
        fast_mask = (
            self.minnow_speeds > config.shark_speed
        ).unsqueeze(-1)
        # Slow minnows move at their fixed 0.2 speed. The 1.0 fast minnow uses
        # action magnitude as a throttle, allowing fluid waiting and circling.
        velocity_controls = torch.where(
            fast_mask,
            throttled_controls,
            unit_directions,
        )
        movement = (
            velocity_controls
            * self.minnow_speeds.unsqueeze(-1)
            * seconds
        )
        applied_movement = movement * active.unsqueeze(-1)
        self.minnow_positions += applied_movement
        self.minnow_positions[..., 0].clamp_(
            config.minnow_collision_radius,
            config.active_right,
        )
        unclamped_y = self.minnow_positions[..., 1].clone()
        self.minnow_positions[..., 1].clamp_(
            config.minnow_collision_radius,
            config.field_height - config.minnow_collision_radius,
        )
        blocked_y_distance = (
            self.minnow_positions[..., 1] - unclamped_y
        ).abs()
        maximum_step_distance = (
            self.minnow_speeds * seconds
        ).clamp_min(1e-8)
        wall_pressure = (
            blocked_y_distance / maximum_step_distance
        ).clamp(0, 1) * active

        newly_safe = active & (self.minnow_positions[..., 0] >= config.active_right)
        self.minnow_positions[..., 0] = torch.where(
            newly_safe,
            torch.full_like(self.minnow_positions[..., 0], config.active_right),
            self.minnow_positions[..., 0],
        )
        self.minnow_safe |= newly_safe
        return newly_safe, wall_pressure

    def _move_sharks(self, seconds: float) -> Tensor:
        config = self.config
        active = self.active_mask
        in_play = (
            (self.minnow_positions[..., 0] > config.active_left)
            & (self.minnow_positions[..., 0] < config.active_right)
        )
        eligible = active & in_play

        offsets = self.minnow_positions - self.shark_positions.unsqueeze(1)
        distances_squared = offsets.square().sum(dim=-1)
        masked_distances = distances_squared.masked_fill(~eligible, torch.inf)
        nearest_distances = masked_distances.amin(dim=1, keepdim=True)
        tied = eligible & torch.isclose(
            distances_squared,
            nearest_distances,
            rtol=1e-6,
            atol=1e-8,
        )
        center_distances = (
            self.minnow_positions[..., 1] - config.field_height / 2
        ).abs()
        target_indices = center_distances.masked_fill(~tied, torch.inf).argmin(dim=1)
        has_target = eligible.any(dim=1) & ~self.done
        target_is_fast = has_target & (
            self.minnow_speeds.gather(1, target_indices.unsqueeze(1)).squeeze(1)
            > config.shark_speed
        )

        target_positions = self.minnow_positions.gather(
            1,
            target_indices[:, None, None].expand(-1, 1, 2),
        ).squeeze(1)
        shark_offsets = target_positions - self.shark_positions
        distances = torch.linalg.vector_norm(shark_offsets, dim=-1)
        directions = shark_offsets / distances.clamp_min(1e-12).unsqueeze(-1)
        step_distances = torch.minimum(
            torch.full_like(distances, config.shark_speed * seconds),
            distances,
        )
        self.shark_positions += (
            directions * step_distances.unsqueeze(-1) * has_target.unsqueeze(-1)
        )
        self.shark_positions[:, 0].clamp_(
            config.active_left + config.shark_collision_radius,
            config.active_right - config.shark_collision_radius,
        )
        self.shark_positions[:, 1].clamp_(
            config.shark_collision_radius,
            config.field_height - config.shark_collision_radius,
        )
        return target_is_fast

    def _resolve_collisions(self) -> Tensor:
        config = self.config
        active = self.active_mask
        in_play = (
            (self.minnow_positions[..., 0] > config.active_left)
            & (self.minnow_positions[..., 0] < config.active_right)
        )
        distances_squared = (
            self.minnow_positions - self.shark_positions.unsqueeze(1)
        ).square().sum(dim=-1)
        collision_distance = (
            config.minnow_collision_radius + config.shark_collision_radius
        )
        newly_dead = (
            active & in_play & (distances_squared <= collision_distance**2)
        )
        self.minnow_alive &= ~newly_dead
        return newly_dead
