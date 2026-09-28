"""Watch a learned or baseline controller play one rendered episode."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random

import pygame
import torch
from torch import Tensor

from environment import EnvironmentConfig, VectorizedSharkEnv
from game_board import GameState, Minnow, WINDOW_SIZE, draw_board
from policy import CentralizedAttentionPolicy


@dataclass(frozen=True)
class Snapshot:
    minnow_positions: Tensor
    minnow_speeds: Tensor
    minnow_alive: Tensor
    minnow_safe: Tensor
    shark_position: Tensor
    elapsed_seconds: float


@dataclass
class FireworkParticle:
    position: pygame.Vector2
    velocity: pygame.Vector2
    color: tuple[int, int, int]
    life: float
    maximum_life: float
    radius: int


class Fireworks:
    """Small replay-only particle celebration for a perfect game."""

    COLORS = (
        (255, 91, 111),
        (255, 205, 72),
        (92, 224, 255),
        (151, 255, 138),
        (207, 142, 255),
        (255, 255, 255),
    )

    def __init__(self, seed: int) -> None:
        self.random = random.Random(seed)
        self.particles: list[FireworkParticle] = []
        self.burst_timer = 0.0
        self.started = False

    def update(
        self,
        seconds: float,
        window_size: tuple[int, int],
        active: bool,
    ) -> None:
        if active:
            if not self.started:
                self.started = True
                for _ in range(5):
                    self._burst(window_size)
                self.burst_timer = 0.35
            else:
                self.burst_timer -= seconds
                if self.burst_timer <= 0:
                    self._burst(window_size)
                    self.burst_timer = self.random.uniform(0.25, 0.55)

        gravity = pygame.Vector2(0, 190)
        living: list[FireworkParticle] = []
        for particle in self.particles:
            particle.life -= seconds
            if particle.life <= 0:
                continue
            particle.velocity += gravity * seconds
            particle.position += particle.velocity * seconds
            living.append(particle)
        self.particles = living

    def _burst(self, window_size: tuple[int, int]) -> None:
        width, height = window_size
        center = pygame.Vector2(
            self.random.uniform(width * 0.18, width * 0.82),
            self.random.uniform(height * 0.10, height * 0.62),
        )
        color = self.random.choice(self.COLORS)
        for particle_index in range(48):
            angle = math.tau * particle_index / 48 + self.random.uniform(-0.05, 0.05)
            speed = self.random.uniform(95, 330)
            life = self.random.uniform(0.9, 1.7)
            self.particles.append(
                FireworkParticle(
                    position=center.copy(),
                    velocity=pygame.Vector2(
                        math.cos(angle) * speed,
                        math.sin(angle) * speed,
                    ),
                    color=color,
                    life=life,
                    maximum_life=life,
                    radius=self.random.randint(2, 5),
                )
            )

    def draw(self, surface: pygame.Surface) -> None:
        for particle in self.particles:
            brightness = max(0.15, particle.life / particle.maximum_life)
            color = tuple(round(channel * brightness) for channel in particle.color)
            pygame.draw.circle(
                surface,
                color,
                (round(particle.position.x), round(particle.position.y)),
                particle.radius,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        choices=("learned", "always-right", "random"),
        default="learned",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--playback-rate", type=float, default=0.5)
    return parser.parse_args()


def capture(environment: VectorizedSharkEnv) -> Snapshot:
    return Snapshot(
        minnow_positions=environment.minnow_positions[0].detach().cpu().clone(),
        minnow_speeds=environment.minnow_speeds[0].detach().cpu().clone(),
        minnow_alive=environment.minnow_alive[0].detach().cpu().clone(),
        minnow_safe=environment.minnow_safe[0].detach().cpu().clone(),
        shark_position=environment.shark_positions[0].detach().cpu().clone(),
        elapsed_seconds=float(environment.elapsed_seconds[0].item()),
    )


def interpolated_game_state(
    previous: Snapshot, current: Snapshot, interpolation: float
) -> GameState:
    interpolation = max(0.0, min(1.0, interpolation))
    minnow_positions = torch.lerp(
        previous.minnow_positions,
        current.minnow_positions,
        interpolation,
    )
    shark_position = torch.lerp(
        previous.shark_position,
        current.shark_position,
        interpolation,
    )
    use_current_status = interpolation >= 0.85
    alive = current.minnow_alive if use_current_status else previous.minnow_alive
    safe = current.minnow_safe if use_current_status else previous.minnow_safe

    minnows = [
        Minnow(
            position=pygame.Vector2(float(position[0]), float(position[1])),
            speed=float(current.minnow_speeds[index]),
            alive=bool(alive[index]),
            safe=bool(safe[index]),
        )
        for index, position in enumerate(minnow_positions)
    ]
    return GameState(
        minnows=minnows,
        shark_position=pygame.Vector2(
            float(shark_position[0]), float(shark_position[1])
        ),
    )


def load_policy(checkpoint_path: Path | None) -> CentralizedAttentionPolicy:
    if checkpoint_path is None:
        raise ValueError("--checkpoint is required for the learned controller")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["policy_state_dict"]
    policy = CentralizedAttentionPolicy(
        minnow_feature_count=state_dict["minnow_encoder.0.weight"].shape[1],
        global_feature_count=state_dict["global_encoder.0.weight"].shape[1],
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def main() -> None:
    args = parse_args()
    if args.playback_rate <= 0:
        raise ValueError("--playback-rate must be positive")

    policy = load_policy(args.checkpoint) if args.controller == "learned" else None
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(args.seed + 1)

    def make_environment() -> VectorizedSharkEnv:
        return VectorizedSharkEnv(1, "cpu", seed=args.seed)

    @torch.no_grad()
    def choose_actions(environment: VectorizedSharkEnv) -> Tensor:
        minnow_features, global_features = environment.observe()
        if args.controller == "learned":
            assert policy is not None
            compatible_minnows = minnow_features[
                ..., : policy.minnow_feature_count
            ]
            compatible_global = global_features[
                ..., : policy.global_feature_count
            ]
            return policy(
                compatible_minnows, compatible_global
            ).preferred_actions
        if args.controller == "always-right":
            return torch.tensor((1.0, 0.0)).expand(
                1, environment.config.minnow_count, 2
            )
        return torch.randn(
            (1, environment.config.minnow_count, 2),
            generator=random_generator,
        )

    pygame.init()
    pygame.display.set_caption(f"Sharks & Minnows Replay — {args.controller}")
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    clock = pygame.time.Clock()
    overlay_font = pygame.font.Font(None, 25)
    celebration_font = pygame.font.Font(None, 88)

    environment = make_environment()
    previous = capture(environment)
    current = previous
    accumulator = 0.0
    fireworks = Fireworks(args.seed + 2)
    paused = False
    running = True

    while running:
        frame_seconds = min(clock.tick(60) / 1000, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    environment = make_environment()
                    previous = capture(environment)
                    current = previous
                    accumulator = 0.0
                    fireworks = Fireworks(args.seed + 2)
                    paused = False

        if not paused and not bool(environment.done[0]):
            accumulator += frame_seconds * args.playback_rate
            while accumulator >= environment.config.decision_seconds:
                previous = current
                environment.step(choose_actions(environment))
                current = capture(environment)
                accumulator -= environment.config.decision_seconds
                if bool(environment.done[0]):
                    break

        interpolation = (
            1.0
            if bool(environment.done[0])
            else accumulator / environment.config.decision_seconds
        )
        state = interpolated_game_state(previous, current, interpolation)
        elapsed = previous.elapsed_seconds + (
            current.elapsed_seconds - previous.elapsed_seconds
        ) * interpolation
        draw_board(
            screen,
            state,
            elapsed_seconds=elapsed,
            episode_seconds=environment.config.episode_seconds,
        )

        saved = int(environment.minnow_safe[0].sum().item())
        perfect = bool(environment.done[0]) and saved == environment.config.minnow_count
        fireworks.update(frame_seconds, screen.get_size(), perfect)
        fireworks.draw(screen)

        if perfect:
            celebration = celebration_font.render(
                "PERFECT 10/10!",
                True,
                (255, 226, 92),
            )
            shadow = celebration_font.render(
                "PERFECT 10/10!",
                True,
                (25, 20, 12),
            )
            center_x = screen.get_width() // 2
            center_y = max(65, screen.get_height() // 9)
            screen.blit(
                shadow,
                shadow.get_rect(center=(center_x + 4, center_y + 4)),
            )
            screen.blit(
                celebration,
                celebration.get_rect(center=(center_x, center_y)),
            )

        status = "PAUSED" if paused else args.controller.upper()
        if bool(environment.done[0]):
            status = f"FINISHED — {saved}/10 SAVED"
        controls = overlay_font.render(
            f"{status}    SPACE pause    R replay    ESC quit",
            True,
            (200, 213, 221),
        )
        screen.blit(
            controls,
            controls.get_rect(
                center=(screen.get_width() // 2, screen.get_height() - 22)
            ),
        )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
