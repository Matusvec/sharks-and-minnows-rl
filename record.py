"""Render one deterministic episode headlessly to an MP4 (for the README demos)."""

import argparse
import dataclasses
import os
from pathlib import Path
import subprocess

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import torch

from environment import EnvironmentConfig, VectorizedSharkEnv
from game_board import draw_board
from replay import Fireworks, capture, interpolated_game_state, load_policy

FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--slow-speed", type=float, required=True)
    parser.add_argument("--seed", type=int, default=None, help="default: first seed matching --want")
    parser.add_argument("--want", choices=("perfect", "nine", "any"), default="perfect")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", default="1280x910")
    parser.add_argument("--playback-rate", type=float, default=1.0)
    return parser.parse_args()


def make_environment(checkpoint: Path, slow_speed: float, seed: int) -> VectorizedSharkEnv:
    # Use the physics the checkpoint was trained with (e.g. older runs had no start jitter).
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["environment_config"]
    names = {field.name for field in dataclasses.fields(EnvironmentConfig)}
    config = EnvironmentConfig(**{
        "minnow_start_y_jitter": 0.0,
        **{key: value for key, value in saved.items() if key in names},
    })
    environment = VectorizedSharkEnv(1, "cpu", config=config, seed=seed)
    environment.set_slow_minnow_speed(slow_speed)
    environment.reset()
    return environment


@torch.no_grad()
def act(policy, environment):
    minnows, globals_ = environment.observe()
    return policy(
        minnows[..., : policy.minnow_feature_count],
        globals_[..., : policy.global_feature_count],
    ).preferred_actions


@torch.no_grad()
def outcome(policy, checkpoint, slow_speed, seed) -> int:
    environment = make_environment(checkpoint, slow_speed, seed)
    while not bool(environment.done[0]):
        environment.step(act(policy, environment))
    return int(environment.minnow_safe[0].sum().item())


def main() -> None:
    args = parse_args()
    policy = load_policy(args.checkpoint)
    seed = args.seed
    if seed is None:
        target = {"perfect": 10, "nine": 9}.get(args.want)
        seed = next(
            s for s in range(1000)
            if target is None or outcome(policy, args.checkpoint, args.slow_speed, s) == target
        )
    print(f"seed {seed}")

    width, height = map(int, args.size.split("x"))
    pygame.init()
    screen = pygame.Surface((width, height))
    font = pygame.font.Font(None, 30)
    big_font = pygame.font.Font(None, 90)
    environment = make_environment(args.checkpoint, args.slow_speed, seed)
    previous = current = capture(environment)
    fireworks = Fireworks(seed)
    decision = environment.config.decision_seconds
    frame_seconds = args.playback_rate / FPS
    accumulator, tail_frames = 0.0, FPS * 2

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
         str(args.output)],
        stdin=subprocess.PIPE,
    )
    while tail_frames > 0:
        done = bool(environment.done[0])
        if not done:
            accumulator += frame_seconds
            while accumulator >= decision and not bool(environment.done[0]):
                previous = current
                environment.step(act(policy, environment))
                current = capture(environment)
                accumulator -= decision
        done = bool(environment.done[0])
        tail_frames -= done
        interpolation = 1.0 if done else accumulator / decision
        elapsed = previous.elapsed_seconds + (current.elapsed_seconds - previous.elapsed_seconds) * interpolation
        draw_board(screen, interpolated_game_state(previous, current, interpolation),
                   elapsed_seconds=elapsed, episode_seconds=environment.config.episode_seconds)

        saved = int(environment.minnow_safe[0].sum().item())
        perfect = done and saved == environment.config.minnow_count
        fireworks.update(1 / FPS, (width, height), perfect)
        fireworks.draw(screen)
        if perfect:
            text = big_font.render("PERFECT 10/10!", True, (255, 226, 92))
            screen.blit(text, text.get_rect(center=(width // 2, 60)))
        label = f"slow minnow speed {args.slow_speed:g}  |  shark 0.5  |  fast minnow 1.0"
        if done:
            label = f"FINISHED - {saved}/10 SAVED  |  " + label
        text = font.render(label, True, (200, 213, 221))
        screen.blit(text, text.get_rect(center=(width // 2, height - 20)))
        ffmpeg.stdin.write(pygame.image.tobytes(screen, "RGB"))
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"wrote {args.output} ({saved}/10 saved)")


if __name__ == "__main__":
    main()
