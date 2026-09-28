"""Evaluate learned and fixed controllers on identical game rules."""

import argparse
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm

from environment import EnvironmentConfig, VectorizedSharkEnv
from policy import CentralizedAttentionPolicy
from visualization import (
    plot_evaluation_metrics,
    plot_training_history,
    write_evaluation_results,
)


Controller = Callable[[Tensor, Tensor], Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate_controller(
    name: str,
    controller: Controller,
    episode_count: int,
    device: torch.device,
    seed: int,
    show_progress: bool = True,
) -> dict[str, float]:
    config = EnvironmentConfig()
    environment = VectorizedSharkEnv(
        episode_count, device=device, config=config, seed=seed
    )
    minnow_features, global_features = environment.observe()

    maximum_steps = round(config.episode_seconds / config.decision_seconds) + 1
    progress = tqdm(
        total=maximum_steps,
        desc=f"Evaluate {name.replace('_', ' ')}",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.2,
        disable=not show_progress,
    )
    for step in range(maximum_steps):
        actions = controller(minnow_features, global_features)
        result = environment.step(actions)
        minnow_features = result.minnow_features
        global_features = result.global_features
        progress.update(1)
        if step % 5 == 0 or result.done.all():
            progress.set_postfix(
                saved=(
                    f"{environment.minnow_safe.sum(dim=1).float().mean().item():.2f}/10"
                ),
                complete=f"{result.done.float().mean().item():.0%}",
                refresh=False,
            )
        if result.done.all():
            progress.total = progress.n
            progress.refresh()
            break
    else:
        progress.close()
        raise RuntimeError(f"{name} evaluation did not terminate")
    progress.close()

    safe = environment.minnow_safe
    speeds = environment.minnow_speeds
    metrics: dict[str, float] = {
        "mean_saved": float(safe.sum(dim=1).float().mean().item()),
        "perfect_games": float(
            (safe.sum(dim=1) == config.minnow_count).float().mean().item()
        ),
    }
    buckets = (
        ("slow_020", 0.20, 0.200001),
        ("fast_100", 1.00, 1.000001),
    )
    for bucket_name, minimum, maximum in buckets:
        mask = (speeds >= minimum) & (speeds < maximum)
        metrics[bucket_name] = float(
            safe[mask].float().mean().item() if mask.any() else float("nan")
        )
    return metrics


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"{name}: saved={metrics['mean_saved']:.3f}/10 perfect={metrics['perfect_games']:.2%}")
    print(
        "  survival "
        f"slow(0.2)={metrics['slow_020']:.2%} "
        f"fast(1.0)={metrics['fast_100']:.2%}"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(args.seed + 1)

    always_right: Controller = lambda minnows, global_state: torch.tensor(
        (1.0, 0.0), device=device
    ).expand(*minnows.shape[:2], 2)
    random_legal: Controller = lambda minnows, global_state: torch.randn(
        (*minnows.shape[:2], 2),
        generator=random_generator,
        device=device,
    )

    results: dict[str, dict[str, float]] = {}
    results["always_right"] = evaluate_controller(
        "always_right",
        always_right,
        args.episodes,
        device,
        args.seed,
        show_progress=not args.no_progress,
    )
    results["random_legal"] = evaluate_controller(
        "random_legal",
        random_legal,
        args.episodes,
        device,
        args.seed,
        show_progress=not args.no_progress,
    )
    print_metrics("always_right", results["always_right"])
    print_metrics("random_legal", results["random_legal"])

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint["policy_state_dict"]
        policy = CentralizedAttentionPolicy(
            minnow_feature_count=state_dict["minnow_encoder.0.weight"].shape[1],
            global_feature_count=state_dict["global_encoder.0.weight"].shape[1],
        ).to(device)
        policy.load_state_dict(state_dict)
        policy.eval()

        def learned(minnows: Tensor, global_state: Tensor) -> Tensor:
            return policy(
                minnows[..., : policy.minnow_feature_count],
                global_state[..., : policy.global_feature_count],
            ).preferred_actions

        results["learned"] = evaluate_controller(
            "learned",
            learned,
            args.episodes,
            device,
            args.seed,
            show_progress=not args.no_progress,
        )
        print_metrics("learned", results["learned"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_evaluation_results(results, args.output_dir / "evaluation_results.json")
    plot_evaluation_metrics(
        results,
        args.episodes,
        args.output_dir / "evaluation_comparison.png",
    )
    if checkpoint is not None and checkpoint.get("history"):
        plot_training_history(
            checkpoint["history"],
            args.output_dir / "training_progress.png",
        )
    print(f"visuals={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
