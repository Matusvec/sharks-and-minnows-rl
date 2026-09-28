"""Train the centralized attention policy with PPO."""

import argparse
import copy
from dataclasses import asdict
import math
from pathlib import Path
import time

import torch
from tqdm.auto import tqdm

from environment import EnvironmentConfig, VectorizedSharkEnv
from policy import CentralizedAttentionPolicy
from ppo import PPOConfig, PPOTrainer
from visualization import plot_training_history, write_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--environments", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--gae-lambda", type=float, default=0.99)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-states", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--rollback-learning-rate-factor", type=float, default=0.5
    )
    parser.add_argument("--max-consecutive-kl-rollbacks", type=int, default=8)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--rollback-kl", type=float, default=0.05)
    parser.add_argument("--curriculum-updates", type=int, default=0)
    parser.add_argument("--curriculum-start-slow-speed", type=float, default=0.4)
    parser.add_argument("--adaptive-curriculum", action="store_true")
    parser.add_argument("--curriculum-speed-step", type=float, default=0.01)
    parser.add_argument("--curriculum-late-speed-step", type=float, default=0.01)
    parser.add_argument("--curriculum-late-speed-threshold", type=float, default=0.30)
    parser.add_argument("--curriculum-saved-threshold", type=float, default=9.8)
    parser.add_argument(
        "--curriculum-fast-survival-threshold", type=float, default=0.90
    )
    parser.add_argument("--curriculum-perfect-threshold", type=float, default=0.90)
    parser.add_argument("--curriculum-window", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=1024)
    parser.add_argument("--validation-seed", type=int, default=50_000)
    parser.add_argument("--validation-rollback-drop", type=float, default=0.75)
    parser.add_argument(
        "--validation-perfect-rollback-drop", type=float, default=0.20
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument(
        "--initialize-actor-from", type=Path,
        help="Copy compatible actor/encoder weights; reset critics, optimizer and history",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    policy: CentralizedAttentionPolicy,
    trainer: PPOTrainer,
    update: int,
    arguments: argparse.Namespace,
    history: list[dict[str, object]],
    training_state: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_arguments = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(arguments).items()
    }
    torch.save(
        {
            "update": update,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "environment_config": asdict(trainer.environment.config),
            "ppo_config": asdict(trainer.config),
            "arguments": serialized_arguments,
            "history": history,
            "training_state": training_state or {},
        },
        path,
    )


@torch.no_grad()
def validate_policy(
    policy: CentralizedAttentionPolicy,
    environment_config: EnvironmentConfig,
    slow_speed: float,
    episode_count: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Evaluate the deterministic controller on fixed held-out games."""
    environment = VectorizedSharkEnv(
        episode_count,
        device=device,
        config=environment_config,
        seed=seed,
    )
    environment.set_slow_minnow_speed(slow_speed)
    minnow_features, global_features = environment.reset()
    policy.eval()
    maximum_steps = round(
        environment_config.episode_seconds
        / environment_config.decision_seconds
    ) + 1
    for _ in range(maximum_steps):
        actions = policy(
            minnow_features, global_features
        ).preferred_actions
        result = environment.step(actions)
        minnow_features = result.minnow_features
        global_features = result.global_features
        if result.done.all():
            break
    else:
        raise RuntimeError("validation episodes did not terminate")

    safe = environment.minnow_safe
    slow = environment.minnow_speeds < environment_config.shark_speed
    fast = environment.minnow_speeds > environment_config.shark_speed
    return {
        "validation_mean_saved": float(
            safe.sum(dim=1).float().mean().item()
        ),
        "validation_slow_survival_rate": float(
            safe[slow].float().mean().item()
        ),
        "validation_fast_survival_rate": float(
            safe[fast].float().mean().item()
        ),
        "validation_perfect_game_rate": float(
            (safe.sum(dim=1) == environment_config.minnow_count)
            .float()
            .mean()
            .item()
        ),
    }


def validation_improves(
    saved: float, perfect: float, best_saved: float, best_perfect: float
) -> bool:
    """Prefer perfect-game probability; mean survivors breaks exact ties."""
    return perfect > best_perfect or (perfect == best_perfect and saved > best_saved)


def initialize_actor(policy: CentralizedAttentionPolicy, checkpoint: dict) -> None:
    """Reuse behavior across reward changes without reusing old value targets."""
    state = policy.state_dict()
    old_state = checkpoint["policy_state_dict"]
    for name in state:
        if not name.startswith(("team_value_head.", "individual_value_head.")):
            state[name] = old_state[name]
    policy.load_state_dict(state)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    environment_config = EnvironmentConfig()
    environment = VectorizedSharkEnv(
        args.environments,
        device=device,
        config=environment_config,
        seed=args.seed,
    )
    if args.curriculum_updates < 0:
        raise ValueError("--curriculum-updates cannot be negative")
    if args.curriculum_speed_step <= 0:
        raise ValueError("--curriculum-speed-step must be positive")
    if args.curriculum_late_speed_step <= 0:
        raise ValueError("--curriculum-late-speed-step must be positive")
    if not (
        environment_config.minnow_min_speed
        <= args.curriculum_late_speed_threshold
        <= args.curriculum_start_slow_speed
    ):
        raise ValueError(
            "--curriculum-late-speed-threshold must be between the minimum "
            "and starting slow-minnow speeds"
        )
    if not 0 <= args.curriculum_saved_threshold <= environment_config.minnow_count:
        raise ValueError(
            "--curriculum-saved-threshold must be between zero and "
            f"{environment_config.minnow_count}"
        )
    if args.curriculum_window < 1:
        raise ValueError("--curriculum-window must be positive")
    if args.validate_every < 1:
        raise ValueError("--validate-every must be positive")
    if args.validation_episodes < 1:
        raise ValueError("--validation-episodes must be positive")
    if not 0 <= args.curriculum_fast_survival_threshold <= 1:
        raise ValueError(
            "--curriculum-fast-survival-threshold must be in [0, 1]"
        )
    if not 0 <= args.curriculum_perfect_threshold <= 1:
        raise ValueError("--curriculum-perfect-threshold must be in [0, 1]")
    if args.validation_rollback_drop <= 0:
        raise ValueError("--validation-rollback-drop must be positive")
    if not 0 < args.validation_perfect_rollback_drop <= 1:
        raise ValueError(
            "--validation-perfect-rollback-drop must be in (0, 1]"
        )
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not 0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError(
            "--minimum-learning-rate must be positive and no greater than "
            "--learning-rate"
        )
    if not 0 < args.rollback_learning_rate_factor < 1:
        raise ValueError(
            "--rollback-learning-rate-factor must be between zero and one"
        )
    if args.max_consecutive_kl_rollbacks < 1:
        raise ValueError("--max-consecutive-kl-rollbacks must be positive")
    if args.target_kl <= 0:
        raise ValueError("--target-kl must be positive")
    if args.rollback_kl <= args.target_kl:
        raise ValueError("--rollback-kl must be greater than --target-kl")
    curriculum_enabled = bool(
        args.curriculum_updates or args.adaptive_curriculum
    )
    if curriculum_enabled:
        environment.set_slow_minnow_speed(args.curriculum_start_slow_speed)
        environment.reset()
    policy = CentralizedAttentionPolicy().to(device)
    ppo_config = PPOConfig(
        rollout_steps=args.rollout_steps,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_states=args.minibatch_states,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        rollback_learning_rate_factor=args.rollback_learning_rate_factor,
        target_kl=args.target_kl,
        rollback_kl=args.rollback_kl,
    )
    trainer = PPOTrainer(policy, environment, ppo_config)
    if args.initialize_actor_from is not None:
        initialize_actor(
            policy, torch.load(args.initialize_actor_from, map_location=device)
        )

    history: list[dict[str, object]] = []
    adaptive_curriculum_speed = args.curriculum_start_slow_speed
    curriculum_recent_results: list[tuple[float, float, float]] = []
    best_scores_by_speed: dict[str, float] = {}
    best_perfect_by_speed: dict[str, float] = {}
    best_overall_speed = float("inf")
    best_overall_score = float("-inf")
    best_overall_perfect = float("-inf")
    validation_safe_points: dict[str, dict[str, object]] = {}
    validation_rollback_count = 0
    consecutive_kl_rollbacks = 0
    starting_update = 1

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        saved_environment_config = checkpoint.get("environment_config")
        if saved_environment_config != asdict(environment_config):
            raise ValueError(
                "resume checkpoint uses different game or reward settings; "
                "use a fresh run or --initialize-actor-from for changed rewards"
            )
        if checkpoint.get("ppo_config") != asdict(ppo_config):
            raise ValueError(
                "resume checkpoint uses different or unrecorded PPO settings; "
                "use --initialize-actor-from to start a new experiment"
            )
        policy.load_state_dict(checkpoint["policy_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.set_learning_rate(
            max(
                args.minimum_learning_rate,
                min(trainer.learning_rate, args.learning_rate),
            )
        )
        history = list(checkpoint.get("history", []))
        starting_update = int(checkpoint["update"]) + 1
        if starting_update > args.updates:
            raise ValueError(
                f"resume checkpoint is already at update {starting_update - 1}; "
                f"--updates must be at least {starting_update}"
            )
        saved_training_state = checkpoint.get("training_state", {})
        if args.adaptive_curriculum:
            adaptive_curriculum_speed = float(
                saved_training_state.get(
                    "adaptive_curriculum_speed",
                    history[-1]["curriculum_slow_speed"]
                    if history
                    else args.curriculum_start_slow_speed,
                )
            )
            curriculum_recent_results = []
            for result in saved_training_state.get(
                "curriculum_recent_results", []
            ):
                saved, fast_survival, *perfect_rate = result
                curriculum_recent_results.append(
                    (
                        float(saved),
                        float(fast_survival),
                        float(perfect_rate[0]) if perfect_rate else 0.0,
                    )
                )
            curriculum_recent_results = curriculum_recent_results[
                -args.curriculum_window :
            ]
            if not curriculum_recent_results and history:
                for row in history[-args.curriculum_window :]:
                    if (
                        math.isclose(
                            float(row["curriculum_slow_speed"]),
                            adaptive_curriculum_speed,
                        )
                        and row.get("mean_saved") is not None
                        and row.get("fast_survival_rate") is not None
                    ):
                        curriculum_recent_results.append(
                            (
                                float(row["mean_saved"]),
                                float(row["fast_survival_rate"]),
                                float(row.get("perfect_game_rate", 0.0)),
                            )
                        )
            environment.set_slow_minnow_speed(adaptive_curriculum_speed)
            trainer.minnow_features, trainer.global_features = environment.reset()
        best_scores_by_speed = {
            str(speed): float(score)
            for speed, score in saved_training_state.get(
                "best_scores_by_speed", {}
            ).items()
        }
        best_perfect_by_speed = {
            str(speed): float(score)
            for speed, score in saved_training_state.get(
                "best_perfect_by_speed", {}
            ).items()
        }
        best_overall_speed = float(
            saved_training_state.get("best_overall_speed", float("inf"))
        )
        best_overall_score = float(
            saved_training_state.get("best_overall_score", float("-inf"))
        )
        best_overall_perfect = float(
            saved_training_state.get("best_overall_perfect", float("-inf"))
        )
        validation_rollback_count = int(
            saved_training_state.get("validation_rollback_count", 0)
        )
        consecutive_kl_rollbacks = int(
            saved_training_state.get("consecutive_kl_rollbacks", 0)
        )

    def current_training_state() -> dict[str, object]:
        return {
            "adaptive_curriculum_speed": adaptive_curriculum_speed,
            "curriculum_recent_results": curriculum_recent_results,
            "best_scores_by_speed": dict(best_scores_by_speed),
            "best_perfect_by_speed": dict(best_perfect_by_speed),
            "best_overall_speed": best_overall_speed,
            "best_overall_score": best_overall_score,
            "best_overall_perfect": best_overall_perfect,
            "validation_rollback_count": validation_rollback_count,
            "consecutive_kl_rollbacks": consecutive_kl_rollbacks,
            "learning_rate": trainer.learning_rate,
        }

    print(
        f"device={device} environments={args.environments} "
        f"rollout_steps={args.rollout_steps} updates={args.updates} "
        f"update_epochs={args.update_epochs} "
        f"minibatch_states={args.minibatch_states} "
        f"learning_rate={trainer.learning_rate:g} "
        f"minimum_learning_rate={args.minimum_learning_rate:g} "
        f"rollback_lr_factor={args.rollback_learning_rate_factor:g} "
        f"target_kl={args.target_kl:g} rollback_kl={args.rollback_kl:g} "
        f"validation_rollback_drop={args.validation_rollback_drop:g} "
        f"perfect_rollback_drop={args.validation_perfect_rollback_drop:g} "
        f"minnow_speeds={environment_config.minnow_min_speed:.1f}/"
        f"{environment_config.minnow_max_speed:.1f} "
        f"fast_minnows={environment_config.fast_minnow_count}"
    )
    training_started = time.perf_counter()
    progress = tqdm(
        range(starting_update, args.updates + 1),
        desc="PPO training",
        unit="update",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for update in progress:
        curriculum_speed = environment_config.minnow_min_speed
        if args.adaptive_curriculum:
            curriculum_speed = adaptive_curriculum_speed
            environment.set_slow_minnow_speed(curriculum_speed)
        elif args.curriculum_updates:
            curriculum_fraction = min(
                (update - 1) / max(args.curriculum_updates - 1, 1),
                1.0,
            )
            curriculum_speed = (
                args.curriculum_start_slow_speed
                + curriculum_fraction
                * (
                    environment_config.minnow_min_speed
                    - args.curriculum_start_slow_speed
                )
            )
            environment.set_slow_minnow_speed(curriculum_speed)
        update_started = time.perf_counter()
        batch, rollout = trainer.collect_rollout()
        losses = trainer.update(batch)
        if losses["kl_rollback"]:
            consecutive_kl_rollbacks += 1
        else:
            consecutive_kl_rollbacks = 0
        optimizer_stalled = (
            consecutive_kl_rollbacks
            >= args.max_consecutive_kl_rollbacks
            and trainer.learning_rate
            <= args.minimum_learning_rate * (1 + 1e-9)
        )
        validation: dict[str, float | None] = {
            "validation_mean_saved": None,
            "validation_slow_survival_rate": None,
            "validation_fast_survival_rate": None,
            "validation_perfect_game_rate": None,
        }
        validation_rollback = False
        validation_best_saved: float | None = None
        validation_best_perfect: float | None = None
        if update % args.validate_every == 0:
            validation.update(
                validate_policy(
                    policy,
                    environment_config,
                    curriculum_speed,
                    args.validation_episodes,
                    device,
                    args.validation_seed,
                )
            )
            validation_speed_key = f"{curriculum_speed:.3f}"
            validation_saved = float(validation["validation_mean_saved"])
            validation_perfect = float(
                validation["validation_perfect_game_rate"]
            )
            safe_point = validation_safe_points.get(validation_speed_key)
            if safe_point is not None:
                validation_best_saved = float(safe_point["score"])
                validation_best_perfect = float(safe_point["perfect_rate"])
            saved_score_collapsed = (
                safe_point is not None
                and validation_perfect <= validation_best_perfect
                and validation_saved
                < validation_best_saved - args.validation_rollback_drop
            )
            perfect_rate_collapsed = (
                safe_point is not None
                and validation_perfect
                < validation_best_perfect
                - args.validation_perfect_rollback_drop
            )
            if (
                safe_point is not None
                and (saved_score_collapsed or perfect_rate_collapsed)
            ):
                learning_rate_before_validation_rollback = (
                    trainer.learning_rate
                )
                policy.load_state_dict(safe_point["policy_state_dict"])
                trainer.optimizer.load_state_dict(
                    safe_point["optimizer_state_dict"]
                )
                trainer.set_learning_rate(
                    max(
                        args.minimum_learning_rate,
                        min(
                            learning_rate_before_validation_rollback,
                            trainer.learning_rate,
                        ),
                    )
                )
                trainer.minnow_features, trainer.global_features = (
                    environment.reset()
                )
                validation_rollback = True
                validation_rollback_count += 1
            elif (
                safe_point is None
                or validation_improves(
                    validation_saved, validation_perfect,
                    validation_best_saved, validation_best_perfect,
                )
            ):
                validation_best_saved = validation_saved
                validation_best_perfect = validation_perfect
                validation_safe_points[validation_speed_key] = {
                    "score": validation_saved,
                    "perfect_rate": validation_perfect,
                    "policy_state_dict": copy.deepcopy(policy.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(
                        trainer.optimizer.state_dict()
                    ),
                }
        update_seconds = time.perf_counter() - update_started
        mean_safe = None if not math.isfinite(rollout.mean_safe) else rollout.mean_safe
        slow_survival = (
            None
            if not math.isfinite(rollout.slow_survival_rate)
            else rollout.slow_survival_rate
        )
        fast_survival = (
            None
            if not math.isfinite(rollout.fast_survival_rate)
            else rollout.fast_survival_rate
        )
        perfect_rate = (
            None
            if not math.isfinite(rollout.perfect_game_rate)
            else rollout.perfect_game_rate
        )
        if (
            args.adaptive_curriculum
            and mean_safe is not None
            and fast_survival is not None
            and perfect_rate is not None
        ):
            curriculum_recent_results.append(
                (mean_safe, fast_survival, perfect_rate)
            )
            curriculum_recent_results = curriculum_recent_results[
                -args.curriculum_window :
            ]
        curriculum_window_saved = (
            sum(saved for saved, _, _ in curriculum_recent_results)
            / len(curriculum_recent_results)
            if curriculum_recent_results
            else None
        )
        curriculum_window_fast_survival = (
            sum(fast for _, fast, _ in curriculum_recent_results)
            / len(curriculum_recent_results)
            if curriculum_recent_results
            else None
        )
        curriculum_window_perfect_rate = (
            sum(perfect for _, _, perfect in curriculum_recent_results)
            / len(curriculum_recent_results)
            if curriculum_recent_results
            else None
        )
        history.append(
            {
                "update": update,
                "completed_episodes": rollout.completed_episodes,
                "mean_saved": mean_safe,
                "slow_survival_rate": slow_survival,
                "fast_survival_rate": fast_survival,
                "perfect_game_rate": perfect_rate,
                **validation,
                "fast_target_rate": rollout.fast_target_rate,
                "fast_decoy_quality": rollout.fast_decoy_quality,
                "curriculum_slow_speed": curriculum_speed,
                "curriculum_window_size": len(curriculum_recent_results),
                "curriculum_window_saved": curriculum_window_saved,
                "curriculum_window_fast_survival": (
                    curriculum_window_fast_survival
                ),
                "curriculum_window_perfect_rate": (
                    curriculum_window_perfect_rate
                ),
                "validation_rollback": float(validation_rollback),
                "validation_best_saved": validation_best_saved,
                "validation_best_perfect": validation_best_perfect,
                "timed_out_episodes": rollout.timed_out_episodes,
                "actor_loss": losses["actor_loss"],
                "team_value_loss": losses["team_value_loss"],
                "individual_value_loss": losses["individual_value_loss"],
                "entropy": losses["entropy"],
                "approximate_kl": losses["approximate_kl"],
                "maximum_kl": losses["maximum_kl"],
                "kl_early_stop": losses["kl_early_stop"],
                "kl_rollback": losses["kl_rollback"],
                "learning_rate": trainer.learning_rate,
                "learning_rate_reduced": losses["learning_rate_reduced"],
                "consecutive_kl_rollbacks": consecutive_kl_rollbacks,
                "optimizer_stalled": float(optimizer_stalled),
                "optimizer_minibatches": losses["optimizer_minibatches"],
                "clip_fraction": losses["clip_fraction"],
                "gradient_norm": losses["gradient_norm"],
                "update_seconds": update_seconds,
            }
        )

        if args.adaptive_curriculum:
            if (
                len(curriculum_recent_results) >= args.curriculum_window
                and curriculum_window_saved is not None
                and curriculum_window_saved
                >= args.curriculum_saved_threshold
                and curriculum_window_fast_survival is not None
                and curriculum_window_fast_survival
                >= args.curriculum_fast_survival_threshold
                and curriculum_window_perfect_rate is not None
                and curriculum_window_perfect_rate >= args.curriculum_perfect_threshold
                and validation["validation_perfect_game_rate"] is not None
                and validation["validation_perfect_game_rate"]
                >= args.curriculum_perfect_threshold
                and not validation_rollback
                and adaptive_curriculum_speed
                > environment_config.minnow_min_speed
            ):
                curriculum_step = (
                    args.curriculum_late_speed_step
                    if adaptive_curriculum_speed
                    <= args.curriculum_late_speed_threshold + 1e-9
                    else args.curriculum_speed_step
                )
                adaptive_curriculum_speed = max(
                    environment_config.minnow_min_speed,
                    round(adaptive_curriculum_speed - curriculum_step, 10),
                )
                environment.set_slow_minnow_speed(adaptive_curriculum_speed)
                trainer.minnow_features, trainer.global_features = environment.reset()
                curriculum_recent_results = []
                validation_safe_points.clear()

        status_line = (
            f"update={update:04d} "
            f"episodes={rollout.completed_episodes:5d} "
            f"saved={rollout.mean_safe:5.2f}/10 "
            f"slow_survival={rollout.slow_survival_rate:6.2%} "
            f"fast_survival={rollout.fast_survival_rate:6.2%} "
            f"perfect={rollout.perfect_game_rate:6.2%} "
            f"fast_target={rollout.fast_target_rate:6.2%} "
            f"decoy_quality={rollout.fast_decoy_quality:6.2%} "
            f"slow_speed={curriculum_speed:.3f} "
            f"curriculum_window={len(curriculum_recent_results)}/"
            f"{args.curriculum_window} "
            f"window_saved={curriculum_window_saved} "
            f"window_fast_survival={curriculum_window_fast_survival} "
            f"window_perfect={curriculum_window_perfect_rate} "
            f"validation_perfect={validation['validation_perfect_game_rate']} "
            f"actor={losses['actor_loss']:+.4f} "
            f"team_value={losses['team_value_loss']:.4f} "
            f"individual_value={losses['individual_value_loss']:.4f} "
            f"entropy={losses['entropy']:.3f} "
            f"kl={losses['approximate_kl']:.5f} "
            f"max_kl={losses['maximum_kl']:.5f} "
            f"kl_rollback={bool(losses['kl_rollback'])} "
            f"learning_rate={trainer.learning_rate:.2e} "
            f"rollback_streak={consecutive_kl_rollbacks}/"
            f"{args.max_consecutive_kl_rollbacks} "
            f"validation_rollback={validation_rollback} "
            f"seconds={update_seconds:.1f}"
        )
        if args.no_progress:
            print(status_line)
        else:
            saved_text = "--" if mean_safe is None else f"{mean_safe:.2f}/10"
            slow_text = "--" if slow_survival is None else f"{slow_survival:.1%}"
            fast_text = "--" if fast_survival is None else f"{fast_survival:.1%}"
            validation_perfect_text = (
                "--"
                if validation["validation_perfect_game_rate"] is None
                else f"{validation['validation_perfect_game_rate']:.0%}"
            )
            progress.set_postfix_str(
                f"saved={saved_text} "
                f"speed={curriculum_speed:.3f} "
                f"slow={slow_text} "
                f"fast={fast_text} "
                f"perfect={rollout.perfect_game_rate:.1%} "
                f"v10={validation_perfect_text} "
                f"gate={len(curriculum_recent_results)}/"
                f"{args.curriculum_window} "
                f"avg={curriculum_window_saved or 0:.2f} "
                f"kl={losses['maximum_kl']:.3f} "
                f"lr={trainer.learning_rate:.1e} "
                f"rb={consecutive_kl_rollbacks}/"
                f"{args.max_consecutive_kl_rollbacks} "
                f"safe={'VRB' if validation_rollback else ('RB' if losses['kl_rollback'] else ('STOP' if losses['kl_early_stop'] else 'OK'))} "
                f"loss={losses['loss']:+.3f} "
                f"ep={rollout.completed_episodes}",
                refresh=False,
            )

        validation_saved = validation["validation_mean_saved"]
        if validation_saved is not None and not validation_rollback:
            speed_key = f"{curriculum_speed:.3f}"
            previous_best = best_scores_by_speed.get(speed_key, float("-inf"))
            previous_best_perfect = best_perfect_by_speed.get(
                speed_key, float("-inf")
            )
            validation_perfect = float(
                validation["validation_perfect_game_rate"]
            )
            validation_improved = validation_improves(
                validation_saved, validation_perfect,
                previous_best, previous_best_perfect,
            )
            if validation_improved:
                best_scores_by_speed[speed_key] = validation_saved
                best_perfect_by_speed[speed_key] = validation_perfect
                if (
                    curriculum_speed < best_overall_speed - 1e-9
                    or (
                        math.isclose(curriculum_speed, best_overall_speed)
                        and validation_improves(
                            validation_saved, validation_perfect,
                            best_overall_score, best_overall_perfect,
                        )
                    )
                ):
                    best_overall_speed = curriculum_speed
                    best_overall_score = validation_saved
                    best_overall_perfect = validation_perfect
                    save_checkpoint(
                        args.checkpoint_dir / "best.pt",
                        policy,
                        trainer,
                        update,
                        args,
                        history,
                        current_training_state(),
                    )
                save_checkpoint(
                    args.checkpoint_dir / f"best_speed_{speed_key}.pt",
                    policy,
                    trainer,
                    update,
                    args,
                    history,
                    current_training_state(),
                )

        training_state = current_training_state()

        if update % args.plot_every == 0:
            write_training_history(history, args.checkpoint_dir)
            plot_training_history(
                history, args.checkpoint_dir / "training_progress.png"
            )

        if update % args.save_every == 0:
            save_checkpoint(
                args.checkpoint_dir / f"policy_update_{update:04d}.pt",
                policy,
                trainer,
                update,
                args,
                history,
                training_state,
            )
            save_checkpoint(
                args.checkpoint_dir / "latest.pt",
                policy,
                trainer,
                update,
                args,
                history,
                training_state,
            )

        if optimizer_stalled:
            save_checkpoint(
                args.checkpoint_dir / "stalled.pt",
                policy,
                trainer,
                update,
                args,
                history,
                current_training_state(),
            )
            write_training_history(history, args.checkpoint_dir)
            plot_training_history(
                history, args.checkpoint_dir / "training_progress.png"
            )
            progress.close()
            raise RuntimeError(
                "PPO stopped after "
                f"{consecutive_kl_rollbacks} consecutive KL rollbacks at "
                f"the minimum learning rate {trainer.learning_rate:g}; "
                f"recoverable checkpoint: {args.checkpoint_dir / 'stalled.pt'}"
            )

    save_checkpoint(
        args.checkpoint_dir / "final.pt",
        policy,
        trainer,
        args.updates,
        args,
        history,
        current_training_state(),
    )
    write_training_history(history, args.checkpoint_dir)
    plot_training_history(history, args.checkpoint_dir / "training_progress.png")
    print(f"training_seconds={time.perf_counter() - training_started:.1f}")


if __name__ == "__main__":
    main()
