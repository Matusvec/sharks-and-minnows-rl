"""Static training and evaluation charts for Sharks and Minnows."""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

_default_matplotlib_config = Path.home() / ".config" / "matplotlib"
if not os.access(_default_matplotlib_config, os.W_OK):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/shark-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#20262E"
GRID = "#D9DEE5"
BLUE = "#2F6B9A"
BLUE_LIGHT = "#AFC9DD"
GOLD = "#C7922D"
GOLD_LIGHT = "#E8D3A6"
ORANGE = "#D66A3A"
GRAY = "#7B8794"


def write_training_history(history: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, allow_nan=False)

    if not history:
        return
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for row in history:
        for fieldname in row:
            if fieldname not in seen_fields:
                seen_fields.add(fieldname)
                fieldnames.append(fieldname)
    with (directory / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _series(history: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [float("nan") if row.get(key) is None else float(row[key]) for row in history],
        dtype=float,
    )


def _rolling_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    smoothed = np.full_like(values, np.nan)
    for index in range(len(values)):
        segment = values[max(0, index - window + 1) : index + 1]
        finite = segment[np.isfinite(segment)]
        if finite.size:
            smoothed[index] = finite.mean()
    return smoothed


def _plot_trend(
    axis: plt.Axes,
    updates: np.ndarray,
    values: np.ndarray,
    color: str,
    label: str,
) -> None:
    finite = np.isfinite(values)
    if not finite.any():
        return
    axis.plot(
        updates[finite],
        values[finite],
        color=color,
        alpha=0.28,
        linewidth=1,
        marker="o" if finite.sum() < 8 else None,
        markersize=3,
    )
    smoothed = _rolling_average(values)
    axis.plot(updates[finite], smoothed[finite], color=color, linewidth=2.2, label=label)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRAY)
    axis.tick_params(colors=INK)
    axis.title.set_color(INK)


def plot_training_history(history: list[dict[str, Any]], output_path: Path) -> None:
    """Plot outcome progress and optimization diagnostics by PPO update."""
    if not history:
        return

    updates = _series(history, "update")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Training progress by PPO update",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.5,
        0.955,
        "Raw update values are faint; solid lines are trailing 10-update averages.",
        ha="center",
        fontsize=10,
        color=GRAY,
    )

    saved = _series(history, "mean_saved")
    _plot_trend(axes[0, 0], updates, saved, BLUE, "Saved")
    axes[0, 0].set_title("Average minnows saved")
    axes[0, 0].set_ylabel("Saved out of 10")
    axes[0, 0].set_ylim(0, 10)

    slow_survival = _series(history, "slow_survival_rate") * 100
    _plot_trend(axes[0, 1], updates, slow_survival, GOLD, "Slower than shark")
    fast_survival = _series(history, "fast_survival_rate") * 100
    _plot_trend(axes[0, 1], updates, fast_survival, BLUE, "Fast minnow")
    perfect_rate = _series(history, "perfect_game_rate") * 100
    _plot_trend(axes[0, 1], updates, perfect_rate, ORANGE, "Perfect games")
    validation_perfect = (
        _series(history, "validation_perfect_game_rate") * 100
    )
    finite_validation = np.isfinite(validation_perfect)
    if finite_validation.any():
        axes[0, 1].plot(
            updates[finite_validation],
            validation_perfect[finite_validation],
            color=INK,
            linewidth=1.2,
            linestyle="--",
            marker="o",
            markersize=3,
            label="Validation perfect",
        )
    axes[0, 1].set_title("Survival and perfect games")
    axes[0, 1].set_ylabel("Survival rate (%)")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend(frameon=False, loc="lower right")

    actor_loss = _series(history, "actor_loss")
    _plot_trend(axes[1, 0], updates, actor_loss, ORANGE, "Actor")
    axes[1, 0].axhline(0, color=INK, linewidth=0.8, linestyle="--")
    axes[1, 0].set_title("PPO actor loss")
    axes[1, 0].set_ylabel("Clipped surrogate loss")
    axes[1, 0].set_xlabel("PPO update")

    team_loss = _series(history, "team_value_loss")
    individual_loss = _series(history, "individual_value_loss")
    _plot_trend(axes[1, 1], updates, team_loss, BLUE, "Team critic")
    _plot_trend(
        axes[1, 1], updates, individual_loss, GOLD, "Individual critic"
    )
    axes[1, 1].set_title("Critic prediction losses")
    axes[1, 1].set_ylabel("Mean squared error")
    axes[1, 1].set_xlabel("PPO update")
    axes[1, 1].legend(frameon=False, loc="upper right")

    for axis in axes.flat:
        _style_axis(axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor="white")
    plt.close(figure)


def plot_evaluation_metrics(
    results: dict[str, dict[str, float]],
    episode_count: int,
    output_path: Path,
) -> None:
    """Compare controller scores and speed-bucket survival on matched games."""
    display_names = {
        "always_right": "Always right",
        "random_legal": "Random legal",
        "learned": "Learned",
    }
    colors = {
        "always_right": GRAY,
        "random_legal": GOLD,
        "learned": BLUE,
    }
    hatches = {
        "always_right": "//",
        "random_legal": "..",
        "learned": "",
    }
    controller_names = list(results)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Controller evaluation",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.5,
        0.93,
        f"Matched random seeds; {episode_count:,} episodes per controller.",
        ha="center",
        fontsize=10,
        color=GRAY,
    )

    saved_values = [results[name]["mean_saved"] for name in controller_names]
    bars = axes[0].bar(
        [display_names[name] for name in controller_names],
        saved_values,
        color=[colors[name] for name in controller_names],
        edgecolor=INK,
        linewidth=0.8,
        hatch=[hatches[name] for name in controller_names],
    )
    axes[0].bar_label(bars, labels=[f"{value:.2f}" for value in saved_values], padding=3)
    axes[0].set_title("Average minnows saved")
    axes[0].set_ylabel("Saved out of 10")
    axes[0].set_ylim(0, 10)

    bucket_keys = (
        "slow_020",
        "fast_100",
    )
    bucket_labels = ("Slow: 0.2", "Fast: 1.0")
    x_positions = np.arange(len(bucket_keys))
    width = 0.8 / len(controller_names)
    for index, name in enumerate(controller_names):
        values = [results[name][key] * 100 for key in bucket_keys]
        axes[1].bar(
            x_positions + (index - (len(controller_names) - 1) / 2) * width,
            values,
            width,
            label=display_names[name],
            color=colors[name],
            edgecolor=INK,
            linewidth=0.7,
            hatch=hatches[name],
        )
    axes[1].set_title("Survival by assigned speed")
    axes[1].set_ylabel("Survival rate (%)")
    axes[1].set_xticks(x_positions, bucket_labels)
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False, loc="upper left")

    for axis in axes:
        _style_axis(axis)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor="white")
    plt.close(figure)


def write_evaluation_results(
    results: dict[str, dict[str, float]], output_path: Path
) -> None:
    serializable_results = {
        controller: {
            metric: value if math.isfinite(value) else None
            for metric, value in metrics.items()
        }
        for controller, metrics in results.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable_results, handle, indent=2, allow_nan=False)
