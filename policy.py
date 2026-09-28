"""Centralized actor-critic policy for the ten-minnow team."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


MINNOW_FEATURE_COUNT = 9
GLOBAL_FEATURE_COUNT = 4
MINIMUM_ACTION_STD = 0.18


@dataclass(frozen=True)
class PolicyOutput:
    """Unbounded policy parameters and the critic's state-value estimate."""

    direction_logits: Tensor
    direction_log_std: Tensor
    team_value: Tensor
    individual_values: Tensor

    @property
    def preferred_actions(self) -> Tensor:
        """Return one bounded 2D velocity control per minnow."""
        return bound_direction_actions(torch.tanh(self.direction_logits))


class CentralizedAttentionPolicy(nn.Module):
    """One permutation-equivariant policy that controls all ten minnows."""

    def __init__(
        self,
        embedding_size: int = 64,
        attention_heads: int = 4,
        attention_layers: int = 2,
        feedforward_size: int = 128,
        minnow_feature_count: int = MINNOW_FEATURE_COUNT,
        global_feature_count: int = GLOBAL_FEATURE_COUNT,
    ) -> None:
        super().__init__()

        if embedding_size % attention_heads != 0:
            raise ValueError("embedding_size must be divisible by attention_heads")
        self.minnow_feature_count = minnow_feature_count
        self.global_feature_count = global_feature_count

        self.minnow_encoder = nn.Sequential(
            nn.Linear(minnow_feature_count, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_count, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.LayerNorm(embedding_size),
        )

        attention_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=attention_heads,
            dim_feedforward=feedforward_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            attention_layer,
            num_layers=attention_layers,
            norm=nn.LayerNorm(embedding_size),
            enable_nested_tensor=False,
        )

        self.actor_head = nn.Linear(embedding_size, 2)
        nn.init.normal_(self.actor_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.actor_head.bias)
        with torch.no_grad():
            self.actor_head.bias[0] = 1.0
        self.team_value_head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, 1),
        )
        self.individual_value_head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, 1),
        )

        # One shared exploration scale preserves minnow interchangeability.
        self.direction_log_std = nn.Parameter(torch.tensor(-0.5))

    def forward(self, minnow_features: Tensor, global_features: Tensor) -> PolicyOutput:
        """Process a batch of complete board states.

        By default, minnow_features has shape [batch, minnows, 9] with
        [x, y, speed, active, dead, safe, bottom wall distance, top wall
        distance, finish distance]. global_features has shape [batch, 4]
        with [shark_x, shark_y, time_remaining, unresolved slow fraction].
        """
        self._validate_shapes(minnow_features, global_features)

        minnow_tokens = self.minnow_encoder(minnow_features)
        global_token = self.global_encoder(global_features).unsqueeze(1)
        tokens = torch.cat((global_token, minnow_tokens), dim=1)
        contextual_tokens = self.attention(tokens)

        contextual_global = contextual_tokens[:, 0]
        contextual_minnows = contextual_tokens[:, 1:]

        direction_logits = self.actor_head(contextual_minnows)
        direction_log_std = self.direction_log_std.clamp(
            math.log(MINIMUM_ACTION_STD), 2.0
        ).expand_as(direction_logits)
        team_value = self.team_value_head(contextual_global).squeeze(-1)
        individual_values = self.individual_value_head(contextual_minnows).squeeze(-1)

        return PolicyOutput(
            direction_logits=direction_logits,
            direction_log_std=direction_log_std,
            team_value=team_value,
            individual_values=individual_values,
        )

    def _validate_shapes(
        self, minnow_features: Tensor, global_features: Tensor
    ) -> None:
        if (
            minnow_features.ndim != 3
            or minnow_features.shape[1] < 1
            or minnow_features.shape[2] != self.minnow_feature_count
        ):
            raise ValueError(
                "minnow_features must have shape [batch, minnows, "
                f"{self.minnow_feature_count}], "
                f"received {tuple(minnow_features.shape)}"
            )
        if (
            global_features.ndim != 2
            or global_features.shape[1] != self.global_feature_count
        ):
            raise ValueError(
                "global_features must have shape [batch, "
                f"{self.global_feature_count}], "
                f"received {tuple(global_features.shape)}"
            )
        if minnow_features.shape[0] != global_features.shape[0]:
            raise ValueError("minnow and global batches must have the same size")


def bound_direction_actions(actions: Tensor) -> Tensor:
    """Keep velocity controls inside the unit circle while preserving throttle."""
    if actions.shape[-1] != 2:
        raise ValueError("direction actions must have a final dimension of 2")
    magnitudes = torch.linalg.vector_norm(actions, dim=-1, keepdim=True)
    return actions / magnitudes.clamp_min(1.0)
