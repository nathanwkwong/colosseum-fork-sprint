"""Three-case collusion identification: matched arms, shared evidence, frozen rule."""

from __future__ import annotations

LABELS = (
    "coordinated_action",
    "common_incentives",
    "cheap_talk",
)

FEATURE_NAMES = (
    "conversation",
    "coalition_advantage_residual",
    "dependence",
)
