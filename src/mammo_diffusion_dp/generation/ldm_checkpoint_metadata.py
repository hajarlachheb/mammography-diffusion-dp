"""Small, dependency-free helpers for LDM checkpoint provenance."""

import math


def _scales_match(left, right):
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-5,
        abs_tol=1e-8,
    )


def resolve_scale_factor(
    checkpoint_scale,
    training_info_scale,
    explicit_scale,
    vae_scale,
    allow_override=False,
):
    """Resolve latent scaling while rejecting inconsistent generation metadata."""
    if (
        checkpoint_scale is not None
        and training_info_scale is not None
        and not _scales_match(checkpoint_scale, training_info_scale)
    ):
        raise ValueError(
            f"Checkpoint scale_factor={checkpoint_scale} conflicts with "
            f"training_info.json scale_factor={training_info_scale}"
        )

    recorded_scale = (
        float(checkpoint_scale)
        if checkpoint_scale is not None
        else (
            float(training_info_scale)
            if training_info_scale is not None
            else None
        )
    )
    if (
        explicit_scale is not None
        and recorded_scale is not None
        and not _scales_match(explicit_scale, recorded_scale)
        and not allow_override
    ):
        raise ValueError(
            f"Requested --scale_factor={explicit_scale}, but the training "
            f"checkpoint records scale_factor={recorded_scale}. Remove the "
            "override or pass --allow_scale_factor_override for an intentional ablation."
        )
    if explicit_scale is not None:
        return float(explicit_scale), "explicit override"
    if recorded_scale is not None:
        return recorded_scale, "checkpoint metadata"
    return float(vae_scale), "VAE config fallback (legacy checkpoint)"
