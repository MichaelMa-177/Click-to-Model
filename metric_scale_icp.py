"""Compatibility imports for the packaged metric-scale implementation."""

from click_to_model.reconstruction.metric_scale import (
    MetricScaleResult,
    RegistrationMetrics,
    estimate_similarity,
    masked_depth_points,
    recover_metric_scale_icp,
    robust_extent,
)

__all__ = [
    "MetricScaleResult",
    "RegistrationMetrics",
    "estimate_similarity",
    "masked_depth_points",
    "recover_metric_scale_icp",
    "robust_extent",
]
