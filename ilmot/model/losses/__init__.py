"""Loss specialized for incremental learning."""
from .distillation_losses import (
    FeatureDistillationLoss,
    ROIDistillationLoss,
    RPNDistillationLoss,
    SimilarityDistillationLoss,
)

from .contrast_losses import (
    MeanContrastLoss,
    HingeContrastLoss,
    AdaptiveContrastLoss,
    AdaptiveContrastLoss2,
    MahaContrastLoss,
    MahaContrastLossNew,
    MahaContrastLossLog,
)


from .base import BaseLoss
from .embedding_distance import EmbeddingDistanceLoss
from .multi_pos_cross_entropy import MultiPosCrossEntropyLoss

__all__ = [
    "FeatureDistillationLoss",
    "ROIDistillationLoss",
    "RPNDistillationLoss",
    "SimilarityDistillationLoss",
    "MeanContrastLoss",
    "HingeContrastLoss",
    "AdaptiveContrastLoss",
    "AdaptiveContrastLoss2",
    "MahaContrastLoss",
    "MahaContrastLossNew",
    "MahaContrastLossLog",
    "BaseLoss",
    "EmbeddingDistanceLoss",
    "MultiPosCrossEntropyLoss",
]
