"""Match predictions and targets according to maximum 2D IoU."""
from typing import List, Tuple

import torch

from ilmot.common.bbox.utils import bbox_iou
from ilmot.struct import Boxes2D

from .base import BaseMatcher, MatchResult


# implementation modified from:
# https://github.com/facebookresearch/detectron2/
class MaxIoUMatcher(BaseMatcher):
    """MaxIoUMatcher class."""

    def __init__(
        self,
        thresholds: List[float],
        labels: List[int],
        allow_low_quality_matches: bool,
    ):
        """Init."""
        super().__init__()
        self.allow_low_quality_matches = allow_low_quality_matches
        if not thresholds[0] > 0:
            raise ValueError(
                f"Lowest threshold {thresholds[0]} must be greater than 0!"
            )
        eps = 1e-4
        thresholds.insert(0, 0.0 - eps)
        thresholds.append(1.0 + eps)
        if not all(
            (lo <= hi for (lo, hi) in zip(thresholds[:-1], thresholds[1:]))
        ):
            raise ValueError("Thresholds must be in ascending order!")

        assert all(
            (v in [-1, 0, 1] for v in labels)
        ), "labels must be in [-1, 0, 1]!"
        assert (
            len(labels) == len(thresholds) - 1
        ), "Labels must be of len(thresholds) + 1."
        self.thresholds = thresholds
        self.labels = labels

    def __call__(  # type: ignore
        self, boxes: List[Boxes2D], targets: List[Boxes2D]
    ) -> List[MatchResult]:
        """Match all boxes to targets based on maximum IoU."""
        result = []
        for b, t in zip(boxes, targets):
            if len(t) == 0:
                matches = torch.zeros(len(b), device=b.device)
                match_labels = torch.zeros(len(b), device=b.device)
                match_iou = torch.zeros(len(b), device=b.device)
            else:
                # M x N matrix, where M = num gt, N = num proposals
                match_quality_matrix = bbox_iou(t, b)

                # matches N x 1 = index of assigned gt i.e.  range [0, M)
                # match_labels N x 1, 0 = negative, -1 = ignore, 1 = positive
                matches, match_labels = self._compute_matches(
                    match_quality_matrix
                )
                match_iou = match_quality_matrix[
                    matches, torch.arange(0, len(b), device=b.device)
                ]

            result.append(
                MatchResult(
                    assigned_gt_indices=matches,
                    assigned_labels=match_labels,
                    assigned_gt_iou=match_iou,
                )
            )
        return result

    def _compute_matches(
        self, match_quality_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute matching boxes and their labels w/ match_quality_matrix."""
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.shape[1],), 0, dtype=torch.int64
            )
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.shape[1],),
                self.labels[0],
                dtype=torch.int8,
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # Max over gt elements (dim 0) --> best gt for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(
            self.labels, self.thresholds[:-1], self.thresholds[1:]
        ):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            _set_low_quality_matches(match_labels, match_quality_matrix)

        return matches, match_labels


def _set_low_quality_matches(
    match_labels: torch.Tensor, match_quality_matrix: torch.Tensor
) -> None:
    """Set matches for predictions that have only low-quality matches.

    See Sec. 3.1.2 of Faster R-CNN: https://arxiv.org/abs/1506.01497
    """
    highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
    pred_inds_with_highest_quality = (
        match_quality_matrix == highest_quality_foreach_gt[:, None]
    ).nonzero()[:, 1]
    match_labels[pred_inds_with_highest_quality] = 1
