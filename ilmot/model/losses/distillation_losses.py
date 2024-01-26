"""Distillation losses for incremental learning."""
from typing import Dict, List

import torch
from torch.nn.functional import l1_loss, mse_loss
from ilmot.model.losses.base import BaseLoss  # type:ignore


class FeatureDistillationLoss(BaseLoss):  # type:ignore
    """Feature distillation loss."""

    def __init__(self, weight: float = 1.0):
        """Init."""
        super().__init__()
        self.weight = weight

    def __call__(  # pylint: disable=arguments-differ
        self,
        x_student: Dict[str, torch.Tensor],
        x_teacher: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_student: (scale_name, b*c*h*w), multiscale feature from the
                student model.
            x_teacher: (scale_name, b*c*h*w), multiscale feature from the
                teacher model.

        Returns:
            The feature distillation loss.

        """
        loss = []
        element = []
        for scale in x_student.keys():
            x_student_current = x_student[scale]
            x_teacher_current = x_teacher[scale]

            x_student_current = x_student_current - x_student_current.mean(
                dim=[-2, -1], keepdim=True
            )
            x_teacher_current = x_teacher_current - x_teacher_current.mean(
                dim=[-2, -1], keepdim=True
            )

            mask = x_teacher_current > x_student_current
            mask = mask.long()
            loss_current = l1_loss(
                x_teacher_current, x_student_current, reduction="none"
            )
            loss_current *= mask
            loss.append(torch.sum(loss_current))  # type: ignore
            element.append(loss_current.numel())

        result = sum(loss) / sum(element)

        return result * self.weight  # type: ignore


class RPNDistillationLoss(BaseLoss):  # type:ignore
    """RPN distillation loss."""

    def __init__(self, reg_valid_threshold: float = 0.1, weight: float = 1.0):
        """Init."""
        super().__init__()
        self.reg_valid_threshold = reg_valid_threshold
        self.weight = weight

    def __call__(  # pylint: disable=arguments-differ
        self,
        rpn_student: Dict[str, List[torch.Tensor]],
        rpn_teacher: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            rpn_student: Dictionary of multiscale classification scores and
            bounding box predictions. For cls scores, the shape is b*x*h*w,
            where x is number of anchors at each location. For bbox preds, the
            shape is b*4x*h*w.
            rpn_teacher: same output by rpn from the teacher model.

        Returns:
            The RPN distillation loss.

        """
        rpn_student_cls = rpn_student["cls_scores"]
        rpn_teacher_cls = rpn_teacher["cls_scores"]
        rpn_student_reg = rpn_student["bbox_preds"]
        rpn_teacher_reg = rpn_teacher["bbox_preds"]

        cls_loss = []
        achnor_num = []
        for rpn_student_cls_current, rpn_teacher_cls_current in zip(
            rpn_student_cls, rpn_teacher_cls
        ):
            cls_mask = rpn_teacher_cls_current - rpn_student_cls_current > 0
            cls_mask = cls_mask.long()
            loss_current = mse_loss(
                rpn_teacher_cls_current,
                rpn_student_cls_current,
                reduction="none",
            )
            loss_current *= cls_mask
            cls_loss.append(torch.sum(loss_current))  # type: ignore
            achnor_num.append(loss_current.numel())
        cls_result = sum(cls_loss) / sum(achnor_num)

        reg_loss = []
        for rpn_student_reg_current, rpn_teacher_reg_current, ss, tt in zip(
            rpn_student_reg, rpn_teacher_reg, rpn_student_cls, rpn_teacher_cls
        ):
            reg_mask = tt - ss > self.reg_valid_threshold
            reg_mask = reg_mask.long()
            reg_mask = torch.repeat_interleave(  # type: ignore
                reg_mask, 4, dim=1
            )
            loss_current = mse_loss(
                rpn_teacher_reg_current,
                rpn_student_reg_current,
                reduction="none",
            )
            assert reg_mask.shape == loss_current.shape
            loss_current *= reg_mask
            reg_loss.append(torch.sum(loss_current))  # type: ignore
        reg_result = sum(reg_loss) / sum(achnor_num)

        result = cls_result + reg_result

        return result * self.weight  # type: ignore


class ROIDistillationLoss(BaseLoss):  # type:ignore
    """ROI distillation loss."""

    def __init__(self, weight: float = 1.0):
        """Init."""
        super().__init__()
        self.weight = weight

    def __call__(  # pylint: disable=arguments-differ
        self,
        roi_student: Dict[str, torch.Tensor],
        roi_teacher: Dict[str, torch.Tensor],
        num_old_cat: int,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            roi_student: Dictionary of classification scores and bounding box
            refinements for the sampled proposals. For cls scores, the shape is
            (b*n) * (cats + 1), where n is sampled proposal in each image, cats
            is the total number of categories without the background. For bbox
            preds, the shape is (b*n) * (4*cats)
            roi_teacher: Same output by roi from the teacher output.
            num_old_cat: Number of old categories in the incremental setting.

        Returns:
            The ROI distillation loss.

        """
        roi_teacher_cls = roi_teacher["cls_score"]
        roi_teacher_reg = roi_teacher["bbox_pred"]
        roi_student_cls = roi_student["cls_score"]
        roi_student_reg = roi_student["bbox_pred"]

        # remove output for new classes in the output
        roi_student_cls = torch.cat(  # type: ignore
            (roi_student_cls[:, :num_old_cat], roi_student_cls[:, [-1]]),
            dim=-1,
        )
        roi_teacher_cls = torch.cat(  # type: ignore
            (roi_teacher_cls[:, :num_old_cat], roi_teacher_cls[:, [-1]]),
            dim=-1,
        )
        roi_student_reg = roi_student_reg[:, : 4 * num_old_cat]
        roi_teacher_reg = roi_teacher_reg[:, : 4 * num_old_cat]

        # subtract the mean of cls score over the cls dimension
        roi_teacher_cls -= torch.mean(  # type: ignore
            roi_teacher_cls, dim=-1, keepdim=True
        )
        roi_student_cls -= torch.mean(  # type: ignore
            roi_student_cls, dim=-1, keepdim=True
        )

        assert roi_student_cls.shape == roi_teacher_cls.shape
        assert roi_student_reg.shape == roi_teacher_reg.shape

        total_num = roi_teacher_cls.numel()
        cls_loss = mse_loss(roi_teacher_cls, roi_student_cls, reduction="none")
        reg_loss = mse_loss(roi_teacher_reg, roi_student_reg, reduction="none")

        result = torch.sum(cls_loss) + torch.sum(reg_loss)  # type: ignore
        result = result / total_num

        return result * self.weight  # type: ignore


class SimilarityDistillationLoss(BaseLoss):  # type:ignore
    """Similarity distillation loss."""

    def __init__(self, weight: float = 1.0):
        """Init."""
        super().__init__()
        self.weight = weight

    def __call__(  # pylint: disable=arguments-differ
        self,
        similarity_student: List[torch.Tensor],
        similarity_teacher: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            similarity_student: List of the embeddings output by the student
            similarity head. The length of the list is equal to the batch size.
            Each element is of shape M*N, where M is the number of proposals
            used for distillation and N is the dimension of the embedding.
            similarity_teacher: Same output from the teacher similarity head.

        Returns:
            The similarity distillation loss.

        """
        similarity_student_l = torch.cat(similarity_student, dim=0)
        similarity_teacher_l = torch.cat(similarity_teacher, dim=0)
        similarity_dot = torch.sum(  # type: ignore
            similarity_teacher_l * similarity_student_l, dim=-1
        )
        norm_student = torch.linalg.norm(similarity_student_l, dim=-1)
        norm_teacher = torch.linalg.norm(similarity_teacher_l, dim=-1)
        loss = (similarity_dot / (norm_student * norm_teacher) - 1) ** 2
        result = torch.sum(loss) / torch.numel(loss)  # type: ignore
        return result * self.weight  # type: ignore
