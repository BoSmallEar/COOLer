"""Incremental learning module for QD-Track with knowledge distillation."""
from typing import Dict, List, Tuple
from functools import reduce
import copy

import torch
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.utilities.distributed import rank_zero_info
from ilmot.struct import (
    ArgsType,
    ModuleCfg,
    LossesType,
    ModelOutput,
    InputSample,
    TLabelInstance,
    LabelInstances,
    Boxes2D,
)
from ilmot.common.module import build_module
from ilmot.model.losses import BaseLoss
from ilmot.model.track.similarity import BaseSimilarityHead
from ilmot.model.track.graph import BaseTrackGraph
from ilmot.model.track.utils import split_key_ref_inputs
from ilmot.common.bbox.utils import bbox_iou
from ilmot.model.utils import postprocess_predictions, predictions_to_scalabel

from .model_utils import (
    align_and_update_state_dicts,
    build_model,
    update_key_to_vis4d,
)
from .detect import ILMOTMMDetector
from .base import ILMOTBaseModel


class QDTrackKD(ILMOTBaseModel):  # type: ignore
    """QD-track based on KD."""

    # pylint: disable=too-many-ancestors
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=abstract-method

    def __init__(
        self,
        detector: ModuleCfg,
        feature_distillation_loss: ModuleCfg,
        rpn_distillation_loss: ModuleCfg,
        roi_distillation_loss: ModuleCfg,
        similarity_distillation_loss: ModuleCfg,
        similarity: ModuleCfg,
        track_graph: ModuleCfg,
        *args: ArgsType,
        polyak_average: bool = False,
        polyak_factor: float = 0.999,
        test_student: bool = True,
        score_threshold: float = 0.7,
        iou_gt_threshold: float = 0.5,
        ignore_new_class: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        # train category must be not not none to perform incremental learning
        assert self.train_category is not None

        self.num_new_cat = len(self.train_category)
        self.num_old_cat = len(self.category_mapping) - len(
            self.train_category
        )

        if ignore_new_class:
            for cat in self.train_category:
                self.category_mapping.pop(cat)

        self.cat_mapping = {v: k for k, v in self.category_mapping.items()}

        self.polyak_average = polyak_average
        self.polyak_factor = polyak_factor
        self.test_student = test_student
        self.score_threshold = score_threshold
        self.iou_gt_threshold = iou_gt_threshold

        rank_zero_info(f"category mapping: {self.category_mapping}")
        detector["category_mapping"] = self.category_mapping

        self.student_detector = build_model(detector)

        assert isinstance(self.student_detector, (ILMOTMMDetector))

        self.teacher_detector = build_model(detector)

        self.teacher_detector.requires_grad_(False)

        self.feature_distillation_loss = build_module(
            feature_distillation_loss, bound=BaseLoss
        )
        self.rpn_distillation_loss = build_module(
            rpn_distillation_loss, bound=BaseLoss
        )
        self.roi_distillation_loss = build_module(
            roi_distillation_loss, bound=BaseLoss
        )
        self.similarity_distillation_loss = build_module(
            similarity_distillation_loss, bound=BaseLoss
        )

        self.student_similarity_head = build_module(
            similarity, bound=BaseSimilarityHead
        )
        self.teacher_similarity_head = build_module(
            similarity, bound=BaseSimilarityHead
        )
        self.teacher_similarity_head.requires_grad_(False)

        self.track_graph = build_module(track_graph, bound=BaseTrackGraph)

    def load(self, ckpt: str, load_vist: bool = False) -> None:
        """Load weights from a checkpoint trained wtih previous class.

        Args:
            ckpt(str): Path of the checkpoint.
            load_vist: Whether or not to load the weight from the previous
            version of vis4d (VisT).
        """
        checkpoint = torch.load(
            ckpt, map_location=torch.device("cpu")  # type: ignore
        )
        loaded_state_dict = checkpoint["state_dict"]
        teacher_detector_state_dict = self.teacher_detector.state_dict()
        student_detector_state_dict = self.student_detector.state_dict()
        teacher_similarity_state_dict = (
            self.teacher_similarity_head.state_dict()
        )
        student_similarity_state_dict = (
            self.student_similarity_head.state_dict()
        )
        loaded_detector_state_dict: Dict[str, torch.Tensor] = {}
        loaded_similarity_state_dict: Dict[str, torch.Tensor] = {}
        for key, value in loaded_state_dict.items():
            if not load_vist:
                if key.startswith("student_detector."):
                    key = key[17:]
                    loaded_detector_state_dict[key] = value
                elif key.startswith("student_similarity_head."):
                    key = key[24:]
                    loaded_similarity_state_dict[key] = value
                if key.startswith("detector."):
                    key = key[9:]
                    loaded_detector_state_dict[key] = value
                elif key.startswith("similarity_head."):
                    key = key[16:]
                    loaded_similarity_state_dict[key] = value
            else:
                if key.startswith("detector."):
                    key = key[9:]
                    key = update_key_to_vis4d(key)
                    loaded_detector_state_dict[key] = value
                elif key.startswith("similarity_head."):
                    key = key[16:]
                    loaded_similarity_state_dict[key] = value

        align_and_update_state_dicts(
            teacher_detector_state_dict, loaded_detector_state_dict
        )
        align_and_update_state_dicts(
            student_detector_state_dict, loaded_detector_state_dict
        )
        align_and_update_state_dicts(
            teacher_similarity_state_dict, loaded_similarity_state_dict
        )
        align_and_update_state_dicts(
            student_similarity_state_dict, loaded_similarity_state_dict
        )
        self.teacher_detector.load_state_dict(teacher_detector_state_dict)
        self.teacher_detector.requires_grad_(False)
        self.teacher_similarity_head.load_state_dict(
            teacher_similarity_state_dict
        )
        self.teacher_similarity_head.requires_grad_(False)
        self.student_detector.load_state_dict(student_detector_state_dict)
        self.student_similarity_head.load_state_dict(
            student_similarity_state_dict
        )

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """Forward inputs while performing knowledge distillation.

        Args:
            batch_inputs: Input of batch samples for training.

        Returns:
            Model losses for back-propogation.

        """
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)
        key_targets, ref_targets = key_inputs.targets, [
            x.targets for x in ref_inputs
        ]
        self.teacher_detector.eval()
        self.teacher_similarity_head.eval()

        if self.polyak_average:
            for target_param, param in zip(
                self.student_detector.parameters(),
                self.teacher_detector.parameters(),
            ):
                param.data.copy_(
                    self.polyak_factor * param.data
                    + target_param.data * (1.0 - self.polyak_factor)
                )
            for target_param, param in zip(
                self.student_similarity_head.parameters(),
                self.teacher_similarity_head.parameters(),
            ):
                param.data.copy_(
                    self.polyak_factor * param.data
                    + target_param.data * (1.0 - self.polyak_factor)
                )

        # feature extraction
        key_inputs_teacher = copy.deepcopy(key_inputs)
        x_student = self.student_detector.extract_features(key_inputs)

        with torch.no_grad():
            x_teacher = self.teacher_detector.extract_features(
                key_inputs_teacher
            )

        # feature distillation loss
        feature_kd_loss = self.feature_distillation_loss(x_student, x_teacher)

        # rpn distillation loss
        rpn_student = self.student_detector.get_rpn_output(x_student)
        with torch.no_grad():
            rpn_teacher = self.teacher_detector.get_rpn_output(x_teacher)
        rpn_kd_loss = self.rpn_distillation_loss(rpn_student, rpn_teacher)

        # sample proposals
        with torch.no_grad():
            proposals_teacher = self.teacher_detector.sample_proposals(
                key_inputs, x_teacher
            )

        # roi distillation loss
        roi_student = self.student_detector.get_roi_output(
            x_student, proposals_teacher
        )
        with torch.no_grad():
            roi_teacher = self.teacher_detector.get_roi_output(
                x_teacher, proposals_teacher
            )
        roi_kd_loss = self.roi_distillation_loss(
            roi_student, roi_teacher, self.num_old_cat
        )

        # similarity distillation loss
        embeddings_student = self.student_similarity_head._head_forward(  # pylint: disable=protected-access
            x_student, proposals_teacher
        )
        with torch.no_grad():
            embeddings_teacher = self.teacher_similarity_head._head_forward(  # pylint: disable=protected-access
                x_teacher, proposals_teacher
            )

        similarity_kd_loss = self.similarity_distillation_loss(
            embeddings_student, embeddings_teacher
        )

        kd_losses = {
            "feature_kd_loss": feature_kd_loss,
            "rpn_kd_loss": rpn_kd_loss,
            "roi_kd_loss": roi_kd_loss,
            "similarity_kd_loss": similarity_kd_loss,
        }

        # original forward pass of qd-track
        key_x = x_student
        ref_x = [
            self.student_detector.extract_features(inp) for inp in ref_inputs
        ]

        # proposal generation
        rpn_losses, key_proposals = self.student_detector.generate_proposals(
            key_inputs, key_x, key_targets
        )
        with torch.no_grad():
            ref_proposals = [
                self.student_detector.generate_proposals(inp, x, tgt)[1]
                for inp, x, tgt in zip(ref_inputs, ref_x, ref_targets)
            ]

        # bbox head
        roi_losses, _ = self.student_detector.generate_detections(
            key_inputs, key_x, key_proposals, key_targets
        )
        det_losses = {**rpn_losses, **roi_losses}

        # track head

        track_losses, _ = self.student_similarity_head.forward_train(
            [key_inputs, *ref_inputs],
            [key_proposals, *ref_proposals],
            [key_x, *ref_x],
            [key_targets, *ref_targets],
        )
        return {**kd_losses, **det_losses, **track_losses}

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute model output during inference.

        Args:
            batch_inputs: Input of batch samples for testing.

        Return:
            Model outputs for post-precessing.

        """
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"

        inputs = batch_inputs[0]
        frame_id = inputs.metadata[0].frameIndex

        test_detector = (
            self.student_detector
            if self.test_student
            else self.teacher_detector
        )

        test_similarity_head = (
            self.student_similarity_head
            if self.test_student
            else self.teacher_similarity_head
        )
        # detector
        feat = test_detector.extract_features(inputs)
        proposals = test_detector.generate_proposals(inputs, feat)
        detections, _ = test_detector.generate_detections(
            inputs, feat, proposals
        )

        if inputs.targets.boxes2d is not None:
            roi_output = test_detector.get_roi_output(
                feat, inputs.targets.boxes2d
            )
            gt_scores = roi_output["cls_score"]
            gt_scores = F.softmax(gt_scores, dim=-1)
            gt_labels = inputs.targets.boxes2d[0].class_ids
        assert detections is not None

        # from vist.vis.image import imshow_bboxes
        # imshow_bboxes(inputs.images.tensor[0], detections)

        # similarity head
        embeddings = test_similarity_head(inputs, detections, feat)

        all_output = {}

        det_outs: Dict[str, List[TLabelInstance]] = {"detect": [d.clone() for d in detections]}  # type: ignore # pylint: disable=line-too-long
        postprocess_predictions(
            inputs,
            det_outs,
            test_detector.clip_bboxes_to_image,
            test_detector.resolve_overlap,
        )
        det_outputs = predictions_to_scalabel(det_outs, self.cat_mapping)
        all_output.update(det_outputs)
        # detection dataset
        if frame_id is None:
            return all_output

        predictions = LabelInstances(detections, instance_masks=None)

        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)
        track_outs: Dict[str, List[TLabelInstance]] = {"track": tracks.boxes2d}  # type: ignore # pylint: disable=line-too-long
        postprocess_predictions(
            inputs, track_outs, test_detector.clip_bboxes_to_image
        )
        track_outputs = predictions_to_scalabel(track_outs, self.cat_mapping)
        all_output.update(track_outputs)

        all_output.update({"scores": [gt_scores, gt_labels]})
        return all_output

    def forward_predict(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Forward the prediction results."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"

        inputs = batch_inputs[0]

        test_detector = (
            self.student_detector
            if self.test_student
            else self.teacher_detector
        )

        test_similarity_head = (
            self.student_similarity_head
            if self.test_student
            else self.teacher_similarity_head
        )

        feat = test_detector.extract_features(inputs)
        proposals = test_detector.generate_proposals(inputs, feat)
        detections, _ = test_detector.generate_detections(
            inputs, feat, proposals
        )
        frame_id = inputs.metadata[0].frameIndex

        # detection dataset
        if frame_id is None:
            for i, detection in enumerate(detections):
                gt = inputs.targets.boxes2d[i]
                gt_new_mask = gt.class_ids >= self.num_old_cat
                gt_new = Boxes2D(
                    gt.boxes[gt_new_mask],
                    gt.class_ids[gt_new_mask],
                    gt.track_ids[gt_new_mask],
                )
                if gt_new.boxes.shape[0] > 0 and detection.boxes.shape[0] > 0:
                    ious = bbox_iou(detection, gt_new)
                    max_iou, _ = torch.max(ious, dim=1)
                else:
                    max_iou = torch.zeros_like(detection.class_ids)
                mask = reduce(
                    torch.logical_and,
                    [
                        detection.class_ids < self.num_old_cat,
                        max_iou < self.iou_gt_threshold,
                        detection.boxes[:, -1] > self.score_threshold,
                    ],
                )
                detection.boxes = detection.boxes[mask]
                detection.class_ids = detection.class_ids[mask]

            det_outs: Dict[str, List[TLabelInstance]] = {"detect": [d.clone() for d in detections]}  # type: ignore # pylint: disable=line-too-long

            postprocess_predictions(
                inputs,
                det_outs,
                test_detector.clip_bboxes_to_image,
                test_detector.resolve_overlap,
            )
            det_outputs = predictions_to_scalabel(det_outs, self.cat_mapping)

            stat_dict = self.compute_statistics(
                detections, inputs.targets.boxes2d
            )

            det_outputs.update(stat_dict)

            return det_outputs
        # tracking dataset

        # filter boxes with low confidence and from new class
        for i, detection in enumerate(detections):
            gt = inputs.targets.boxes2d[i]
            gt_new_mask = gt.class_ids >= self.num_old_cat
            gt_new = Boxes2D(
                gt.boxes[gt_new_mask],
                gt.class_ids[gt_new_mask],
                gt.track_ids[gt_new_mask],
            )
            if gt_new.boxes.shape[0] > 0 and detection.boxes.shape[0] > 0:
                ious = bbox_iou(detection, gt_new)
                max_iou, _ = torch.max(ious, dim=1)
            else:
                max_iou = torch.zeros_like(detection.class_ids)
            mask = reduce(
                torch.logical_and,
                [
                    detection.class_ids < self.num_old_cat,
                    max_iou < self.iou_gt_threshold,
                ],
            )
            detection.boxes = detection.boxes[mask]
            detection.class_ids = detection.class_ids[mask]

        # similarity head
        embeddings = test_similarity_head(inputs, detections, feat)
        predictions = LabelInstances(detections, instance_masks=None)
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)
        track_outs: Dict[str, List[TLabelInstance]] = {"track": tracks.boxes2d}  # type: ignore # pylint: disable=line-too-long

        postprocess_predictions(
            inputs, track_outs, test_detector.clip_bboxes_to_image
        )
        track_outputs = predictions_to_scalabel(track_outs, self.cat_mapping)

        stat_dict = self.compute_statistics(
            tracks.boxes2d, inputs.targets.boxes2d
        )

        track_outputs.update(stat_dict)

        return track_outputs

    def compute_statistics(self, predicted_boxes, gt_boxes):
        """Compute fp, fn, tp of the model."""
        old_cat_fps = np.zeros(self.num_old_cat)
        old_cat_fns = np.zeros(self.num_old_cat)
        old_cat_tps = np.zeros(self.num_old_cat)
        for pred, gt in zip(predicted_boxes, gt_boxes):
            for i in range(self.num_old_cat):
                pred_box = pred.boxes[pred.class_ids == i]
                gt_box = gt.boxes[gt.class_ids == i]
                if pred_box.shape[0] == 0:
                    old_cat_fns[i] += gt_box.shape[0]
                elif gt_box.shape[0] == 0:
                    old_cat_fps[i] += pred_box.shape[0]
                else:
                    pred_box = Boxes2D(pred_box)
                    gt_box = Boxes2D(gt_box)
                    iou_result = bbox_iou(pred_box, gt_box)
                    max_iou_gt, _ = torch.max(iou_result, dim=0)
                    max_iou_pred, _ = torch.max(iou_result, dim=1)
                    old_cat_fns[i] += torch.sum(
                        max_iou_gt < self.iou_gt_threshold
                    )
                    old_cat_fps[i] += torch.sum(
                        max_iou_pred < self.iou_gt_threshold
                    )
                    old_cat_tps[i] += torch.sum(
                        max_iou_gt >= self.iou_gt_threshold
                    )

        return {
            "num_tp": old_cat_tps,
            "num_fn": old_cat_fns,
            "num_fp": old_cat_fps,
        }

    def forward_embeddings(
        self, batch_inputs: List[InputSample], add_background: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the embeddings of GTs for visualization.

        Args:
            batch_inputs: A single input from a driving sequence.
            add_background: Whether to also forward embeddings from the bg.

        Returns:
            A tuple of three tensors, where each row corresponds to an object
            to be tracked. The first tensor is the embedding, the second one is
            the class id, and the last one is the track id.

        """
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"

        inputs = batch_inputs[0]
        frame_id = inputs.metadata[0].frameIndex
        if frame_id == 0:
            frame_id = inputs.metadata[0].frameIndex

        test_detector = (
            self.student_detector
            if self.test_student
            else self.teacher_detector
        )

        test_similarity_head = (
            self.student_similarity_head
            if self.test_student
            else self.teacher_similarity_head
        )

        # detector
        feat = test_detector.extract_features(inputs)
        detections = inputs.targets.boxes2d
        assert detections is not None

        # from vist.vis.image import imshow_bboxes
        # imshow_bboxes(inputs.images.tensor[0], detections)

        # similarity head
        embeddings = test_similarity_head(inputs, detections, feat)
        embeddings = embeddings[0]
        class_ids = inputs.targets.boxes2d[0].class_ids
        track_ids = inputs.targets.boxes2d[0].track_ids

        if add_background:
            proposals = test_detector.generate_proposals(inputs, feat)
            ious = bbox_iou(proposals[0], detections[0])
            max_iou, _ = torch.max(ious, dim=1)
            mask = max_iou < 0.3
            proposals[0] = proposals[0]
            embeddings_bg = test_similarity_head(inputs, proposals, feat)
            embeddings_bg = embeddings_bg[0][mask]
            embeddings_bg = embeddings_bg[
                torch.randperm(embeddings_bg.shape[0])[:10]
            ]
            embeddings = torch.cat([embeddings, embeddings_bg], dim=0)
            class_ids = torch.cat(
                [
                    class_ids,
                    torch.full((embeddings_bg.shape[0],), -1).to(
                        class_ids.device
                    ),
                ]
            )
            track_ids = torch.cat(
                [
                    track_ids,
                    torch.full((embeddings_bg.shape[0],), -1).to(
                        track_ids.device
                    ),
                ]
            )

        return embeddings, class_ids, track_ids
