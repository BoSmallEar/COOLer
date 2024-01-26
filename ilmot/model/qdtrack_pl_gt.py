"""Incremental learning module for QD-Track with pseudo-label (PL)."""
from typing import Dict, List, Tuple
from functools import reduce

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
)
from ilmot.common.module import build_module
from ilmot.model.track.similarity import BaseSimilarityHead
from ilmot.model.track.graph import BaseTrackGraph
from ilmot.model.track.utils import split_key_ref_inputs
from ilmot.common.bbox.utils import bbox_iou
from ilmot.model.utils import postprocess_predictions, predictions_to_scalabel
from ilmot.struct import Boxes2D

from .model_utils import (
    align_and_update_state_dicts,
    build_model,
    update_key_to_vis4d,
)
from .detect import ILMOTMMDetector
from .base import ILMOTBaseModel


class QDTrackPLGT(ILMOTBaseModel):  # type: ignore
    """QD-track based on pseudo-label."""

    # pylint: disable=too-many-ancestors
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=abstract-method

    def __init__(
        self,
        detector: ModuleCfg,
        similarity: ModuleCfg,
        track_graph: ModuleCfg,
        *args: ArgsType,
        score_threshold: float = 0.7,
        iou_gt_threshold: float = 0.5,
        ignore_new_class: bool = False,
        use_det_pl: bool = False,
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

        self.score_threshold = score_threshold
        self.iou_gt_threshold = iou_gt_threshold
        self.use_det_pl = use_det_pl
        self.total_det_box = 0

        rank_zero_info(f"category mapping: {self.category_mapping}")
        detector["category_mapping"] = self.category_mapping

        self.detector = build_model(detector)

        assert isinstance(self.detector, (ILMOTMMDetector))

        self.similarity_head = build_module(
            similarity, bound=BaseSimilarityHead
        )

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
        detector_state_dict = self.detector.state_dict()
        similarity_state_dict = self.similarity_head.state_dict()

        loaded_detector_state_dict: Dict[str, torch.Tensor] = {}
        loaded_similarity_state_dict: Dict[str, torch.Tensor] = {}
        for key, value in loaded_state_dict.items():
            if not load_vist:
                if key.startswith("detector."):
                    key = key[9:]
                    loaded_detector_state_dict[key] = value
                elif key.startswith("similarity_head."):
                    key = key[16:]
                    loaded_similarity_state_dict[key] = value
            else:
                if key.startswith("student_detector."):
                    key = key[17:]
                    key = update_key_to_vis4d(key)
                    loaded_detector_state_dict[key] = value
                elif key.startswith("detector."):
                    key = key[9:]
                    key = update_key_to_vis4d(key)
                    loaded_detector_state_dict[key] = value
                elif key.startswith("student_similarity_head."):
                    key = key[24:]
                    loaded_similarity_state_dict[key] = value
                elif key.startswith("similarity_head."):
                    key = key[16:]
                    loaded_similarity_state_dict[key] = value

        align_and_update_state_dicts(
            detector_state_dict, loaded_detector_state_dict
        )

        align_and_update_state_dicts(
            similarity_state_dict, loaded_similarity_state_dict
        )
        self.detector.load_state_dict(detector_state_dict)
        self.similarity_head.load_state_dict(similarity_state_dict)

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

        # feature extraction
        x_student = self.detector.extract_features(key_inputs)

        # original forward pass of qd-track
        key_x = x_student
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        # proposal generation
        rpn_losses, key_proposals = self.detector.generate_proposals(
            key_inputs, key_x, key_targets
        )
        with torch.no_grad():
            ref_proposals = [
                self.detector.generate_proposals(inp, x, tgt)[1]
                for inp, x, tgt in zip(ref_inputs, ref_x, ref_targets)
            ]

        # bbox head
        roi_losses, _ = self.detector.generate_detections(
            key_inputs, key_x, key_proposals, key_targets
        )
        det_losses = {**rpn_losses, **roi_losses}

         # if use detection pseudo label, remove track label from old classes
        if self.use_det_pl:
            for k_t in key_targets:
                for bb in k_t.boxes2d:
                    mask = bb.class_ids >= self.num_old_cat
                    bb.class_ids = bb.class_ids[mask]
                    bb.boxes = bb.boxes[mask]
                    bb.track_ids = bb.track_ids[mask]
 
            for r_tt in ref_targets:
                for r_t in r_tt:
                    for bb in r_t.boxes2d:
                        mask = bb.class_ids >= self.num_old_cat
                        bb.class_ids = bb.class_ids[mask]
                        bb.boxes = bb.boxes[mask]
                        bb.track_ids = bb.track_ids[mask]
                        
        # track head
        track_losses, _ = self.similarity_head.forward_train(
            [key_inputs, *ref_inputs],
            [key_proposals, *ref_proposals],
            [key_x, *ref_x],
            [key_targets, *ref_targets],
        )

        return {**det_losses, **track_losses}

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

        # detector
        feat = self.detector.extract_features(inputs)
        proposals = self.detector.generate_proposals(inputs, feat)
        detections, _ = self.detector.generate_detections(
            inputs, feat, proposals
        )

        # if inputs.targets.boxes2d is not None:
        #     roi_output = self.detector.get_roi_output(
        #         feat, inputs.targets.boxes2d
        #     )
        #     gt_scores = roi_output["cls_score"]
        #     gt_scores = F.softmax(gt_scores, dim=-1)
        #     gt_labels = inputs.targets.boxes2d[0].class_ids
        assert detections is not None

        # from vist.vis.image import imshow_bboxes
        # imshow_bboxes(inputs.images.tensor[0], detections)

        # similarity head
        embeddings = self.similarity_head(inputs, detections, feat)

        all_output = {}

        det_outs: Dict[str, List[TLabelInstance]] = {"detect": [d.clone() for d in detections]}  # type: ignore # pylint: disable=line-too-long
        postprocess_predictions(
            inputs,
            det_outs,
            self.detector.clip_bboxes_to_image,
            self.detector.resolve_overlap,
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
            inputs, track_outs, self.detector.clip_bboxes_to_image
        )
        track_outputs = predictions_to_scalabel(track_outs, self.cat_mapping)
        all_output.update(track_outputs)

        # all_output.update({"scores": [gt_scores, gt_labels]})
        return all_output

    def forward_predict(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Forward the prediction results."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"

        inputs = batch_inputs[0]
        feat = self.detector.extract_features(inputs)
        proposals = self.detector.generate_proposals(inputs, feat)
        detections, _ = self.detector.generate_detections(
            inputs, feat, proposals
        )
        frame_id = inputs.metadata[0].frameIndex

        # detection dataset
        if self.use_det_pl:
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

            det_outs: Dict[str, List[TLabelInstance]] = {"track": [d.clone() for d in detections]}  # type: ignore # pylint: disable=line-too-long

            postprocess_predictions(
                inputs,
                det_outs,
                self.detector.clip_bboxes_to_image,
                self.detector.resolve_overlap,
            )
            det_outputs = predictions_to_scalabel(det_outs, self.cat_mapping)

            current_det_box = 0

            for det_frame in det_outputs["track"]:
                current_det_box += len(det_frame)
                for det in det_frame:
                    det.id = str(int(det.id) + self.total_det_box)

            self.total_det_box += current_det_box

            # stat_dict = self.compute_statistics(
            #     detections, inputs.targets.boxes2d
            # )

            # det_outputs.update(stat_dict)

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
        embeddings = self.similarity_head(inputs, detections, feat)
        predictions = LabelInstances(detections, instance_masks=None)
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)
        track_outs: Dict[str, List[TLabelInstance]] = {"track": tracks.boxes2d}  # type: ignore # pylint: disable=line-too-long

        postprocess_predictions(
            inputs, track_outs, self.detector.clip_bboxes_to_image
        )
        track_outputs = predictions_to_scalabel(track_outs, self.cat_mapping)

        # stat_dict = self.compute_statistics(
        #     tracks.boxes2d, inputs.targets.boxes2d
        # )

        # track_outputs.update(stat_dict)

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

        # feature extraction
        x_student = self.detector.extract_features(inputs)

        # original forward pass of qd-track
        key_x = x_student

        # proposal generation
        _, key_proposals = self.detector.generate_proposals(
            inputs, key_x, inputs.targets
        )

        # contrast head
        sampled_proposals = self.detector.sample_proposals_contrast(
            inputs.targets.boxes2d, key_proposals, use_bg=add_background
        )

        class_ids = [p.class_ids for p in sampled_proposals]

        embeddings = []
        for proposal in sampled_proposals:
            embedding = self.detector.get_proposals_embeddings(
                key_x, [proposal]
            )
            embeddings.append(embedding)

        embeddings = torch.cat(embeddings, dim=0)
        class_ids = torch.cat(class_ids, dim=0)

        return embeddings, class_ids