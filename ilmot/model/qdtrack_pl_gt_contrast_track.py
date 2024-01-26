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
    Boxes2D,
)
from ilmot.common.module import build_module
from ilmot.model.losses import BaseLoss
from ilmot.model.track.similarity import BaseSimilarityHead
from ilmot.model.track.graph import BaseTrackGraph
from ilmot.model.track.utils import cosine_similarity
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
from .losses import MeanContrastLoss  # pylint: disable=unused-import


class QDTrackPLGTContrastTrack(ILMOTBaseModel):  # type: ignore
    """QD-track based on pseudo-label."""

    # pylint: disable=too-many-ancestors
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=abstract-method

    def __init__(
        self,
        detector: ModuleCfg,
        similarity: ModuleCfg,
        track_graph: ModuleCfg,
        contrast_loss: ModuleCfg,
        *args: ArgsType,
        score_threshold: float = 0.7,
        iou_gt_threshold: float = 0.5,
        skip_first: bool = False,
        skip_only_new: bool = False,
        use_bg: bool = True,
        ignore_new_class: bool = False,
        enable_det_contrast: bool = False,
        enable_track_contrast: bool = False,
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
        self.skip_first = skip_first
        self.skip_only_new = skip_only_new
        self.use_bg = use_bg

        rank_zero_info(f"category mapping: {self.category_mapping}")
        detector["category_mapping"] = self.category_mapping

        self.detector = build_model(detector)

        assert isinstance(self.detector, (ILMOTMMDetector))

        self.similarity_head = build_module(
            similarity, bound=BaseSimilarityHead
        )

        self.track_graph = build_module(track_graph, bound=BaseTrackGraph)

        # align with the OWOD
        self.memory_per_class = 1000
        self.memory_num_for_contrast = 100
        self.det_embedding_length = 1024
        self.track_embedding_length = 1024

        self.enable_det_contrast = enable_det_contrast
        self.enable_track_contrast = enable_track_contrast

        if self.enable_det_contrast:
            self.register_buffer(
                "det_embedding_means",
                torch.zeros(
                    [
                        len(self.category_mapping) + 1,
                        self.det_embedding_length,
                    ],
                    dtype=torch.float,
                ),
            )
            self.register_buffer(
                "det_embedding_stds",
                torch.zeros(
                    [
                        len(self.category_mapping) + 1,
                        self.det_embedding_length,
                    ],
                    dtype=torch.float,
                ),
            )
            self.register_buffer(
                "det_memory_embedding",
                torch.zeros(
                    [
                        len(self.category_mapping) + 1,
                        self.memory_per_class,
                        self.det_embedding_length,
                    ],
                    dtype=torch.float,
                ),
            )
            self.register_buffer(
                "det_memory_pointer",
                torch.zeros(
                    [len(self.category_mapping) + 1], dtype=torch.long
                ),
            )

            self.register_buffer(
                "det_memory_num",
                torch.zeros(
                    [len(self.category_mapping) + 1], dtype=torch.long
                ),
            )
            self.det_contrast_loss = build_module(
                contrast_loss, bound=BaseLoss
            )

        if self.enable_track_contrast:
            self.register_buffer(
                "track_embedding_means",
                torch.zeros(
                    [
                        len(self.category_mapping) + 1,
                        self.track_embedding_length,
                    ],
                    dtype=torch.float,
                ),
            )
            self.register_buffer(
                "track_embedding_stds",
                torch.zeros(
                    [
                        len(self.category_mapping) + 1,
                        self.track_embedding_length,
                    ],
                    dtype=torch.float,
                ),
            )
            self.register_buffer(
                "track_memory_embedding",
                torch.zeros(
                    [
                        len(self.category_mapping) + 1,
                        self.memory_per_class,
                        self.track_embedding_length,
                    ],
                    dtype=torch.float,
                ),
            )
            self.register_buffer(
                "track_memory_pointer",
                torch.zeros(
                    [len(self.category_mapping) + 1], dtype=torch.long
                ),
            )

            self.register_buffer(
                "track_memory_num",
                torch.zeros(
                    [len(self.category_mapping) + 1], dtype=torch.long
                ),
            )
            self.track_contrast_loss = build_module(
                contrast_loss, bound=BaseLoss
            )

        # sample proposal to update the embeddings
        self.sample_num_per_image = 2
        self.polyak_factor = 0.999
        # try only perform contrast learning between old and new classes

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
        if (
            not load_vist
            and ("det_embedding_means" in loaded_state_dict.keys())
            and self.enable_det_contrast
        ):
            ## load register buffer
            self.det_embedding_means[: self.num_old_cat, ...] = loaded_state_dict[
                "det_embedding_means"
            ][: self.num_old_cat, ...]
            self.det_embedding_means[-1, ...] = loaded_state_dict[
                "det_embedding_means"
            ][-1, ...]
            self.det_embedding_stds[: self.num_old_cat, ...] = loaded_state_dict[
                "det_embedding_stds"
            ][: self.num_old_cat, ...]
            self.det_embedding_stds[-1, ...] = loaded_state_dict[
                "det_embedding_stds"
            ][-1, ...]
            self.det_memory_embedding[: self.num_old_cat, ...] = loaded_state_dict[
                "det_memory_embedding"
            ][: self.num_old_cat, ...]
            self.det_memory_embedding[-1, ...] = loaded_state_dict[
                "det_memory_embedding"
            ][-1, ...]
            self.det_memory_pointer[: self.num_old_cat] = loaded_state_dict[
                "det_memory_pointer"
            ][: self.num_old_cat]
            self.det_memory_pointer[-1] = loaded_state_dict["det_memory_pointer"][
                -1
            ]
            self.det_memory_num[: self.num_old_cat] = loaded_state_dict[
                "det_memory_num"
            ][: self.num_old_cat]
            self.det_memory_num[-1] = loaded_state_dict["det_memory_num"][-1]

        if (
            not load_vist
            and ("track_embedding_means" in loaded_state_dict.keys())
            and self.enable_track_contrast
        ):
            ## load register buffer
            self.track_embedding_means[: self.num_old_cat, ...] = loaded_state_dict[
                "track_embedding_means"
            ][: self.num_old_cat, ...]
            self.track_embedding_means[-1, ...] = loaded_state_dict[
                "track_embedding_means"
            ][-1, ...]
            self.track_embedding_stds[: self.num_old_cat, ...] = loaded_state_dict[
                "track_embedding_stds"
            ][: self.num_old_cat, ...]
            self.track_embedding_stds[-1, ...] = loaded_state_dict[
                "track_embedding_stds"
            ][-1, ...]
            self.track_memory_embedding[: self.num_old_cat, ...] = loaded_state_dict[
                "track_memory_embedding"
            ][: self.num_old_cat, ...]
            self.track_memory_embedding[-1, ...] = loaded_state_dict[
                "track_memory_embedding"
            ][-1, ...]
            self.track_memory_pointer[: self.num_old_cat] = loaded_state_dict[
                "track_memory_pointer"
            ][: self.num_old_cat]
            self.track_memory_pointer[-1] = loaded_state_dict[
                "track_memory_pointer"
            ][-1]
            self.track_memory_num[: self.num_old_cat] = loaded_state_dict[
                "track_memory_num"
            ][: self.num_old_cat]
            self.track_memory_num[-1] = loaded_state_dict["track_memory_num"][-1]

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

        # track head
        track_losses, _ = self.similarity_head.forward_train(
            [key_inputs, *ref_inputs],
            [key_proposals, *ref_proposals],
            [key_x, *ref_x],
            [key_targets, *ref_targets],
        )

        # contrast head

        sampled_proposals = self.detector.sample_proposals_contrast(
            key_inputs.targets.boxes2d, key_proposals, use_bg=self.use_bg
        )

        sample_class_ids = [p.class_ids for p in sampled_proposals]
        all_class_ids = torch.cat(sample_class_ids, dim=0)

        ct_losses = {}

        if self.enable_det_contrast:
            det_embeddings = []
            for proposal in sampled_proposals:
                det_embedding = self.detector.get_proposals_embeddings(
                    key_x, [proposal]
                )
                det_embeddings.append(det_embedding)

            self.detector.eval()
            with torch.no_grad():
                det_embeddings_contrast = []
                for proposal in sampled_proposals:
                    det_embedding = self.detector.get_proposals_embeddings(
                        key_x, [proposal]
                    )
                    det_embeddings_contrast.append(det_embedding)
            self.detector.train()
            # update_memory

            self.update_det_memory(det_embeddings_contrast, sample_class_ids)
            exclude_classes = self.update_det_statistics()

            class_mask = torch.ones(
                [len(self.category_mapping) + 1],
                dtype=torch.bool,
                device=self.device,
            )

            for class_id in exclude_classes:
                class_mask[class_id] = 0
            ### log mean statistics:

            for i in range(self.det_embedding_means.shape[0] - 1):
                for j in range(i + 1, self.det_embedding_means.shape[0]):
                    cat_mean_1 = self.det_embedding_means[i, :]
                    cat_mean_2 = self.det_embedding_means[j, :]
                    cat_name_1 = self.cat_mapping[i]
                    cat_name_2 = (
                        self.cat_mapping[j]
                        if j < self.det_embedding_means.shape[0] - 1
                        else "background"
                    )
                    self.log(
                        f"train/det_embedding_distance_{cat_name_1}_{cat_name_2}",
                        torch.linalg.norm(cat_mean_1 - cat_mean_2),
                    )

            det_all_embeddings = torch.cat(det_embeddings, dim=0)

            if len(exclude_classes) > 0:
                all_mask = reduce(
                    torch.logical_and,
                    [all_class_ids != cat for cat in exclude_classes],
                )

                det_all_embeddings = det_all_embeddings[all_mask]
                det_all_class_ids = all_class_ids[all_mask]
            else:
                det_all_class_ids = all_class_ids

            # make detection losses neglegible
            # for key in det_losses.keys():
            #     det_losses[key] = det_losses[key] * 0.00001

            # require at least two classes to perform pushing loss
            if (
                det_all_embeddings.shape[0] > 0
                and self.det_embedding_means[class_mask].shape[0] >= 2
            ):
                loss_pull, loss_push = self.det_contrast_loss(
                    det_all_embeddings,
                    det_all_class_ids,
                    self.det_embedding_means,
                    self.det_embedding_stds,
                    class_mask,
                )
                ct_losses.update(
                    {
                        "loss_pull_det": loss_pull,
                        "loss_push_det": loss_push,
                    }
                )

        if self.enable_track_contrast:

            track_embeddings = self.similarity_forward_fc(
                key_x, sampled_proposals
            )

            self.similarity_head.eval()
            with torch.no_grad():
                track_embeddings_contrast = self.similarity_forward_fc(
                    key_x, sampled_proposals
                )
            self.similarity_head.train()
            # update_memory

            self.update_track_memory(
                track_embeddings_contrast, sample_class_ids
            )
            exclude_classes = self.update_track_statistics()

            class_mask = torch.ones(
                [len(self.category_mapping) + 1],
                dtype=torch.bool,
                device=self.device,
            )

            for class_id in exclude_classes:
                class_mask[class_id] = 0
            ### log mean statistics:

            for i in range(self.track_embedding_means.shape[0] - 1):
                for j in range(i + 1, self.track_embedding_means.shape[0]):
                    cat_mean_1 = self.track_embedding_means[i, :]
                    cat_mean_2 = self.track_embedding_means[j, :]
                    cat_name_1 = self.cat_mapping[i]
                    cat_name_2 = (
                        self.cat_mapping[j]
                        if j < self.track_embedding_means.shape[0] - 1
                        else "background"
                    )
                    self.log(
                        f"train/track_embedding_distance_{cat_name_1}_{cat_name_2}",
                        torch.linalg.norm(cat_mean_1 - cat_mean_2),
                    )

            track_all_embeddings = torch.cat(track_embeddings, dim=0)

            if len(exclude_classes) > 0:
                all_mask = reduce(
                    torch.logical_and,
                    [all_class_ids != cat for cat in exclude_classes],
                )

                track_all_embeddings = track_all_embeddings[all_mask]
                track_all_class_ids = all_class_ids[all_mask]
            else:
                track_all_class_ids = all_class_ids

            # make detection losses neglegible
            # for key in det_losses.keys():
            #     det_losses[key] = det_losses[key] * 0.00001

            # require at least two classes to perform pushing loss
            if (
                track_all_embeddings.shape[0] > 0
                and self.track_embedding_means[class_mask].shape[0] >= 2
            ):
                loss_pull, loss_push = self.track_contrast_loss(
                    track_all_embeddings,
                    track_all_class_ids,
                    self.track_embedding_means,
                    self.track_embedding_stds,
                    class_mask,
                )
                ct_losses.update(
                    {
                        "loss_pull_track": loss_pull,
                        "loss_push_track": loss_push,
                    }
                )

        if ct_losses:
            return {**ct_losses, **det_losses, **track_losses}
        return {**det_losses, **track_losses}

    def similarity_forward_fc(
        self, features, boxes
    ) -> List[torch.Tensor]:
        """Similarity head forward pass."""
        assert self.similarity_head.in_features is not None
        features_list = [features[f] for f in self.similarity_head.in_features]
        x = self.similarity_head.roi_pooler(features_list, boxes)

        # convs
        if self.similarity_head.num_convs > 0:
            for conv in self.similarity_head.convs:
                x = conv(x)

        # fcs
        x = torch.flatten(x, start_dim=1)
        if self.similarity_head.num_fcs > 0:
            for fc in self.similarity_head.fcs:
                x = fc(x)

        embeddings: List[torch.Tensor] = x.split(
            [len(b) for b in boxes]
        )
        return embeddings

    def update_det_memory(self, embeddings, sample_class_ids):
        """Update the embeddings memory bank."""
        # only use pos proposals for contrastive learning
        for embedding, class_ids in zip(embeddings, sample_class_ids):
            for cat_id in range(-1, len(self.category_mapping)):
                embedding_cat = embedding[class_ids == cat_id, :]
                embedding_num = embedding_cat.shape[0]

                if embedding_num > self.sample_num_per_image:
                    perm = torch.randperm(embedding_num)  # type: ignore
                    embedding_cat = embedding_cat[
                        perm[: self.sample_num_per_image], :
                    ]
                    embedding_num = self.sample_num_per_image

                self.det_memory_num[cat_id] += embedding_num

                if (
                    self.det_memory_pointer[cat_id] + embedding_num
                    <= self.memory_per_class
                ):
                    self.det_memory_embedding[
                        cat_id,
                        self.det_memory_pointer[
                            cat_id
                        ] : self.det_memory_pointer[cat_id]
                        + embedding_num,
                        :,
                    ] = embedding_cat
                    self.det_memory_pointer[cat_id] = (
                        self.det_memory_pointer[cat_id] + embedding_num
                    )
                else:
                    exceed_num = (
                        self.det_memory_pointer[cat_id]
                        + embedding_num
                        - self.memory_per_class
                    )
                    self.det_memory_embedding[
                        cat_id, self.det_memory_pointer[cat_id] :, :
                    ] = embedding_cat[:-exceed_num, :]

                    self.det_memory_embedding[
                        cat_id, :exceed_num, :
                    ] = embedding_cat[-exceed_num:, :]
                    self.det_memory_pointer[cat_id] = exceed_num

    def update_det_statistics(self):
        """Update class means and stds."""
        exclude_classes = []
        # ignore the new class in the first epoch
        if self.current_epoch == 0 and self.skip_first:
            if self.skip_only_new:
                for cat_id in range(
                    len(self.category_mapping) - self.num_new_cat,
                    len(self.category_mapping),
                ):
                    exclude_classes.append(cat_id)
            else:
                for cat_id in range(-1, len(self.category_mapping)):
                    exclude_classes.append(cat_id)

        for cat_id in range(-1, len(self.category_mapping)):
            # ignore class if less than 100 embeddings
            if self.det_memory_num[cat_id] < self.memory_num_for_contrast:
                if not cat_id in exclude_classes:
                    exclude_classes.append(cat_id)
                continue
            if self.det_memory_num[cat_id] >= self.memory_per_class:
                current_class_mean = torch.mean(
                    self.det_memory_embedding[cat_id, :, :], dim=0
                )
                current_class_std = torch.sqrt(
                    torch.mean(
                        (
                            self.det_memory_embedding[cat_id, :, :]
                            - self.det_embedding_means[cat_id, :]
                        )
                        ** 2,
                        dim=0,
                    )
                )
            else:
                current_class_mean = torch.mean(
                    self.det_memory_embedding[
                        cat_id, : self.det_memory_num[cat_id], :
                    ],
                    dim=0,
                )

                current_class_std = torch.sqrt(
                    torch.mean(
                        (
                            self.det_memory_embedding[cat_id, :, :]
                            - self.det_embedding_means[cat_id, :]
                        )
                        ** 2,
                        dim=0,
                    )
                )

            if torch.sum(self.det_embedding_means[cat_id, :]) == 0:
                self.det_embedding_means[cat_id, :] = current_class_mean
                self.det_embedding_stds[cat_id, :] = current_class_std
            else:
                self.det_embedding_means[cat_id, :] = (
                    self.det_embedding_means[cat_id, :] * self.polyak_factor
                    + (1 - self.polyak_factor) * current_class_mean
                )
                self.det_embedding_stds[cat_id, :] = (
                    self.det_embedding_stds[cat_id, :] * self.polyak_factor
                    + (1 - self.polyak_factor) * current_class_std
                )
            # log variance
            std = torch.linalg.norm(self.det_embedding_stds[cat_id, :])
            cat_name = (
                self.cat_mapping[cat_id] if cat_id >= 0 else "background"
            )
            self.log(f"train/det_embedding_std_{cat_name}", std)

        return exclude_classes

    def update_track_memory(self, embeddings, sample_class_ids):
        """Update the embeddings memory bank."""
        # only use pos proposals for contrastive learning
        for embedding, class_ids in zip(embeddings, sample_class_ids):
            for cat_id in range(-1, len(self.category_mapping)):
                embedding_cat = embedding[class_ids == cat_id, :]
                embedding_num = embedding_cat.shape[0]

                if embedding_num > self.sample_num_per_image:
                    perm = torch.randperm(embedding_num)  # type: ignore
                    embedding_cat = embedding_cat[
                        perm[: self.sample_num_per_image], :
                    ]
                    embedding_num = self.sample_num_per_image

                self.track_memory_num[cat_id] += embedding_num

                if (
                    self.track_memory_pointer[cat_id] + embedding_num
                    <= self.memory_per_class
                ):
                    self.track_memory_embedding[
                        cat_id,
                        self.track_memory_pointer[
                            cat_id
                        ] : self.track_memory_pointer[cat_id]
                        + embedding_num,
                        :,
                    ] = embedding_cat
                    self.track_memory_pointer[cat_id] = (
                        self.track_memory_pointer[cat_id] + embedding_num
                    )
                else:
                    exceed_num = (
                        self.track_memory_pointer[cat_id]
                        + embedding_num
                        - self.memory_per_class
                    )
                    self.track_memory_embedding[
                        cat_id, self.track_memory_pointer[cat_id] :, :
                    ] = embedding_cat[:-exceed_num, :]

                    self.track_memory_embedding[
                        cat_id, :exceed_num, :
                    ] = embedding_cat[-exceed_num:, :]
                    self.track_memory_pointer[cat_id] = exceed_num

    def update_track_statistics(self):
        """Update class means and stds."""
        exclude_classes = []
        # ignore the new class in the first epoch
        if self.current_epoch == 0 and self.skip_first:
            if self.skip_only_new:
                for cat_id in range(
                    len(self.category_mapping) - self.num_new_cat,
                    len(self.category_mapping),
                ):
                    exclude_classes.append(cat_id)
            else:
                for cat_id in range(-1, len(self.category_mapping)):
                    exclude_classes.append(cat_id)

        for cat_id in range(-1, len(self.category_mapping)):
            # ignore class if less than 100 embeddings
            if self.track_memory_num[cat_id] < self.memory_num_for_contrast:
                if not cat_id in exclude_classes:
                    exclude_classes.append(cat_id)
                continue
            if self.track_memory_num[cat_id] >= self.memory_per_class:
                current_class_mean = torch.mean(
                    self.track_memory_embedding[cat_id, :, :], dim=0
                )
                current_class_std = torch.sqrt(
                    torch.mean(
                        (
                            self.track_memory_embedding[cat_id, :, :]
                            - self.track_embedding_means[cat_id, :]
                        )
                        ** 2,
                        dim=0,
                    )
                )
            else:
                current_class_mean = torch.mean(
                    self.track_memory_embedding[
                        cat_id, : self.track_memory_num[cat_id], :
                    ],
                    dim=0,
                )

                current_class_std = torch.sqrt(
                    torch.mean(
                        (
                            self.track_memory_embedding[cat_id, :, :]
                            - self.track_embedding_means[cat_id, :]
                        )
                        ** 2,
                        dim=0,
                    )
                )

            if torch.sum(self.track_embedding_means[cat_id, :]) == 0:
                self.track_embedding_means[cat_id, :] = current_class_mean
                self.track_embedding_stds[cat_id, :] = current_class_std
            else:
                self.track_embedding_means[cat_id, :] = (
                    self.track_embedding_means[cat_id, :] * self.polyak_factor
                    + (1 - self.polyak_factor) * current_class_mean
                )
                self.track_embedding_stds[cat_id, :] = (
                    self.track_embedding_stds[cat_id, :] * self.polyak_factor
                    + (1 - self.polyak_factor) * current_class_std
                )
            # log variance
            std = torch.linalg.norm(self.track_embedding_stds[cat_id, :])
            cat_name = (
                self.cat_mapping[cat_id] if cat_id >= 0 else "background"
            )
            self.log(f"train/track_embedding_std_{cat_name}", std)

        return exclude_classes

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
                self.detector.clip_bboxes_to_image,
                self.detector.resolve_overlap,
            )
            det_outputs = predictions_to_scalabel(det_outs, self.cat_mapping)

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

        # detector
        feat = self.detector.extract_features(inputs)
        detections = inputs.targets.boxes2d
        assert detections is not None

        # from vist.vis.image import imshow_bboxes
        # imshow_bboxes(inputs.images.tensor[0], detections)

        # similarity head
        embeddings = self.similarity_head(inputs, detections, feat)
        embeddings = embeddings[0]
        class_ids = inputs.targets.boxes2d[0].class_ids
        track_ids = inputs.targets.boxes2d[0].track_ids

        if add_background:
            proposals = self.detector.generate_proposals(inputs, feat)
            ious = bbox_iou(proposals[0], detections[0])
            max_iou, _ = torch.max(ious, dim=1)
            mask = max_iou < 0.3
            proposals[0] = proposals[0]
            embeddings_bg = self.similarity_head(inputs, proposals, feat)
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
