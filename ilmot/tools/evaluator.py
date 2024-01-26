"""Evaluation callback modified from vist to support more evaluations."""
import logging
import itertools
import os
import copy
from functools import reduce
from operator import add
from collections import defaultdict
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import (
    rank_zero_info,
)
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
from scalabel.common import mute
from scalabel.label.io import save, group_and_sort
from scalabel.label.typing import Category, Frame
from ilmot.data.datasets import BaseDatasetLoader
from ilmot.data.datasets.base import _eval_mapping
from ilmot.struct import InputSample, ModelOutput, MetricLogs
from ilmot.common.utils.distributed import (
    all_gather_object_cpu,
    all_gather_object_gpu,
)

from ilmot.engine.evaluator import (  # type: ignore
    StandardEvaluatorCallback,
)

from .hota import evaluate_track_hota

mute(True)  # turn off undesired logs during eval
logger = logging.getLogger("pytorch_lightning")


class ILMOTEvaluatorCallback(StandardEvaluatorCallback):  # type: ignore
    """Evaluate model using metrics supported by Scalabel.

    Args:
        dataset_loader: Dataset used to load Scalabel format data.
        categoty_mapping: A mapping from category names to their ids.
        output_dir: Directory to store (temporary) evaluation results.
        test_category: Categories that are kept in calculating average and
            overall results.
        compute_statistics: Whether to update threshold for the track graph.

    """

    def __init__(
        self,
        dataloader_idx: int,
        dataset_loader: BaseDatasetLoader,
        output_dir: Optional[str],
        category_mapping: Dict[str, int],
        train_category: Optional[List[str]] = None,
        test_category: Optional[List[str]] = None,
        compute_statistics: bool = False,
        vis_dir: Optional[str] = None,
    ) -> None:
        """Init."""
        self.test_category = test_category
        self.train_category = train_category
        self.category_mapping = category_mapping
        self.compute_statistics = compute_statistics
        self.vis_dir = vis_dir
        self._gt_scores = []  # type: ignore
        self._gt_labels = []  # type: ignore
        self.cats_id2name = {v: k for k, v in category_mapping.items()}
        super().__init__(dataloader_idx, dataset_loader, output_dir)
        if self.test_category is None:
            self.test_category = list(self.category_mapping.keys())
        if self.train_category is None:
            self.train_category = list(self.category_mapping.keys())

        dataset_loader.metadata_cfg.categories = []
        new_class = []
        for cat in self.train_category:
            new_class.append(Category(name=cat, subcategories=None))

        if len(new_class) > 0:
            dataset_loader.metadata_cfg.categories.append(
                Category(name="new_class", subcategories=new_class)
            )
        old_class = []
        for i in range(len(self.test_category) - len(self.train_category)):
            old_class.append(
                Category(name=self.test_category[i], subcategories=None)
            )
        if len(old_class) > 0:
            dataset_loader.metadata_cfg.categories.append(
                Category(name="old_class", subcategories=old_class)
            )

        self.all_keys = copy.deepcopy(self.test_category)
        self.all_keys.append("new_class")
        self.all_keys.append("old_class")
        self.dataset_loader = dataset_loader

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = defaultdict(list)  # type: ignore
        self._gts = []  # type: ignore
        self._gt_scores = []
        self._gt_labels = []

    def all_gather_stats(  # pylint: disable=no-self-use
        self,
        stats: List[Any],  # type: ignore
        pl_module: pl.LightningModule,
        collect_device: str,
    ) -> Optional[List[Any]]:  # type: ignore
        """Gather prediction dict in distributed setting."""
        if collect_device == "gpu":
            stats_list = all_gather_object_gpu(stats, pl_module)
        elif collect_device == "cpu":
            stats_list = all_gather_object_cpu(stats, pl_module)
        else:
            raise ValueError(f"Collect device {collect_device} unknown.")

        if stats_list is None:
            return None

        result = reduce(add, stats_list)

        return result

    def all_gather_predictions(
        self,
        predictions: Dict[str, List[Frame]],
        pl_module: pl.LightningModule,
        collect_device: str,
    ) -> Optional[Dict[str, List[Frame]]]:  # pragma: no cover
        """Gather prediction dict in distributed setting."""
        if collect_device == "gpu":
            predictions_list = all_gather_object_gpu(
                predictions, pl_module, rank_zero_only=False
            )
        elif collect_device == "cpu":
            predictions_list = all_gather_object_cpu(
                predictions, pl_module, rank_zero_only=False
            )
        else:
            raise ValueError(f"Collect device {collect_device} unknown.")

        if predictions_list is None:
            return None

        result = {}
        for key in predictions:
            prediction_list = [p[key] for p in predictions_list]
            result[key] = list(itertools.chain(*prediction_list))
        return result

    def all_gather_gts(
        self,
        gts: List[Frame],
        pl_module: pl.LightningModule,
        collect_device: str,
    ) -> Optional[List[Frame]]:  # pragma: no cover
        """Gather gts list in distributed setting."""
        if collect_device == "gpu":
            gts_list = all_gather_object_gpu(
                gts, pl_module, rank_zero_only=False
            )
        elif collect_device == "cpu":
            gts_list = all_gather_object_cpu(
                gts, pl_module, rank_zero_only=False
            )
        else:
            raise ValueError(f"Collect device {collect_device} unknown.")

        if gts_list is None:
            return None

        return list(itertools.chain(*gts_list))

    def gather(self, pl_module: pl.LightningModule) -> None:
        """Gather accumulated data."""
        preds = self.all_gather_predictions(
            self._predictions, pl_module, self.collect
        )
        if preds is not None:
            self._predictions = preds  # type: ignore
        gts = self.all_gather_gts(self._gts, pl_module, self.collect)
        if gts is not None:
            self._gts = gts

        # gt_scores = self.all_gather_stats(
        #     self._gt_scores, pl_module, self.collect
        # )
        # if gt_scores is not None:
        #     self._gt_scores = gt_scores  # type: ignore

        # gt_labels = self.all_gather_stats(
        #     self._gt_labels, pl_module, self.collect
        # )
        # if gt_labels is not None:
        #     self._gt_labels = gt_labels  # type: ignore

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Wait for on_test_epoch_end PL hook to call 'evaluate'.

        Args:
            trainer: Pytorch-lightning trainer object.
            pl_module: Pytorch-lightning module used for running evaluation.

        """
        self.gather(pl_module)
        rank_zero_info("finish gatherining, prepare for evaluation")
        self.evaluate(trainer.current_epoch)
        if self.compute_statistics and trainer.is_global_zero:
            self.statistics()
        self.reset()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Wait for on_validation_epoch_end PL hook to call 'evaluate'.

        Args:
            trainer: Pytorch-lightning trainer object.
            pl_module: Pytorch-lightning module used for running evaluation.

        """
        self.gather(pl_module)
        rank_zero_info("finish gatherining, prepare for evaluation")
        self.evaluate(trainer.current_epoch)
        if self.compute_statistics and trainer.is_global_zero:
            self.statistics()
        self.reset()

    def statistics(self):
        """Compute the statisics and plot them as graphs."""
        assert isinstance(self.test_category, List)
        detect_preds = self._predictions["track"]
        full_category = copy.deepcopy(self.test_category)
        full_category.append("background")
        full_category_dict = {v: i for i, v in enumerate(full_category)}

        full_category_pred = copy.deepcopy(list(self.cats_id2name.values()))
        full_category_pred.append("background")

        pred_matched_dict: Dict[str, Dict[str, List[float]]] = {
            cat: {cat2: [] for cat2 in full_category}
            for cat in self.cats_id2name.values()
        }
        gt_matched_dict: Dict[str, Dict[str, List[float]]] = {
            cat: {cat2: [] for cat2 in full_category_pred}
            for cat in self.test_category
        }

        confusion_matrix_gt = np.zeros(
            [len(self.test_category), len(full_category_pred)],
            dtype=np.double,
        )

        confusion_matrix_pl = np.zeros(
            [len(self.cats_id2name), len(full_category)], dtype=np.double
        )

        for gt_score, gt_label in zip(self._gt_scores, self._gt_labels):
            for i, label in enumerate(gt_label):
                cat = full_category[label]
                for idx, cat2 in enumerate(full_category_pred):
                    gt_matched_dict[cat][cat2].append(gt_score[i, idx])
                    confusion_matrix_gt[label, idx] += gt_score[i, idx]

        for pred, gt in zip(detect_preds, self._gts):
            bbox_dict_pred: Dict[str, List[float]] = {
                cat: [] for cat in self.test_category
            }
            conf_dict_pred: Dict[str, List[float]] = {
                cat: [] for cat in self.test_category
            }
            bbox_gt: List[float] = []
            label_gt: List[float] = []

            for x in pred.labels:
                bbox_dict_pred[x.category].append(x.box2d.x1)
                bbox_dict_pred[x.category].append(x.box2d.y1)
                bbox_dict_pred[x.category].append(x.box2d.x2)
                bbox_dict_pred[x.category].append(x.box2d.y2)
                conf_dict_pred[x.category].append(x.score)
            for x in gt.labels:
                bbox_gt.append(x.box2d.x1)
                bbox_gt.append(x.box2d.y1)
                bbox_gt.append(x.box2d.x2)
                bbox_gt.append(x.box2d.y2)
                label_gt.append(x.category)
            if len(bbox_gt) > 0:
                for id, pred_cat in self.cats_id2name.items():
                    if len(bbox_dict_pred[pred_cat]) > 0:
                        bbox_pred = torch.Tensor(
                            bbox_dict_pred[pred_cat]
                        ).reshape(-1, 4)
                        bbox_gt = torch.Tensor(bbox_gt).reshape(-1, 4)
                        iou_calc = BboxOverlaps2D()
                        iou = iou_calc(bbox_pred, bbox_gt)
                        max_iou, max_id = torch.max(iou, dim=-1)
                        for i in range(max_iou.shape[0]):
                            if max_iou[i] > 0.5:
                                gt_cat = label_gt[max_id[i]]
                                pred_matched_dict[pred_cat][gt_cat].append(
                                    conf_dict_pred[pred_cat][i]
                                )
                                # if conf_dict_pred[pred_cat][i] > 0.7:
                                confusion_matrix_pl[
                                    id, full_category_dict[gt_cat]
                                ] += 1
                            else:
                                pred_matched_dict[pred_cat][
                                    "background"
                                ].append(conf_dict_pred[pred_cat][i])
                                # if conf_dict_pred[pred_cat][i] > 0.7:
                                confusion_matrix_pl[id, -1] += 1
                            # if this is a pseudo label

            else:
                for id, pred_cat in self.cats_id2name.items():
                    for i in range(len(conf_dict_pred[pred_cat])):
                        pred_matched_dict[pred_cat]["background"].append(
                            conf_dict_pred[pred_cat][i]
                        )
                        # if conf_dict_pred[pred_cat][i] > 0.7:
                        confusion_matrix_pl[id, -1] += 1

        for cat in self.cats_id2name.values():
            plt.figure()
            bottom = np.zeros(10)
            for gt_cat in full_category:
                fig = plt.hist(
                    np.array(pred_matched_dict[cat][gt_cat]),
                    bins=10,
                    range=(0.0, 1.0),
                    label=gt_cat,
                    bottom=bottom,
                )
                bottom += np.array(fig[0])

            plt.legend(loc="upper right")
            plt.xlabel("confidence score")
            plt.ylabel("number of instances")
            plt.title(f"confidence score and gt label for all predicted {cat}")
            if self.vis_dir is None:
                plt.savefig(f"{cat}_pred_conf.png")
            else:
                plt.savefig(f"{self.vis_dir}/{cat}_pred_conf.png")

        for cat in self.test_category:
            plt.figure()
            bottom = np.zeros(10)
            for pred_cat in full_category_pred:
                fig = plt.hist(
                    np.array(gt_matched_dict[cat][pred_cat]),
                    bins=10,
                    range=(0.0, 1.0),
                    label=pred_cat,
                    bottom=bottom,
                )
                bottom += np.array(fig[0])

            plt.legend(loc="upper right")
            plt.xlabel("confidence score")
            plt.ylabel("number of instances")
            plt.title(f"confidence distribution of classes for all gt {cat}")
            if self.vis_dir is None:
                plt.savefig(f"{cat}_gt_conf.png")
            else:
                plt.savefig(f"{self.vis_dir}/{cat}_gt_conf.png")

        plt.figure(figsize=(10, 10))
        group_counts = [
            "{0:0.0f}".format(value) for value in confusion_matrix_gt.flatten()
        ]
        confusion_matrix_gt = confusion_matrix_gt / np.sum(
            confusion_matrix_gt, axis=-1, keepdims=True
        )

        group_percentages = [
            "{0:.2%}".format(value) for value in confusion_matrix_gt.flatten()
        ]

        labels = [
            f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)
        ]

        labels = np.asarray(labels).reshape(len(self.test_category), -1)

        ax = sns.heatmap(
            confusion_matrix_gt, annot=labels, fmt="", cmap="Blues"
        )

        ax.set_title("Confusion Matrix with GT Labels\n\n")
        ax.set_xlabel("\nPredicted Classes")
        ax.set_ylabel("GT Classes")

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(full_category_pred)
        ax.yaxis.set_ticklabels(self.test_category)

        ## Display the visualization of the Confusion Matrix.
        if self.vis_dir is None:
            plt.savefig("confusion_matrix_gt.png")
        else:
            plt.savefig(
                f"{self.vis_dir}/confusion_matrix_gt.png", bbox_inches="tight"
            )

        plt.figure(figsize=(10, 10))
        group_counts = [
            "{0:0.0f}".format(value) for value in confusion_matrix_pl.flatten()
        ]
        confusion_matrix_pl = confusion_matrix_pl / np.sum(
            confusion_matrix_pl, axis=-1, keepdims=True
        )

        group_percentages = [
            "{0:.2%}".format(value) for value in confusion_matrix_pl.flatten()
        ]

        labels = [
            f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)
        ]

        labels = np.asarray(labels).reshape(len(self.cats_id2name), -1)

        ax = sns.heatmap(
            confusion_matrix_pl, annot=labels, fmt="", cmap="Blues"
        )

        ax.set_title("Confusion Matrix of Tracking Result\n\n")
        ax.set_xlabel("GT Classes")
        ax.set_ylabel("Predicted Classes")

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(full_category)
        ax.yaxis.set_ticklabels((self.cats_id2name.values()))

        ## Display the visualization of the Confusion Matrix.
        if self.vis_dir is None:
            plt.savefig("confusion_matrix_pl.png")
        else:
            plt.savefig(
                f"{self.vis_dir}/confusion_matrix_pl.png", bbox_inches="tight"
            )

    def process(
        self, inputs: List[List[InputSample]], outputs: ModelOutput
    ) -> None:
        """Process the pair of inputs and outputs."""
        for inp in inputs:
            self._gts.append(copy.deepcopy(inp[0].metadata[0]))

        for key, output in outputs.items():
            if not key == "scores":
                for inp, out in zip(inputs, output):
                    prediction = copy.deepcopy(inp[0].metadata[0])
                    prediction.labels = out
                    self._predictions[key].append(prediction)
            else:
                self._gt_scores.append(
                    output[0].to(torch.device("cpu")).detach().numpy()
                )
                self._gt_labels.append(
                    output[1].to(torch.device("cpu")).detach().numpy()
                )

    def evaluate(self, epoch: int) -> Dict[str, MetricLogs]:
        """Evaluate the performance after processing all input/output pairs.

        Args:
            epoch: Number of epochs(steps) in training when running evaluation.

        Returns:
            A mapping from each evaluation process to its results.

        """
        results = {}
        if not self.logging_disabled:
            rank_zero_info("Running evaluation on dataset %s...", self.name)
        # pylint: disable=too-many-nested-blocks
        for key, predictions in self._predictions.items():
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                file_path = os.path.join(
                    self.output_dir, f"{key}_predictions.json"
                )
                save(file_path, predictions)

            if key in self.metrics:
                full_results = _eval_mapping[key](
                    predictions,
                    self._gts,
                    self.dataset_loader.metadata_cfg,
                    self.dataset_loader.ignore_unknown_cats,
                )
                log_dict = {
                    f"{key}/{k}": v for k, v in full_results.summary().items()
                }
                log_str = str(full_results)
                results[key] = log_dict
                if not self.logging_disabled:
                    for k, v in log_dict.items():
                        self.log(k, v, rank_zero_only=True)  # type: ignore # pylint: disable=no-member,line-too-long

                    for metric, scores_list in full_results.dict().items():
                        if isinstance(scores_list, list):
                            for score in scores_list:
                                scores_list[0].update(score)
                            for cat in self.all_keys:
                                self.log(  # pylint: disable=no-member
                                    f"{key}/{metric}_{cat}",
                                    scores_list[0].get(cat, 0),
                                    rank_zero_only=True,
                                )

                    rank_zero_info(
                        "Showing results for epoch=%d: %s", epoch, key
                    )
                    rank_zero_info(log_str)

        ### also evaluate hota

        hota_results = evaluate_track_hota(
            group_and_sort(self._gts),
            group_and_sort(self._predictions["track"]),
            self.dataset_loader.metadata_cfg,
            nproc=1,
        )

        log_dict = {f"hota/{k}": v for k, v in hota_results.summary().items()}
        log_str = str(hota_results)
        results["hota"] = log_dict
        if not self.logging_disabled:
            for k, v in log_dict.items():
                self.log(k, v, rank_zero_only=True)  # type: ignore # pylint: disable=no-member,line-too-long

            for metric, scores_list in hota_results.dict().items():
                if isinstance(scores_list, list):
                    for score in scores_list:
                        scores_list[0].update(score)
                    for cat in self.all_keys:
                        self.log(  # pylint: disable=no-member
                            f"hota/{metric}_{cat}",
                            scores_list[0].get(cat, 0),
                            rank_zero_only=True,
                        )

            rank_zero_info("Showing results for epoch=%d: %s", epoch, "hota")
            rank_zero_info(log_str)

        return results
