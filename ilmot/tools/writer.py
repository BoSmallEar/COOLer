"""Visualizer class."""
import os
from typing import Any, Sequence, List, Dict, Optional
from collections import defaultdict
from functools import reduce
from operator import add
import pickle
import copy

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import (
    rank_zero_warn,
    rank_zero_info,
)
import numpy as np
from scalabel.label.io import save
from scalabel.label.typing import FrameGroup
from scalabel.vis.label import LabelViewer, UIConfig
from ilmot.engine.writer import ScalabelWriterCallback
from ilmot.engine.utils import all_gather_predictions
from ilmot.struct import InputSample, ModelOutput
from ilmot.vis.utils import preprocess_image
from ilmot.common.utils.distributed import (
    all_gather_object_cpu,
    all_gather_object_gpu,
)


class ILMOTWriterCallback(ScalabelWriterCallback):  # type: ignore
    """Run model and visualize & save output."""

    def __init__(
        self,
        dataloader_idx: int,
        output_dir: str,
        visualize: bool = True,
        category_mapping: Dict[str, int] = None,
        train_category: Optional[List[str]] = None,
        test_category: Optional[List[str]] = None,
    ) -> None:
        """Init."""
        assert train_category is not None
        assert test_category is not None
        assert category_mapping is not None
        self.train_category = train_category
        self.test_category = test_category
        self.category_mapping = category_mapping
        self.num_new_cat = len(train_category)
        self.num_old_cat = len(category_mapping) - len(train_category)
        self._stats = {
            "num_tp": np.zeros(self.num_old_cat),
            "num_fp": np.zeros(self.num_old_cat),
            "num_fn": np.zeros(self.num_old_cat),
        }

        super().__init__(dataloader_idx, output_dir, visualize)

    def gather(self, pl_module: pl.LightningModule) -> None:
        """Gather accumulated data."""
        preds = all_gather_predictions(self._predictions, pl_module, "cpu")  # type: ignore #pylint: disable=line-too-long
        # stats = self.all_gather_stats(self._stats, pl_module, "cpu")
        if preds is not None:
            self._predictions = preds
        # if stats is not None:
        #     self._stats = stats

    def on_predict_epoch_end(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Sequence[Any],
    ) -> None:
        """Hook for on_predict_epoch_end."""
        self.gather(pl_module)
        if trainer.is_global_zero:
            self.write()
        self.reset()

    def reset(self) -> None:
        """Preparation for a new round of evaluation."""
        self._predictions = defaultdict(list)
        self._stats = {
            "num_tp": np.zeros(self.num_old_cat),
            "num_fp": np.zeros(self.num_old_cat),
            "num_fn": np.zeros(self.num_old_cat),
        }

    def all_gather_stats(
        self,
        stats: Dict[str, np.ndarray],  # type: ignore
        pl_module: pl.LightningModule,
        collect_device: str,
    ) -> Optional[Dict[str, np.ndarray]]:  # type: ignore
        """Gather prediction dict in distributed setting."""
        if collect_device == "gpu":
            stats_list = all_gather_object_gpu(stats, pl_module)
        elif collect_device == "cpu":
            stats_list = all_gather_object_cpu(stats, pl_module)
        else:
            raise ValueError(f"Collect device {collect_device} unknown.")

        if stats_list is None:
            return None

        result = {}
        for key in stats:
            stat_list = [p[key] for p in stats_list]
            result[key] = reduce(add, stat_list)
        return result

    def write(self) -> None:
        """Write the aggregated output."""
        # rank_zero_info("Display the statistics of pseudo labels!!!")
        # for i in range(self.num_old_cat):
        #     tps = self._stats["num_tp"][i]
        #     fps = self._stats["num_fp"][i]
        #     fns = self._stats["num_fn"][i]
        #     cat = self.test_category[i]
        #     rank_zero_info(f"num_tp_{cat}: {tps}")
        #     rank_zero_info(f"num_fn_{cat}: {fns}")
        #     rank_zero_info(f"num_fp_{cat}: {fps}")
        for key, predictions in self._predictions.items():
            os.makedirs(os.path.join(self._output_dir, key), exist_ok=True)
            filename = key
            with open(
                os.path.join(self._output_dir, filename + ".pkl"), "wb"
            ) as file:
                file.write(pickle.dumps(predictions))
            save(
                os.path.join(self._output_dir, filename + ".json"),
                predictions,
            )

    def process(
        self, inputs: List[List[InputSample]], outputs: ModelOutput
    ) -> None:
        """Process the pair of inputs and outputs."""
        for key, output in outputs.items():
            if key in ["num_tp", "num_fn", "num_fp"]:
                self._stats[key] += output
            else:
                for inp, out in zip(inputs, output):
                    metadata = inp[0].metadata[0]
                    prediction = copy.deepcopy(metadata)
                    prediction.labels = out
                    self._predictions[key].append(prediction)
                    if self._visualize and isinstance(prediction, FrameGroup):
                        rank_zero_warn(  # pragma: no cover
                            "Visualization not supported for multi-sensor datasets"  # pylint: disable=line-too-long
                        )
                    elif self._visualize:
                        # fmt: off
                        if self.viewer is None or metadata.frameIndex in [None,0,]: # type:ignore # pylint: disable=line-too-long
                        # fmt: on
                            size = metadata.size
                            assert size is not None
                            w, h = size.width, size.height
                            self.viewer = LabelViewer(
                                UIConfig(width=w, height=h)
                            )

                        video_name = (
                            prediction.videoName
                            if prediction.videoName is not None
                            else ""
                        )
                        save_path = os.path.join(
                            self._output_dir,
                            f"{key}_visualization",
                            video_name,
                            prediction.name,
                        )
                        self.viewer.draw(
                            np.array(
                                preprocess_image(inp[0].images.tensor[0])
                            ),
                            prediction,
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        self.viewer.save(save_path)
