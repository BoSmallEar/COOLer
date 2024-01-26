"""ILMOT dataset and utils."""
from collections import defaultdict
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset
from pytorch_lightning.utilities.distributed import rank_zero_info
from scalabel.label.utils import get_leaf_categories
from ilmot.data.mapper import BaseSampleMapper
from ilmot.data.reference import BaseReferenceSampler
from ilmot.struct import ModuleCfg, InputSample
from ilmot.data.datasets import BaseDatasetLoader
from ilmot.data.utils import (
    filter_attributes,
    DatasetFromList,
    prepare_labels,
    print_class_histogram,
)
from ilmot.common.registry import build_component
from ilmot.common.utils.time import Timer


class ScalabelDataset(Dataset):  # type: ignore
    """Preprocess Scalabel format data into Vis4D input format."""

    def __init__(
        self,
        dataset: BaseDatasetLoader,
        training: bool,
        mapper: Union[BaseSampleMapper, ModuleCfg] = BaseSampleMapper(),
        ref_sampler: Union[
            BaseReferenceSampler, ModuleCfg
        ] = BaseReferenceSampler(),
    ):
        """Init."""
        rank_zero_info("Initializing dataset: %s", dataset.name)
        self.training = training
        cats_name2id = dataset.category_mapping
        if cats_name2id is not None:
            if isinstance(list(cats_name2id.values())[0], int):
                class_list = list(set(cls for cls in cats_name2id))
            else:
                class_list = list(
                    set(
                        cls
                        for field in cats_name2id
                        for cls in list(cats_name2id[field].keys())  # type: ignore  # pylint: disable=line-too-long
                    )
                )
            discard_labels_outside_set(dataset.frames, class_list)
        else:
            class_list = list(
                set(
                    c.name
                    for c in get_leaf_categories(
                        dataset.metadata_cfg.categories
                    )
                )
            )
            cats_name2id = {v: i for i, v in enumerate(class_list)}
        self.cats_name2id = cats_name2id
        if isinstance(mapper, dict):
            if "type" not in mapper:
                mapper["type"] = "BaseSampleMapper"
            self.mapper: BaseSampleMapper = build_component(
                mapper, bound=BaseSampleMapper
            )
        else:
            self.mapper = mapper
        self.mapper.setup_categories(cats_name2id)
        self.mapper.set_training(self.training)

        dataset.frames = filter_attributes(dataset.frames, dataset.attributes)

        t = Timer()
        frequencies = prepare_labels(
            dataset.frames,
            class_list,
            dataset.compute_global_instance_ids,
        )
        rank_zero_info(
            f"Preprocessing {len(dataset.frames)} frames takes {t.time():.2f}"
            " seconds."
        )
        print_class_histogram(frequencies)

        self.dataset = dataset
        self.dataset.frames = DatasetFromList(self.dataset.frames)
        if self.dataset.groups is not None:
            t.reset()
            prepare_labels(
                self.dataset.groups,
                class_list,
                dataset.compute_global_instance_ids,
            )
            rank_zero_info(
                f"Preprocessing {len(self.dataset.groups)} groups takes "
                f"{t.time():.2f} seconds."
            )
            self.dataset.groups = DatasetFromList(self.dataset.groups)

        self._fallback_candidates = set(range(len(self.dataset.frames)))

        if isinstance(ref_sampler, dict):
            if "type" not in ref_sampler:
                ref_sampler["type"] = "BaseReferenceSampler"
            self.ref_sampler: BaseReferenceSampler = build_component(
                ref_sampler, bound=BaseReferenceSampler
            )
        else:
            self.ref_sampler = ref_sampler  # pragma: no cover
        self.ref_sampler.create_mappings(
            self.dataset.frames, self.dataset.groups
        )

        self.has_sequences = bool(self.ref_sampler.video_to_indices)
        self._show_retry_warn = True

    def __len__(self) -> int:
        """Return length of dataset."""
        if self.dataset.groups is not None and not self.training:
            return len(self.dataset.groups)
        return len(self.dataset.frames)

    def __getitem__(self, idx: int) -> List[InputSample]:
        """Fully prepare a sample for training/inference."""
        retry_count = 0
        cur_idx = int(idx)

        if not self.training:
            if self.dataset.groups is not None:
                group = self.dataset.groups[cur_idx]
                if not self.dataset.multi_sensor_inference:
                    cur_data = self.mapper(
                        self.dataset.frames[
                            self.ref_sampler.frame_name_to_idx[group.frames[0]]
                        ]
                    )
                    assert cur_data is not None
                    return [cur_data]

                group_data = self.mapper(group)
                assert group_data is not None
                data = [group_data]
                for fname in group.frames:
                    cur_data = self.mapper(
                        self.dataset.frames[
                            self.ref_sampler.frame_name_to_idx[fname]
                        ],
                    )
                    assert cur_data is not None
                    data.append(cur_data)
                return data

            cur_data = self.mapper(self.dataset.frames[cur_idx])
            assert cur_data is not None
            data = [cur_data]
            return data

        while True:
            if self.dataset.groups is not None:
                group = self.dataset.groups[
                    self.ref_sampler.frame_to_group[
                        self.ref_sampler.frame_name_to_idx[
                            self.dataset.frames[cur_idx].name
                        ]
                    ]
                ]
                input_data = self.mapper(
                    self.dataset.frames[cur_idx],
                    group_url=group.url,
                    group_extrinsics=group.extrinsics,
                )
            else:
                input_data = self.mapper(self.dataset.frames[cur_idx])
            if input_data is not None:
                if input_data.metadata[0].attributes is None:
                    input_data.metadata[0].attributes = {}
                input_data.metadata[0].attributes["keyframe"] = True

                if self.ref_sampler.num_ref_imgs > 0:
                    ref_data = self.ref_sampler(
                        cur_idx, input_data, self.mapper
                    )
                    if ref_data is not None:
                        return [input_data] + ref_data
                else:
                    return [input_data]

            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = random.sample(self._fallback_candidates, k=1)[0]

            if self._show_retry_warn and retry_count >= 5:
                rank_zero_warn(
                    f"Failed to get an input sample for idx {cur_idx} after "
                    f"{retry_count} retries, this happens e.g. when "
                    "skip_empty_samples is activated and there are many "
                    "samples without (valid) labels. Please check your class "
                    "configuration and/or dataset labels if this is "
                    "undesired behavior."
                )
                self._show_retry_warn = False


class ILMOTDataset(ScalabelDataset):  # type: ignore
    """Preprocess Scalabel format data into ILMOT input format.

    Args:
        dataset: Dataset used to load Scalabel format data.
        training: Whether the data are used in training or testing.
        cats_name2id: A mapping from category names to their ids.
        image_channel_mode: The channel ordering of the original image.
        train_category: Categories used to train the model.
        test_category: Categories used to test the model.
        keep_old_class: If training keep all annotations of the old classes in
        frames of new classes, if testing, keep only sequences with where the
        new class is presented.


    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        dataset: BaseDatasetLoader,
        training: bool,
        mapper: Union[BaseSampleMapper, ModuleCfg] = BaseSampleMapper(),
        ref_sampler: Union[
            BaseReferenceSampler, ModuleCfg
        ] = BaseReferenceSampler(),
        train_category: Optional[List[str]] = None,
        test_category: Optional[List[str]] = None,
        pseudo_dataset=None,
        keep_anno: bool = True,
        keep_old_class: bool = False,
    ):
        """Init."""
        rank_zero_info("Initializing dataset: %s", dataset.name)
        self.training = training
        self.train_category = train_category
        cats_name2id = dataset.category_mapping
        self.cats_name2id = cats_name2id
        self.keep_old_class = keep_old_class
        self.dataset = dataset

        # first filter attributes
        dataset.frames = filter_attributes(dataset.frames, dataset.attributes)

        if train_category is not None and training and not keep_old_class:
            self.discard_labels_outside_set(train_category)
        elif test_category is not None and not training:
            self.discard_labels_outside_set(test_category)
        elif cats_name2id is not None:
            self.discard_labels_outside_set(list(cats_name2id.keys()))
        else:
            cats_name2id = {
                v: i
                for i, v in enumerate(
                    [
                        c.name
                        for c in get_leaf_categories(
                            dataset.metadata_cfg.categories
                        )
                    ]
                )
            }

        if not self.training and test_category is not None:
            cats_name2id = {v: i for i, v in enumerate(test_category)}

        video_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, entry in enumerate(dataset.frames):
            if entry.videoName is not None:
                video_to_indices[entry.videoName].append(idx)
        has_sequences = bool(video_to_indices)
        if self.training:  # pylint: disable=too-many-nested-blocks
            if train_category is not None:
                if has_sequences:  # if it is from the tracking dataset
                    discarded_video_frames = []
                    for frame_ids in video_to_indices.values():
                        have_class = False
                        for i in frame_ids:
                            for ann in dataset.frames[i].labels:
                                if ann.category in train_category:
                                    have_class = True
                                    break
                        if not have_class:
                            for i in frame_ids:
                                discarded_video_frames.append(i)
                    for i in reversed(sorted(discarded_video_frames)):
                        dataset.frames.pop(i)
                else:  # from the detection dataset
                    discarded_video_frames = []
                    for idx, frame in enumerate(dataset.frames):
                        have_class = False
                        if frame.labels is None:
                            discarded_video_frames.append(idx)
                        else:
                            for ann in frame.labels:
                                if ann.category in train_category:
                                    have_class = True
                                    break
                            if not have_class:
                                discarded_video_frames.append(idx)
                    for i in reversed(sorted(discarded_video_frames)):
                        dataset.frames.pop(i)

            else:
                discarded_video_frames = []
                for frame_ids in video_to_indices.values():
                    if len(frame_ids) < 5:
                        for i in frame_ids:
                            discarded_video_frames.append(i)
                for i in reversed(sorted(discarded_video_frames)):
                    dataset.frames.pop(i)

            if pseudo_dataset is not None:
                ## pseudo labels are generated using the same training dataset
                assert len(pseudo_dataset) == len(dataset.frames)
                for pl_sample in pseudo_dataset:
                    for pl_label in pl_sample.labels:
                        pl_label.attributes = {}
                        old_id = pl_label.id
                        pl_label.id = "pl" + old_id
                        pl_label.score = None
                        pl_label.attributes["occluded"] = False
                        pl_label.attributes["truncated"] = False
                        pl_label.attributes["crowd"] = False
                        pl_label.attributes["category_id"] = cats_name2id[
                            pl_label.category
                        ]
                        pl_label.attributes["instance_id"] = (
                            int(old_id) + 100000
                        )

                for pl_sample, gt_sample in zip(
                    pseudo_dataset, dataset.frames
                ):
                    gt_sample.labels.extend(pl_sample.labels)

        if isinstance(mapper, dict):
            if "type" not in mapper:
                mapper["type"] = "BaseSampleMapper"
            self.mapper: BaseSampleMapper = build_component(
                mapper, bound=BaseSampleMapper
            )
        else:
            self.mapper = mapper
        self.mapper.setup_categories(cats_name2id)
        self.mapper.set_training(keep_anno)

        class_list = list(set(cls for cls in cats_name2id))

        video_to_indices = defaultdict(list)
        for idx, entry in enumerate(dataset.frames):
            if entry.videoName is not None:
                video_to_indices[entry.videoName].append(idx)
        self.video_to_indices = video_to_indices

        t = Timer()
        frequencies = prepare_labels(
            dataset.frames,
            class_list,
            dataset.compute_global_instance_ids,
        )
        rank_zero_info(
            f"Preprocessing {len(dataset.frames)} frames takes {t.time():.2f}"
            " seconds."
        )
        print_class_histogram(frequencies)

        self.dataset.frames = DatasetFromList(self.dataset.frames)
        if self.dataset.groups is not None:
            t.reset()
            prepare_labels(
                self.dataset.groups,
                class_list,
                dataset.compute_global_instance_ids,
            )
            rank_zero_info(
                f"Preprocessing {len(self.dataset.groups)} groups takes "
                f"{t.time():.2f} seconds."
            )
            self.dataset.groups = DatasetFromList(self.dataset.groups)

        self._fallback_candidates = set(range(len(self.dataset.frames)))

        if isinstance(ref_sampler, dict):
            if "type" not in ref_sampler:
                ref_sampler["type"] = "BaseReferenceSampler"
            self.ref_sampler: BaseReferenceSampler = build_component(
                ref_sampler, bound=BaseReferenceSampler
            )
        else:
            self.ref_sampler = ref_sampler  # pragma: no cover
        self.ref_sampler.create_mappings(
            self.dataset.frames, self.dataset.groups
        )

        self.has_sequences = bool(self.ref_sampler.video_to_indices)
        self._show_retry_warn = True

    # DISCUSS: Check the right way to sample ref views when discarding frames.
    def discard_labels_outside_set(self, class_set: List[str]) -> None:
        """Discard labels outside given set of classes.

        Args:
            class_set: List of clases whose annotations are kept.

        """
        remove_ids = []
        assert self.cats_name2id is not None
        for idx, frame in enumerate(self.dataset.frames):
            remove_anns = []
            if frame.labels is not None:
                for i, ann in enumerate(frame.labels):
                    if not ann.category in class_set:
                        remove_anns.append(i)
                for i in reversed(remove_anns):
                    frame.labels.pop(i)

                if 0 == len(frame.labels):
                    remove_ids.append(idx)
            else:
                remove_ids.append(idx)

        if self.training and self.train_category is None:
            for i in reversed(remove_ids):
                self.dataset.frames.pop(i)
