"""Trainer wrapper for ILMOT."""
import os
import os.path as osp
import sys
from typing import List, Optional, Tuple
import pickle
import copy

import numpy as np
import numpy.typing as npt
import torch
from torch.utils import data
from torch.distributed import destroy_process_group
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import (
    rank_zero_warn,
)
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin, DDPSpawnPlugin
from pytorch_lightning.utilities.device_parser import parse_gpu_ids
from tqdm import tqdm
from scalabel.vis.label import LabelViewer, UIConfig
from ilmot.config import parse_config
from ilmot.engine.trainer import (
    setup_category_mapping,
)
from ilmot.data.handler import Vis4DDatasetHandler
from ilmot.struct import ModuleCfg
from ilmot.model import BaseModel
from ilmot.common.registry import build_component
from ilmot.data.datasets import BaseDatasetLoader
from ilmot.data import Vis4DDataModule
from ilmot.data.samplers import build_data_sampler
from ilmot.engine.utils import (
    split_args,
    setup_logging,
    Vis4DTQDMProgressBar,
)
from ilmot.struct import DictStrAny
from ilmot.vis.utils import preprocess_image
from ilmot.config.config import load_config

from .config import ILMOTConfig
from .config.defaults import default_argument_parser
from .data import ILMOTDataset
from .model import build_model
from .tools import (
    ILMOTEvaluatorCallback,
    ILMOTWriterCallback,
    visualize_embeddings,
)


def default_setup(
    cfg: ILMOTConfig,
    trainer_args: Optional[DictStrAny] = None,
    training: bool = True,
) -> pl.Trainer:
    """Perform some basic common setups at the beginning of a job.

    1. Set all seeds
    2. Setup callback: tensorboard logger, LRMonitor, GPUMonitor, Checkpoint
    3. Init pytorch lightning Trainer
    4. Set up cmd line logger
    5. Log basic information about environment, trainer arguments, and config
    6. Backup the args / config to the output directory
    """
    # set seed to be fixed as 777
    pl.seed_everything(cfg.launch.seed, workers=True)

    # prepare trainer args
    if trainer_args is None:
        trainer_args = {}  # pragma: no cover
    if "trainer" in cfg.dict().keys():
        trainer_args.update(cfg.dict()["trainer"])

    # setup experiment logging
    if "logger" not in trainer_args or (
        isinstance(trainer_args["logger"], bool) and trainer_args["logger"]
    ):
        if cfg.launch.wandb:  # pragma: no cover
            exp_logger = pl.loggers.WandbLogger(
                save_dir=cfg.launch.work_dir,
                project=cfg.launch.exp_name,
                name=cfg.launch.version,
            )
        else:
            exp_logger = pl.loggers.TensorBoardLogger(  # type: ignore
                save_dir=cfg.launch.work_dir,
                name=cfg.launch.exp_name,
                version=cfg.launch.version,
                default_hp_metric=False,
            )
        trainer_args["logger"] = exp_logger

    # add learning rate / GPU stats monitor (logs to tensorboard)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # add progress bar (train progress separate from validation)

    progress_bar = Vis4DTQDMProgressBar()

    # add Model checkpointer
    output_dir = osp.join(
        cfg.launch.work_dir, cfg.launch.exp_name, cfg.launch.version
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=osp.join(output_dir, "checkpoints"),
        filename="best",
        verbose=True,
        mode="max",
        monitor="track/MOTA_new_class",
        save_top_k=1,
        save_last=True,
        every_n_epochs=cfg.launch.checkpoint_period,
        save_on_train_epoch_end=False,
    )

    # resume from checkpoint if specified
    if cfg.launch.resume:  # pragma: no cover
        if cfg.launch.weights is not None:
            resume_path = cfg.launch.weights
        elif osp.exists(osp.join(output_dir, "checkpoints/last.ckpt")):
            resume_path = osp.join(output_dir, "checkpoints/last.ckpt")
        else:
            raise ValueError(
                "cfg.launch.resume set to True but there is no checkpoint to "
                "resume! Please specify a checkpoint via cfg.launch.weights "
                "or configure a directory that contains a checkpoint at "
                "work_dir/exp_name/version/checkpoints/last.ckpt."
            )

        trainer_args["resume_from_checkpoint"] = resume_path

    # add distributed plugin
    if "gpus" in trainer_args:  # pragma: no cover
        gpu_ids = parse_gpu_ids(trainer_args["gpus"])
        num_gpus = len(gpu_ids) if gpu_ids is not None else 0
        if num_gpus > 1:
            if (
                trainer_args["accelerator"] == "ddp"
                or trainer_args["accelerator"] is None
            ):
                ddp_plugin = DDPPlugin(
                    find_unused_parameters=cfg.launch.find_unused_parameters
                )
                trainer_args["plugins"] = [ddp_plugin]
            elif trainer_args["accelerator"] == "ddp_spawn":
                ddp_plugin = DDPSpawnPlugin(
                    find_unused_parameters=cfg.launch.find_unused_parameters
                )  # type: ignore
                trainer_args["plugins"] = [ddp_plugin]
            elif trainer_args["accelerator"] == "ddp2":
                ddp_plugin = DDP2Plugin(
                    find_unused_parameters=cfg.launch.find_unused_parameters
                )
                trainer_args["plugins"] = [ddp_plugin]
            if (
                cfg.data is not None
                and "train_sampler" in cfg.data
                and cfg.data["train_sampler"] is not None
                and training
            ):
                # using custom sampler
                trainer_args["replace_sampler_ddp"] = False

    # create trainer
    trainer_args["callbacks"] = [lr_monitor, progress_bar, checkpoint]

    trainer = pl.Trainer(**trainer_args)

    # setup cmd line logging, print and save info about trainer / cfg / env
    setup_logging(output_dir, trainer_args, cfg)
    return trainer


def build_datasets(  # pylint: disable=dangerous-default-value
    dataset_cfgs: List[ModuleCfg],
    image_channel_mode: str,
    training: bool = True,
    handler_cfg: Optional[ModuleCfg] = None,
    train_category: Optional[List[str]] = None,
    test_category: Optional[List[str]] = None,
    pseudo_datasets=[],
    keep_anno: bool = True,
    keep_old_class: bool = False,
) -> Tuple[List[Vis4DDatasetHandler], List[ILMOTDataset]]:
    """Build datasets based on configs."""
    datasets: List[ILMOTDataset] = []
    tmpdir = os.getenv("TMPDIR")
    assert tmpdir is not None
    _handler_cfgs = []
    i = 0
    for dl_cfg in dataset_cfgs:
        dl_cfg["annotations"] = osp.join(tmpdir, dl_cfg["annotations"])
        dl_cfg["data_root"] = osp.join(tmpdir, dl_cfg["data_root"])
        mapper_cfg = dl_cfg.pop("sample_mapper", {})
        ref_cfg = dl_cfg.pop("ref_sampler", {})
        if (
            "image_channel_mode" in mapper_cfg
            and mapper_cfg["image_channel_mode"] != image_channel_mode
        ):  # pragma: no cover
            rank_zero_warn(
                f"'image_channel_mode'={mapper_cfg['image_channel_mode']} "
                "specified in SampleMapper configuration, but model expects "
                f"{image_channel_mode}. Switching to mode required by model."
            )
        mapper_cfg["image_channel_mode"] = image_channel_mode

        # TODO Temporary fix to keep configs compatible, will be removed once static configurations are replaced # pylint: disable=line-too-long,fixme
        _handler_cfg = {}
        _handler_cfg["clip_bboxes_to_image"] = mapper_cfg.pop(
            "clip_bboxes_to_image", True
        )
        _handler_cfg["min_bboxes_area"] = mapper_cfg.pop(
            "min_bboxes_area", 7.0 * 7.0
        )
        _handler_cfg["transformations"] = mapper_cfg.pop("transformations", [])
        _handler_cfgs.append(_handler_cfg)
        pseudo_dataset = (
            pseudo_datasets[i] if len(pseudo_datasets) > 0 else None
        )
        i = i + 1

        datasets.append(
            ILMOTDataset(
                build_component(dl_cfg, bound=BaseDatasetLoader),
                training,
                mapper_cfg,
                ref_cfg,
                train_category,
                test_category,
                pseudo_dataset,
                keep_anno,
                keep_old_class,
            )
        )
    if handler_cfg is None:
        result = []
        for ds, _handler_cfg in zip(datasets, _handler_cfgs):
            _handler_cfg["datasets"] = [ds]
            if "type" not in _handler_cfg:
                _handler_cfg["type"] = "Vis4DDatasetHandler"
            result.append(
                build_component(_handler_cfg, bound=Vis4DDatasetHandler)
            )
    else:
        handler_cfg["datasets"] = datasets
        if "type" not in handler_cfg:
            handler_cfg["type"] = "Vis4DDatasetHandler"
        result = [build_component(handler_cfg, bound=Vis4DDatasetHandler)]
    return result, datasets


def build_callbacks(
    cfg: ILMOTConfig,
    datasets: List[ILMOTDataset],
    out_dir: str = None,
    is_predict: bool = False,
    vis: bool = False,
) -> List[Callback]:
    """Build callbacks."""
    callbacks: List[Callback] = []
    for i, d in enumerate(datasets):
        if not is_predict:
            callbacks.append(
                ILMOTEvaluatorCallback(
                    i,
                    d.dataset,
                    None,
                    cfg.model["category_mapping"],
                    cfg.model["train_category"],
                    cfg.model["test_category"],
                    cfg.compute_statistics,
                    out_dir,
                )
            )
        else:
            assert out_dir is not None
            callbacks.append(
                ILMOTWriterCallback(
                    i,
                    out_dir,
                    vis,
                    cfg.model["category_mapping"],
                    cfg.model["train_category"],
                    cfg.model["test_category"],
                )
            )
    return callbacks


def setup_experiment(
    cfg: ILMOTConfig, trainer_args: DictStrAny
) -> Tuple[pl.Trainer, BaseModel, pl.LightningDataModule]:
    """Build trainer, model, and data module."""
    # setup trainer
    is_train = cfg.launch.action in ("train", "predict")
    trainer = default_setup(cfg, trainer_args, training=is_train)

    # setup model
    cfg.model["ignore_new_class"] = cfg.ignore_new_class

    model = build_model(
        cfg.model,
        cfg.launch.weights if not cfg.launch.resume or not is_train else None,
        not cfg.launch.not_strict,
        cfg.launch.legacy_ckpt,
        cfg.custom_load and is_train,
        cfg.load_vist,
    )

    # setup category_mappings
    setup_category_mapping(
        cfg.train + cfg.test + cfg.predict, cfg.model["category_mapping"]
    )

    # build datasets
    cmode = (
        cfg.model["image_channel_mode"]
        if "image_channel_mode" in cfg.model
        else "RGB"
    )

    pseudo_datasets = []
    if cfg.pl_dataset_path is not None:
        with open(cfg.pl_dataset_path, "rb") as file:
            pseudo_datasets.append(pickle.loads(file.read()))

    if cfg.launch.action == "train":
        train_handlers, _ = (
            build_datasets(
                cfg.train,
                cmode,
                True,
                cfg.train_handler,
                cfg.model["train_category"],
                cfg.model["test_category"],
                pseudo_datasets,
                True,
                False,
            )
            if is_train
            else (None, None)
        )
    else:
        train_handlers = None
    if train_handlers is not None:
        if len(train_handlers) > 1:
            train_handler = Vis4DDatasetHandler(train_handlers, False, 0.0)
        else:
            train_handler = train_handlers[0]
    else:
        train_handler = None

    test_handlers, test_datasets, = (
        None,
        None,
    )
    predict_handlers, predict_datasets = None, None
    train_sampler: Optional[data.Sampler[List[int]]] = None
    if cfg.launch.action == "train":
        if cfg.data is not None and "train_sampler" in cfg.data:
            # build custom train sampler
            train_sampler = build_data_sampler(
                cfg.data["train_sampler"],
                train_handler,
                cfg.launch.samples_per_gpu,
            )

    if cfg.launch.action == "predict":
        if cfg.launch.input_dir:
            input_dir = cfg.launch.input_dir
            if input_dir[-1] == "/":
                input_dir = input_dir[:-1]
            dataset_name = osp.basename(input_dir)
            predict_loaders = [
                dict(type="Custom", name=dataset_name, data_root=input_dir)
            ]
        else:
            predict_loaders = cfg.predict
        predict_handlers, predict_datasets = build_datasets(
            predict_loaders,
            cmode,
            True,
            None,
            cfg.model["train_category"],
            cfg.model["test_category"],
            [],
            True,
            True,
        )
    else:
        test_handlers, test_datasets = build_datasets(
            cfg.test,
            cmode,
            False,
            None,
            cfg.model["train_category"],
            cfg.model["test_category"],
            [],
            True,
            False,
        )

    # build data module
    data_module = Vis4DDataModule(
        cfg.launch.samples_per_gpu,
        cfg.launch.workers_per_gpu,
        train_datasets=train_handler,
        test_datasets=test_handlers,
        predict_datasets=predict_handlers,
        seed=cfg.launch.seed,
        train_sampler=train_sampler,
    )

    # setup callbacks
    if cfg.launch.action == "train":
        if test_datasets is not None and len(test_datasets) > 0:
            trainer.callbacks += build_callbacks(  # pylint: disable=no-member
                cfg, test_datasets
            )
    elif cfg.launch.action == "test":
        assert test_datasets is not None and len(
            test_datasets
        ), "No test datasets specified!"

        vis_dir = osp.join(
            "ilmot-images", cfg.launch.exp_name, cfg.launch.version
        )
        os.makedirs(vis_dir, exist_ok=True)

        trainer.callbacks += build_callbacks(  # pylint: disable=no-member
            cfg, test_datasets, vis_dir
        )

    elif cfg.launch.action == "predict":
        assert (
            predict_datasets is not None and len(predict_datasets) > 0
        ), "No predict datasets specified!"
        predict_dir = osp.join(
            "ilmot-pseudolabel", cfg.launch.exp_name, cfg.launch.version
        )
        os.makedirs(predict_dir, exist_ok=True)

        trainer.callbacks += build_callbacks(  # pylint: disable=no-member
            cfg, predict_datasets, predict_dir, True, cfg.launch.visualize
        )
    else:
        raise NotImplementedError(f"Action {cfg.launch.action} not known!")

    return trainer, model, data_module


def visualize(cfg: ILMOTConfig) -> None:
    """Visualization function.

    Args:
        cfg: Config object parsed from the config file.

    """
    model = build_model(
        cfg.model,
        cfg.launch.weights,
        not cfg.launch.not_strict,
        cfg.launch.legacy_ckpt,
        False,
    )
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
        model.cuda()

    cmode = (
        cfg.model["image_channel_mode"]
        if "image_channel_mode" in cfg.model
        else "RGB"
    )

    # setup category_mappings
    setup_category_mapping(cfg.test, cfg.model["category_mapping"])

    _, visualize_datasets = build_datasets(
        cfg.test,
        cmode,
        False,
        None,
        cfg.model["train_category"],
        cfg.model["test_category"],
        [],
        True,
    )
    visulaize_dataset = visualize_datasets[0]
    visulaize_dataset.mapper.set_training(True)
    output_dir = f"ilmot-images/{cfg.launch.exp_name}"
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # detect dataset
    if not bool(visulaize_dataset.video_to_indices):
        indices = range(len(visulaize_dataset))
    # use the first sequence by default
    elif isinstance(cfg.sequence_id, int):
        indices = list(visulaize_dataset.video_to_indices.values())[
            cfg.sequence_id  # type: ignore
        ]
    else:
        indices = visulaize_dataset.video_to_indices[cfg.sequence_id]  # type: ignore # pylint: disable=line-too-long,

    # indices = []
    # indices.extend(visulaize_dataset.video_to_indices["b1c9c847-3bda4659"])
    # indices.extend(visulaize_dataset.video_to_indices["b1c66a42-6f7d68ca"])
    # indices.extend(visulaize_dataset.video_to_indices["b1ca2e5d-84cf9134"])
    # indices.extend(visulaize_dataset.video_to_indices["b1e1a7b8-b397c445"])

    # indices = list(range(len(visulaize_dataset)))
    # import random

    # random.Random(1).shuffle(indices)
    # indices = indices[:600]

    if cfg.vis_sequence:
        metadata = visulaize_dataset.__getitem__(0)[0].metadata[0]
        viewer = LabelViewer(
            UIConfig(width=metadata.size.width, height=metadata.size.height),
            label_cfg=visulaize_dataset.dataset.metadata_cfg,
        )
        for i in tqdm(indices):
            input_sample = visulaize_dataset.__getitem__(i)[0]
            image = input_sample.images.__getitem__(0)
            input_sample = input_sample.to(model.device)
            with torch.no_grad():
                bboxes = model.forward_test([input_sample])
            out = bboxes["detect"][0] if cfg.vis_detect else bboxes["track"][0]
            metadata = input_sample.metadata[0]
            prediction = copy.deepcopy(metadata)
            prediction.labels = out

            video_name = (
                prediction.videoName
                if prediction.videoName is not None
                else ""
            )
            save_path = os.path.join(
                output_dir,
                video_name,
                prediction.name,
            )

            viewer.draw(
                np.array(preprocess_image(image.tensor[0])),
                prediction,
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            viewer.save(save_path)

    else:
        embeddings_list: List[npt.NDArray[np.float64]] = []
        class_ids_list: List[npt.NDArray[np.int64]] = []
        for i in tqdm(indices):
            input_sample = visulaize_dataset.__getitem__(i)[0]
            input_sample = input_sample.to(model.device)
            with torch.no_grad():
                embeddings, class_ids, track_ids = model.forward_embeddings(
                    [input_sample], add_background=False
                )

            embeddings_list.append(embeddings.cpu().numpy())
            class_ids_list.append(class_ids.cpu().numpy())

        all_embeddings = np.concatenate(embeddings_list, axis=0)  # type: ignore # pylint: disable=line-too-long,
        all_class_ids = np.concatenate(class_ids_list, axis=0)  # type: ignore
        visualize_embeddings(
            all_embeddings,
            all_class_ids,
            cfg.model["category_mapping"],
            output_dir,
        )


def train(
    trainer: pl.Trainer, model: BaseModel, data_module: pl.LightningDataModule
) -> None:
    """Training function."""
    trainer.fit(model, data_module)


def test(
    trainer: pl.Trainer, model: BaseModel, data_module: pl.LightningDataModule
) -> None:
    """Test function."""
    trainer.test(model, data_module, verbose=False)


def predict(
    trainer: pl.Trainer, model: BaseModel, data_module: pl.LightningDataModule
) -> None:
    """Prediction function."""
    trainer.predict(model, data_module)


def cli_main() -> None:  # pragma: no cover
    """Main function when called from command line."""
    parser = default_argument_parser()
    args = parser.parse_args()
    vis4d_args, trainer_args = split_args(args)
    cfg = parse_config(vis4d_args)
    cfg = ILMOTConfig(**cfg.dict())

    if args.action == "visualize":
        visualize(cfg)
    else:
        # setup experiment
        trainer, model, data_module = setup_experiment(cfg, trainer_args)

        if args.action == "train":
            train(trainer, model, data_module)
        elif args.action == "test":
            test(trainer, model, data_module)
        elif args.action == "predict":
            predict(trainer, model, data_module)

        else:
            raise NotImplementedError(f"Action {args.action} not known!")


def run_stage(cfg, trainer_args):
    """Run a single CL stage, given cfg and args."""
    if cfg.launch.action == "visualize":
        visualize(cfg)
    else:
        # setup experiment
        trainer, model, data_module = setup_experiment(cfg, trainer_args)

        if cfg.launch.action == "train":
            train(trainer, model, data_module)
        elif cfg.launch.action == "test":
            test(trainer, model, data_module)
        elif cfg.launch.action == "predict":
            predict(trainer, model, data_module)

        else:
            raise NotImplementedError(f"Action {cfg.launch.action} not known!")

    # kill subprocesses

    destroy_process_group()
    if model.global_rank != 0:
        sys.exit(0)


def get_args(use_run_cfg=False):
    """Get configs and args from the config file."""
    parser = default_argument_parser()
    if not use_run_cfg:
        args = parser.parse_args()
    else:
        run_cfg_path = sys.argv[1]
        run_cfg = load_config(run_cfg_path)

        args = parser.parse_args(sys.argv[2:])
    vis4d_args, trainer_args = split_args(args)
    cfg = parse_config(vis4d_args)
    cfg = ILMOTConfig(**cfg.dict())
    if not use_run_cfg:
        return cfg, trainer_args
    return cfg, trainer_args, run_cfg


if __name__ == "__main__":  # pragma: no cover
    cli_main()
