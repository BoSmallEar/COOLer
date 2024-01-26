"""Utilites related to models."""
import copy
from typing import Dict, Optional

import torch
from pytorch_lightning.utilities.distributed import rank_zero_info
from ilmot.common.registry import RegistryHolder
from ilmot.model.base import BaseModel
from ilmot.struct import ModuleCfg


def build_model(
    cfg: ModuleCfg,
    ckpt: Optional[str] = None,
    strict: bool = True,
    legacy_ckpt: bool = False,
    custom_load: bool = False,
    load_vist: bool = False,
) -> BaseModel:
    """Build VisT model and optionally load weights from ckpt.

    Args:
        cfg: Loaded model config.
        ckpt: Path of the checkpoint.
        strict: Load exact the same state dict.
        legacy_ckpt: Legacy checkpoints.
        custom_load: Whether or not to load the weight using a customized load
            function to support features like weight expansion or freeze.
        load_vist: Load checkpoints from the previous Vist module.

    Raises:
        ValueError: If no model type is specified.
        NotImplementedError: If model type is not found.

    Returns:
        Pytorch module with optional init weights.
    """
    registry = RegistryHolder.get_registry(BaseModel)
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type", None)
    if model_type is None:
        raise ValueError(f"Need type argument in module config: {cfg}")
    if model_type in registry:
        if custom_load:
            module = registry[model_type](**cfg)
            if ckpt is not None:
                module.load(ckpt, load_vist)
        else:
            if ckpt is None:
                module = registry[model_type](**cfg)
            else:
                module = registry[model_type].load_from_checkpoint(
                    ckpt, strict=strict, **cfg, legacy_ckpt=legacy_ckpt
                )
        assert isinstance(module, BaseModel)
        return module
    raise NotImplementedError(f"Model {model_type} not found.")


def align_and_update_state_dicts(
    model_state_dict: Dict[str, torch.Tensor],
    loaded_state_dict: Dict[str, torch.Tensor],
) -> None:
    """Adapted from mask-rcnn benchmark.

    Args:
        model_state_dict: State dict of the unintialized model.
        loaded_state_dict: State dict load from the checkpoint.
    """
    mm_dynamic_params_cls = [
        "roi_head.mm_roi_head.bbox_head.fc_cls.weight",
        "roi_head.mm_roi_head.bbox_head.fc_cls.bias",
    ]

    mm_dynamic_params_reg = [
        "roi_head.mm_roi_head.bbox_head.fc_reg.weight",
        "roi_head.mm_roi_head.bbox_head.fc_reg.bias",
    ]
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches,
    # where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0
        for i in current_keys
        for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(  # type: ignore
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)  # type: ignore
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if model_state_dict[key].size() == loaded_state_dict[key_old].size():
            model_state_dict[key] = loaded_state_dict[key_old]
        else:
            new_size = list(model_state_dict[key].size())
            rank_zero_info(f"align tensor | new | size of {key}: {new_size}")
            loaded_size = list(loaded_state_dict[key_old].size())
            rank_zero_info(
                f"align tensor | loaded | size of {key_old}: {loaded_size}"
            )
            assert key_old in mm_dynamic_params_cls + mm_dynamic_params_reg
            if key_old in mm_dynamic_params_cls:
                model_state_dict[key][
                    : loaded_size[0] - 1
                ] = loaded_state_dict[key_old][:-1]
                model_state_dict[key][-1] = loaded_state_dict[key_old][-1]
            elif key in mm_dynamic_params_reg:
                model_state_dict[key][: loaded_size[0]] = loaded_state_dict[
                    key_old
                ]


def update_key_to_vis4d(key: str):
    """Update vist to vis4d detector keys."""
    if key.startswith("mm_detector.backbone"):
        key = key[20:]
        key = "backbone.mm_backbone" + key
    elif key.startswith("mm_detector.neck"):
        key = key[16:]
        key = "backbone.neck.mm_neck" + key
    elif key.startswith("mm_detector.rpn_head"):
        key = key[20:]
        key = "rpn_head.mm_dense_head" + key
    elif key.startswith("mm_detector.roi_head"):
        key = key[20:]
        key = "roi_head.mm_roi_head" + key
    else:
        NotImplementedError(f"Key {key} not found.")
    return key
