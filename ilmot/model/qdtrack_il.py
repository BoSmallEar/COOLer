"""Incremental learning module for QD-Track with naive finetuning."""
from typing import Dict

import torch
from ilmot.model.qdtrack import QDTrack
from .model_utils import align_and_update_state_dicts, update_key_to_vis4d


class QDTrackIL(QDTrack):  # type: ignore # pylint: disable=too-many-ancestors
    """QDTrack incremental learning Module."""

    # pylint: disable=abstract-method

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
                    key = key[17:]
                    loaded_detector_state_dict[key] = value
                elif key.startswith("similarity_head."):
                    key = key[24:]
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
            detector_state_dict, loaded_detector_state_dict
        )
        align_and_update_state_dicts(
            similarity_state_dict, loaded_detector_state_dict
        )
        self.detector.load_state_dict(detector_state_dict)

        self.similarity_head.load_state_dict(similarity_state_dict)
