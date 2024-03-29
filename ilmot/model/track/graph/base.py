"""Tracking base class."""

import abc
from typing import List, Optional, Union, cast, overload

import torch

from ilmot.common import Vis4DModule
from ilmot.struct import InputSample, LabelInstances, LossesType


class BaseTrackGraph(Vis4DModule[LabelInstances, LossesType]):
    """Base class for tracking graph optimization."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset track memory during inference."""
        raise NotImplementedError

    @overload  # type: ignore[override]
    def __call__(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        **kwargs: torch.Tensor,
    ) -> LabelInstances:  # noqa: D102
        ...

    @overload
    def __call__(
        self,
        inputs: List[InputSample],
        predictions: List[LabelInstances],
        targets: Optional[List[LabelInstances]],
        **kwargs: List[torch.Tensor],
    ) -> LossesType:
        ...

    def __call__(
        self,
        inputs: Union[List[InputSample], InputSample],
        predictions: Union[List[LabelInstances], LabelInstances],
        targets: Optional[List[LabelInstances]] = None,
        **kwargs: Union[List[torch.Tensor], torch.Tensor],
    ) -> Union[LabelInstances, LossesType]:
        """Forward method. Decides between train / test logic."""
        if targets is not None:  # pragma: no cover
            inputs = cast(List[InputSample], inputs)
            predictions = cast(List[LabelInstances], predictions)
            return self.forward_train(inputs, predictions, targets, **kwargs)
        inputs = cast(InputSample, inputs)
        predictions = cast(LabelInstances, predictions)
        return self.forward_test(inputs, predictions, **kwargs)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: List[InputSample],
        predictions: List[LabelInstances],
        targets: List[LabelInstances],
        **kwargs: List[torch.Tensor],
    ) -> LossesType:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        **kwargs: torch.Tensor,
    ) -> LabelInstances:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError
