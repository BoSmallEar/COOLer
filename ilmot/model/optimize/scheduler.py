"""Vis4D LR schedulers."""
from typing import List

from pydantic import BaseModel, validator
from torch.optim import Optimizer, lr_scheduler

from ilmot.common.registry import RegistryHolder
from ilmot.struct import DictStrAny


class LRSchedulerConfig(BaseModel):
    """Config for Vis4D model LR scheduler."""

    type: str = "StepLR"
    mode: str = "epoch"
    warmup: str = "linear"
    warmup_steps: int = 500
    warmup_ratio: float = 0.001
    kwargs: DictStrAny = {"step_size": 10}

    @validator("mode", check_fields=False)
    def validate_mode(  # pylint: disable=no-self-argument,no-self-use
        cls, value: str
    ) -> str:
        """Check mode attribute."""
        if value not in ["step", "epoch"]:
            raise ValueError("mode must be step or epoch")
        return value

    @validator("warmup", check_fields=False)
    def validate_warmup(  # pylint: disable=no-self-argument,no-self-use
        cls, value: str
    ) -> str:
        """Check warmup attribute."""
        if value not in ["none", "constant", "linear", "exponential"]:
            raise ValueError(
                "warmup must be none, constant, linear or exponential"
            )
        return value

    @validator("warmup_steps", check_fields=False)
    def validate_warmup_steps(  # pylint: disable=no-self-argument,no-self-use
        cls, value: int
    ) -> int:
        """Check warmup steps attribute."""
        if not value > 0:
            raise ValueError("warmup_steps must be a positive integer")
        return value

    @validator("warmup_ratio", check_fields=False)
    def validate_warmup_ratio(  # pylint: disable=no-self-argument,no-self-use
        cls, value: int
    ) -> int:
        """Check warmup_ratio attribute."""
        if not 0 < value <= 1.0:
            raise ValueError("warmup_ratio must be in range (0,1]")
        return value


def get_warmup_lr(
    cfg: LRSchedulerConfig, cur_steps: int, regular_lr: float
) -> float:
    """Compute current learning rate according to warmup configuration."""
    warmup_lr = regular_lr
    if cfg.warmup == "constant":
        warmup_lr = regular_lr * cfg.warmup_ratio
    elif cfg.warmup == "linear":
        k = (1 - cur_steps / cfg.warmup_steps) * (1 - cfg.warmup_ratio)
        warmup_lr = regular_lr * (1 - k)
    elif cfg.warmup == "exponential":
        k = cfg.warmup_ratio ** (1 - cur_steps / cfg.warmup_steps)
        warmup_lr = regular_lr * k
    return warmup_lr


class BaseLRScheduler(lr_scheduler._LRScheduler, metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long,protected-access
    """Dummy LRScheduler class supporting Vis4D registry."""

    def get_lr(self) -> List[float]:
        """Compute current learning rate."""
        raise NotImplementedError


class PolyLRScheduler(BaseLRScheduler):
    """Polynomial learning rate decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Initialize PolyLRScheduler."""
        self.max_steps = max_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute current learning rate."""
        if self._step_count >= self.max_steps:  # pragma: no cover
            return [self.min_lr for _ in self.base_lrs]
        coeff = (1 - self._step_count / self.max_steps) ** self.power
        return [
            (base_lr - self.min_lr) * coeff + self.min_lr
            for base_lr in self.base_lrs
        ]


def build_lr_scheduler(
    optimizer: Optimizer, cfg: LRSchedulerConfig
) -> BaseLRScheduler:
    """Build LR Scheduler from config."""
    registry = RegistryHolder.get_registry(BaseLRScheduler)
    if cfg.type in registry:
        scheduler = registry[cfg.type]
    elif hasattr(lr_scheduler, cfg.type):
        scheduler = getattr(lr_scheduler, cfg.type)
    else:
        raise ValueError(f"LR Scheduler {cfg.type} not known!")
    module = scheduler(optimizer, **cfg.kwargs)
    return module  # type: ignore
