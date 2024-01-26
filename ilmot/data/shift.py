"""Load and convert shift labels"""
from scalabel.label.io import load
from scalabel.label.typing import Dataset

from ilmot.data.datasets.base import BaseDatasetLoader


class Shift(BaseDatasetLoader):  # type: ignore
    """BDD100K dataloading class."""

    def load_dataset(self) -> Dataset:
        """Convert BDD100K annotations to Scalabel format and prepare them."""
        assert self.annotations is not None
        return load(
            self.annotations,
            nprocs=self.num_processes,
        )
