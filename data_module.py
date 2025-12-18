import os
import json
import numpy as np
import torch
import lightning as pl
from torch.utils.data import TensorDataset, Subset, DataLoader


class NPZTensorDataModule(pl.LightningDataModule):
    """
    Loads a tensor dataset from .npz files.

    Expects the following files in the given path:
    - train.npz : for training & validation data
    - test.npz : for testing data
    - predict.npz : for prediction data (might not contain targets)


    Each data file is expected to be a collection of d-dimensional numpy array of shape (num_samples, *sample_shape):
    - x: input
    - y: target (if applicable)

    """

    def __init__(
        self,
        path,
        *,
        batch_size: int,
        train_val_split: float = 0.8,
        input_transform=None,
        target_transform=None,
        num_workers: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ):
        """
        Initialize the data module.

        Args:
        path (str): Path to the directory containing the .npz files.
        batch_size (int): Batch size for the data loaders.
        train_val_split (float): Proportion of training data to use for training (rest for validation).
        input_transform (callable, optional): A transformation to apply to the inputs.
        target_transform (callable, optional): A transformation to apply to the targets.
        num_workers (int, optional): Number of subprocesses to use for data loading.
        persistent_workers (bool): Whether to keep data loader workers alive between epochs.
        pin_memory (bool): Whether to pin memory in data loaders.
        """
        super().__init__()

        assert 0.0 < train_val_split < 1.0, "train_val_split must be in (0, 1)"

        self._path = path
        self._input_transform = input_transform
        self._target_transform = target_transform
        self._batch_size = batch_size
        self._train_val_split = train_val_split
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory
        self._train_dataset: TensorDataset | None = None
        self._val_dataset: TensorDataset | None = None
        self._test_dataset: TensorDataset | None = None
        self._predict_dataset: TensorDataset | None = None

        self._input_shape: tuple[int, ...] | None = None
        self._output_shape: tuple[int, ...] | None = None

    @property
    def input_shape(self) -> tuple[int, ...]:
        if self._input_shape is None:
            self._cache_shapes()
        return self._input_shape or ()

    @property
    def output_shape(self) -> tuple[int, ...]:
        if self._output_shape is None:
            self._cache_shapes()
        return self._output_shape or ()

    def _cache_shapes(self):
        shapes = _get_npy_shapes(os.path.join(self._path, "test.npz"))
        self._input_shape = shapes["x"][1:]
        self._output_shape = shapes["y"][1:]

    def setup(self, stage: str = "fit"):

        match stage:
            case "fit":
                self._setup_train_val()
            case "test":
                self._setup_test()
            case "predict":
                self._setup_predict()
            case _:
                # unclear stage, setup training & validation by default
                self._setup_train_val()

    def _setup_train_val(self):
        if self._train_dataset is not None and self._val_dataset is not None:
            # already setup
            return

        # load train + val
        train_dataset = self._load_xy("train")
        # split
        generator = torch.Generator().manual_seed(2025)
        train_size = int(self._train_val_split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self._train_dataset, self._val_dataset = map(
            _consolidate,
            torch.utils.data.random_split(
                train_dataset, [train_size, val_size], generator=generator
            ),
        )

    def _setup_test(self):
        if self._test_dataset is not None:
            # already setup
            return

        self.test_dataset = self._load_xy("test")

    def _setup_predict(self):
        if self._predict_dataset is not None:
            # already setup
            return

        # note: Does not load y!
        self.predict_dataset = self._load_x("predict")

    def train_dataloader(self):
        if self._train_dataset is None:
            raise RuntimeError(
                "The data module has not been setup for training yet. Call `setup('fit')` first."
            )

        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers or 0,
            persistent_workers=self._persistent_workers,
            pin_memory=self._pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        if self._val_dataset is None:
            raise RuntimeError(
                "The data module has not been setup for validation yet. Call `setup('fit')` first."
            )

        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers or 0,
            persistent_workers=self._persistent_workers,
            pin_memory=self._pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError(
                "The data module has not been setup for testing yet. Call `setup('test')` first."
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers or 0,
            persistent_workers=self._persistent_workers,
            pin_memory=self._pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        if self.predict_dataset is None:
            raise RuntimeError(
                "The data module has not been setup for prediction yet. Call `setup('predict')` first."
            )

        return DataLoader(
            self.predict_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers or 0,
            persistent_workers=self._persistent_workers,
            pin_memory=self._pin_memory,
            shuffle=False,
        )

    def _load_xy(self, tag) -> TensorDataset:
        data = np.load(os.path.join(self._path, f"{tag}.npz"))

        x = torch.tensor(data["x"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float32)

        if self._input_transform is not None:
            x = self._input_transform(x)
        if self._target_transform is not None:
            y = self._target_transform(y)

        return TensorDataset(x, y)

    def _load_x(self, tag) -> TensorDataset:
        data = np.load(os.path.join(self._path, f"{tag}.npz"))

        x = torch.tensor(data["x"], dtype=torch.float32)

        if self._input_transform is not None:
            x = self._input_transform(x)

        return TensorDataset(x)


def _consolidate(subset: Subset) -> TensorDataset:
    """
    Convert Subset into TensorDataset by consolidating the memory.
    note: Copies all elements into a new dataset.
    """
    xs, ys = zip(*[subset[i] for i in range(len(subset))])
    return TensorDataset(torch.stack(xs), torch.stack(ys))


def _get_npy_shapes(path):
    """
    Get the shapes of arrays stored in a .npz file.
    Args:
        path (str): Path to the .npz file.
    Returns:
        dict: A dictionary with keys as array names and values as their shapes.
    """
    sample = np.load(path, mmap_mode="r")
    shapes = {}
    for key in sample.files:
        shapes[key] = sample[key].shape
    return shapes
