import numpy as np
import torch
import pytorch_lightning as pl


class NearestNeighbourModel(pl.LightningModule):
    """
    Simple nearest neighbour model.

    note: This model is used for comparison only and does not implement any training.
    """

    def __init__(self, x, y):
        super().__init__()
        assert len(x) == len(y), "Number of samples in x and y must be the same."
        self._xs = x
        self._ys = y

    @classmethod
    def add_model_specific_args(cls, parser):
        return parser

    @classmethod
    def init_from_hparams(cls, hparams):
        """Initialize a new model based on"""
        return cls(**hparams)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("NearestNeighbourModel does not implement training.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError(
            "NearestNeighbourModel does not implement validation."
        )

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("NearestNeighbourModel does not implement testing.")

    def predict_step(self, batch, batch_idx):
        (x,) = batch
        return self(x)

    def forward(self, xs):
        xs = xs.numpy()

        def f(x):
            idx = arg_nearest_neighbour(self._xs, x)
            return self._ys[idx]

        return torch.tensor(np.array([f(x) for x in xs]))


def arg_nearest_neighbour(xs: np.ndarray, x: np.ndarray) -> int:
    """Finds a nearest neighbour in an array for given point and returns its index."""
    assert x.shape == xs.shape[1:], "Input point has incorrect shape."
    dist = np.linalg.norm(xs - x, axis=1)
    idx = np.argmin(dist)
    return idx


class NearestNeighboursInterpolationModel(pl.LightningModule):
    """
    Simple k nearest neighbours (inverse distance weighted) interpolation model.

    note: This model is used for comparison only and does not implement any training.
    """

    def __init__(self, x, y, k: int = 3, epsilon: float = 1e-8):
        super().__init__()

        assert len(x) == len(y), "Number of samples in x and y must be the same."
        assert k > 0, "k must be positive."
        assert k <= len(x), "k must be less than or equal to number of samples."
        assert epsilon > 0, "epsilon must be positive."

        self._xs = x
        self._ys = y
        self._k = k
        self._epsilon = epsilon

    @classmethod
    def add_model_specific_args(cls, parser):
        return parser

    @classmethod
    def init_from_hparams(cls, hparams):
        """Initialize a new model based on"""
        return cls(**hparams)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("NearestNeighbourModel does not implement training.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError(
            "NearestNeighbourModel does not implement validation."
        )

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("NearestNeighbourModel does not implement testing.")

    def predict_step(self, batch, batch_idx):
        (x,) = batch
        return self(x)

    def forward(self, xs):
        xs = xs.numpy()

        def f(x):
            indices, weights = arg_nearest_neighbours(
                self._xs,
                x,
                k=self._k,
                epsilon=self._epsilon,
            )
            return np.sum(weights[:, None] * self._ys[indices], axis=0)

        return torch.tensor(np.array([f(x) for x in xs]))


def arg_nearest_neighbours(xs, x, k: int, epsilon: float = 1e-8):
    """Finds k nearest neighbours in an array for given point and returns their indicies + (inverse-distance) weights."""
    dist = np.linalg.norm(xs - x, axis=1)
    indices = np.argsort(dist)[:k]
    weights = dist[indices] + epsilon  # avoids division by zero
    weights = (1.0 / weights) / np.sum(1.0 / weights)
    return indices, weights
