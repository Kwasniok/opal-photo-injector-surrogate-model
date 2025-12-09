from abc import ABC, abstractmethod
from enum import Enum
import lightning as pl
import torch
import torch.nn as nn


class OptimizerTypes(str, Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"


class Model(pl.LightningModule, ABC):
    """
    Base model with given loss function and optimizer.

    note:
    Losses will be logged to "train/loss" and "val/loss" respectively.
    """

    def __init__(
        self,
        loss_fn,
        optimizer,
    ):
        super().__init__()

        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @classmethod
    def add_model_specific_args(cls, parser):
        return parser

    @classmethod
    def init_from_hparams(cls, hparams):
        """Initialize a new model based on"""
        return cls(**hparams)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test/loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        (x,) = batch
        return self(x)

    def configure_optimizers(self):
        return self.optimizer(self.parameters())


class MultiLayerLeakyReLUModel(Model):
    """
    Multi-layer fully connected feedforward neural network with LeakyReLU activations.

    loss_fn: Mean Squared Error
    optimizer: Stochastic Gradient Descent

    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        hidden_layer_sizes: list[int],
        leaky_relu_factor: float,
        learning_rate: float,
        weight_decay: float,
    ):

        loss_fn = nn.MSELoss()
        optimizer = lambda params: torch.optim.SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        self.save_hyperparameters(
            dict(
                input_shape=input_shape,
                output_shape=output_shape,
                hidden_layer_sizes=hidden_layer_sizes,
                leaky_relu_factor=leaky_relu_factor,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        )

        # activate each hidden + input layer with leaky ReLu
        # but not final layer
        seq = []
        if len(hidden_layer_sizes) == 0:
            seq += [
                nn.Linear(input_shape, output_shape),
            ]
        else:
            seq += [
                nn.Linear(input_shape, hidden_layer_sizes[0]),
                nn.LeakyReLU(leaky_relu_factor),
            ]
            for i, o in zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:]):
                seq += [
                    nn.Linear(i, o),
                    nn.LeakyReLU(leaky_relu_factor),
                ]
            seq += [
                nn.Linear(hidden_layer_sizes[-1], output_shape),
            ]
        self.sequence = nn.Sequential(*seq)

    @classmethod
    def add_model_specific_args(cls, parser):
        super().add_model_specific_args(parser)
        # model
        parser.add_argument("--hidden_layer_sizes", type=int, nargs="*", default=[])
        parser.add_argument("--leaky_relu_factor", type=float, default=0.1)
        # optimizer
        parser.add_argument("--learning_rate", type=float, default=0.01)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        return parser

    def forward(self, x):
        return self.sequence(x)
