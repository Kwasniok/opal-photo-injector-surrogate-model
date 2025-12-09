import numpy as np
import lightning as pl


class LossRatioMonitor(pl.Callback):
    """
    Callback to monitor the ratio of (smoothed) validation to training loss.

    optional: Early Stopping on persistent high ratio values.

    note:
    - Intended for estimating the generalization of the model during training.
    - Smoothing via averaging over a sliding window.
    """

    def __init__(
        self,
        training_window_size: int = 1,
        validation_window_size: int = 1,
        ratio_upper_threshold: float | None = None,
        bad_epochs_limit: int = 1,
    ):
        super().__init__()
        self.training_window_size: int = training_window_size
        self.validation_window_size: int = validation_window_size
        self.ratio_upper_threshold: float | None = ratio_upper_threshold
        self.bad_epochs_limit: int = bad_epochs_limit
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.bad_epochs: int = 0

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train/loss")
        if train_loss is not None:
            self.append_train_loss(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            self.append_val_loss(val_loss.item())

        self.process_ratio(trainer)

    def append_train_loss(self, loss):
        """Append a new training loss to the buffer."""
        self.train_losses.append(loss)
        if len(self.train_losses) > self.training_window_size:
            self.train_losses.pop(0)

    def append_val_loss(self, loss):
        """Append a new validation loss to the buffer."""
        self.val_losses.append(loss)
        if len(self.val_losses) > self.validation_window_size:
            self.val_losses.pop(0)

    def process_ratio(self, trainer):
        """Log the current ratio of validation to training loss."""
        if self.train_losses and self.val_losses:
            smooth_val = np.mean(self.val_losses)
            smooth_train = np.mean(self.train_losses)
            ratio = smooth_val / smooth_train
            trainer.logger.experiment.add_scalar(
                "val_train_loss_ratio", ratio, trainer.current_epoch
            )

            # early stopping
            if (
                self.ratio_upper_threshold is not None
                and ratio > self.ratio_upper_threshold
            ):
                self.bad_epochs += 1
            else:
                self.bad_epochs = 0

            if self.bad_epochs >= self.bad_epochs_limit:
                print(
                    f"Stopping training: val/train loss ratio (currently at {ratio:.2f}) exceeded threshold of {self.ratio_upper_threshold} for {self.bad_epochs_limit} epochs."
                )
                trainer.should_stop = True
