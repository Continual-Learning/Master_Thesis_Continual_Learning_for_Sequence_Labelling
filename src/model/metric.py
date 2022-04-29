import typing as t
from os import PathLike

import pandas as pd
import torch


class ClassificationMetrics():
    """Class to keep track of confusion matrix to calculate classification metrics in batches.

    Handles also metrics that cannot be batch-averaged (e.g. recall, which is not computed over all samples in a batch).
    """
    def __init__(self, subplot="", writer=None):
        self.metrics_df = []
        self._tot_loss = {}
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0
        self.n_updates = 0
        self.writer = writer
        self.subplot = subplot

    def log_to_writer(self):
        if self.writer is not None:
            for metric, value in self.metrics().items():
                self.writer.add_scalar(metric, value)

    def update(self, loss_components: t.Dict, output: torch.Tensor, target: torch.Tensor, log: bool = True) -> None:
        """Updates confusion matrix given model outputs and expected values.
        """
        if isinstance(output, tuple):
            output = output[0]
        for loss_name, loss_item in loss_components.items():
            self._tot_loss[loss_name] = self._tot_loss.get(loss_name, 0.0) + loss_item
        self.n_updates += 1
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            self._tp += torch.sum((pred == target) * (target == 1)).item()
            self._tn += torch.sum((pred == target) * (target == 0)).item()
            self._fp += torch.sum((pred != target) * (target == 0)).item()
            self._fn += torch.sum((pred != target) * (target == 1)).item()

        if log:
            self.log_to_writer()

    def reset(self) -> None:
        """Resets the confusion matrix to evaluate e.g. a new epoch. Retains old information to store to file.
        """
        self._tot_loss = {}
        self.n_updates = 0
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0

    def __metrics(self, loss: t.Dict={}, acc=0.0, prec=0.0, rec=0.0, f1=0.0):
        losses = {f"{name}{self.subplot}": value for name, value in loss.items()}
        return {
            **losses,
            f"acc{self.subplot}": acc,
            f"prec{self.subplot}": prec,
            f"rec{self.subplot}": rec,
            f"f1_score{self.subplot}": f1
        }

    def metrics(self) -> dict:
        """Get current values of classification metrices."""
        if self.n_updates == 0:
            return self.__metrics()
        precision = self._tp / (self._tp + self._fp) if self._tp > 0 else 0
        recall = self._tp / (self._tp + self._fn) if self._tp > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if self._tp > 0 else 0
        loss = {name: value / self.n_updates for name, value in self._tot_loss.items()}
        return self.__metrics(
            loss=loss,
            acc=(self._tp + self._tn) / (self._tp + self._tn + self._fp + self._fn),
            prec=precision,
            rec=recall,
            f1=f1_score
        )

    def metric_name(self, name: str) -> str:
        """Returns the task-specific name used for a metric"""
        if name in self.metrics():
            raise KeyError("Metric not found among those computed by class.")
        return f"{name}{self.subplot}"

    def save_metrics(self, step: t.Any) -> None:
        """Stores e.g. end of epoch metrics to be dumped to file."""
        self.metrics_df.append({
            "step": step,
            **self.metrics()
        })

    def dump_metrics(self, filename: PathLike) -> None:
        """Saves currently stored metrics to file."""
        pd.DataFrame(self.metrics_df).to_csv(filename, index=False)

    def __getitem__(self, item):
        return self.metrics()[f"{item}{self.subplot}"]
