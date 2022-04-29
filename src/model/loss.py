from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Tuple

from tqdm import tqdm
import torch
import torch.nn.functional as F


class Loss():
    def __init__(self,
                 label_key: str = 'labels',
                 filter_key: str = None,  # Do not apply loss if this target is true
                 filter_value: Any = None,  # Do not apply loss if this target is true
                 device: str = 'cpu') -> None:
        self.label_key = label_key
        self.device = device
        self.filter_key = filter_key
        self.filter_value = filter_value

    def _binary(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 1:
            target = torch.transpose(torch.stack([1-target, target]), 0, 1).type(torch.FloatTensor)
        return target.to(self.device)

    def _get_filter_mask(self, target):
        if self.filter_key is not None:
            return target[self.filter_key] == self.filter_value
        return [True] * target[self.label_key].shape[0]

    @abstractmethod
    def __call__(self, output, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError("__call__ not implemented...")


class BCELoss(Loss):
    loss_name = "BCELoss"

    def __call__(self, output, target) -> torch.Tensor:
        if isinstance(output, tuple):
            output = output[0]
        mask = self._get_filter_mask(target)
        if sum(mask) == 0:
            return 0.0  # Avoid NaN when all elements in batch are filtered out
        output = output[mask, :]
        target = self._binary(target[self.label_key])[mask, :]
        loss = F.binary_cross_entropy_with_logits(output, target)
        return loss, {self.loss_name: loss.item()}


class WeightedBCELoss(Loss):
    loss_name = "WeightedBCELoss"

    def __init__(self,
                 weight_key: str = 'weights',
                 label_key: str = 'labels',
                 filter_key: str = None,  # Do not apply loss if this target is true
                 filter_value: Any = None,  # Do not apply loss if this target is true
                 device: str = 'cpu') -> None:
        super().__init__(label_key, filter_key, filter_value, device)
        self.weight_key = weight_key

    def __call__(self, output, target):
        if isinstance(output, tuple):
            output = output[0]
        mask = self._get_filter_mask(target)
        if sum(mask) == 0:
            return 0.0  # Avoid NaN when all elements in batch are filtered out
        weights = torch.reshape(target[self.weight_key], (output.shape[0], 1))[mask].to(self.device)
        output = output[mask, :]
        target = self._binary(target[self.label_key])[mask, :]
        loss = F.binary_cross_entropy_with_logits(output, target, weight=weights)
        return loss, {self.loss_name: loss.item()}


class HiddenL2Loss(Loss):
    loss_name = "HiddenL2Loss"

    def __init__(self,
                 label_key: str = 'embeddings',
                 output_component: int = 1,
                 filter_key: str = None,  # Do not apply loss if this target is true
                 filter_value: Any = None,  # Do not apply loss if this target is true
                 device: str = 'cpu') -> None:
        super().__init__(label_key, filter_key, filter_value, device)
        self.output_component = output_component

    def __call__(self, output, target):
        if isinstance(output, tuple):
            output = output[self.output_component]
        mask = self._get_filter_mask(target)
        if sum(mask) == 0:
            return 0.0  # Avoid NaN when all elements in batch are filtered out
        output = output[mask, :]
        target = target[self.label_key][mask, :].to(self.device)
        loss = F.mse_loss(output, target)
        return loss, {self.loss_name: loss.item()}


class EWCLoss(object):
    loss_name = "EWCLoss"

    def __init__(self,
                 dataloader: torch.utils.data.DataLoader,
                 task_loss: Loss,
                 model: torch.nn.Module = None,
                 device: str = 'cpu') -> None:
        self.dataloader = dataloader
        self.task_loss = task_loss
        self.device = device

        self._diag_fisher_matrix = None
        self.original_params = None

        if model is not None:
            self.set_model(model)

    def _diag_fisher(self, model: torch.nn.Module):
        model.eval()
        diag_fisher_matrix = {n: 0.0 for n, _ in model.named_parameters()}
        for data, target in tqdm(self.dataloader):
            # Prepare weights
            target["labels"] = target["labels"].to(self.device)
            target["weights"] = target["weights"].to(self.device)
            data = {k: t.to(self.device) for k, t in data.items()}

            model.zero_grad()
            output = model(**data, cls_head_id=target["tasks"])
            loss, _ = self.task_loss(output, target)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # Compute average as expectation of derivative
                    diag_fisher_matrix[n] += p.grad.detach() ** 2 / len(self.dataloader)

        return diag_fisher_matrix

    def set_model(self, model: torch.nn.Module) -> None:
        self.original_params = {n: deepcopy(p) for n, p in model.named_parameters() if p.requires_grad}
        self._diag_fisher_matrix = self._diag_fisher(model)

    def __call__(self, model: torch.nn.Module) -> torch.tensor:
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += torch.sum(self._diag_fisher_matrix[n] * (p - self.original_params[n]) ** 2)
        return loss, {self.loss_name: loss.item()}


class AdditiveMultiLoss():
    def __init__(self, loss_configs, device='cpu') -> None:
        self.losses = []
        self.lambdas = []
        for config in loss_configs:
            self.losses.append(globals()[config["type"]](**config["args"], device=device))
            self.lambdas.append(config["lambda"])

    def add_loss(self, loss: Loss, loss_importance: float) -> None:
        self.losses.append(loss)
        self.lambdas.append(loss_importance)

    def __call__(self, output, target):
        all_losses = [loss(output, target) for loss in self.losses]
        loss_components = {}
        for _, loss in all_losses:
            loss_components.update(loss)
        return sum([l * loss[0] for loss, l in zip(all_losses, self.lambdas)]), loss_components
