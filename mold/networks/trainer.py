from __future__ import annotations

import time
from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn

from mold.detector import Detector
from mold.networks.base_model import BaseModel


class Trainer(BaseModel):
    """Trainer class for the MoLD detector."""

    def __init__(self, opt: Namespace) -> None:
        super().__init__(opt)
        self.opt = opt
        self.model = Detector(opt.arch)

        # Initialize FC layers.
        nn.init.normal_(self.model.fc1.weight.data, 0.0, opt.init_gain)
        nn.init.normal_(self.model.fc2.weight.data, 0.0, opt.init_gain)
        nn.init.normal_(self.model.fc3.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone:
            weights_to_train = [
                "fc1.weight",
                "fc1.bias",
                "fc2.weight",
                "fc2.bias",
                "fc3.weight",
                "fc3.bias",
            ]
            params: list[nn.Parameter] = []
            for name, p in self.model.named_parameters():
                if name in weights_to_train:
                    print("param name:", name)
                    params.append(p)
                else:
                    p.requires_grad = False
        else:
            print(
                "Your backbone is not fixed. Are you sure you want to proceed? "
                "If this is a mistake, enable the --fix_backbone command during "
                "training and rerun"
            )
            self.model.vit.class_embedding.requires_grad = False
            self.model.vit.positional_embedding.requires_grad = False
            params = []
            for name, p in self.model.named_parameters():
                if "positional" in name:
                    p.requires_grad = False
            time.sleep(3)
            params = list(self.model.vit.transformer.parameters()) + list(
                self.model.fc.parameters()
            )

        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=opt.lr,
                momentum=0.0,
                weight_decay=opt.weight_decay,
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model.to(opt.gpu_ids[0])

        # Runtime attributes.
        self.input: torch.Tensor
        self.label: torch.Tensor
        self.output: torch.Tensor
        self.loss: torch.Tensor

    @property
    def name(self) -> str:
        return "Trainer"

    def adjust_learning_rate(self, min_lr: float = 1e-6) -> bool:
        """Reduce learning rate by a factor of 10.

        Returns:
            True if learning rate was adjusted, False if below minimum.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def set_input(self, input: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Set the input data and labels."""
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self) -> None:
        """Forward pass through the model."""
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self) -> torch.Tensor:
        """Compute the loss."""
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self) -> None:
        """Forward pass, compute loss, and update parameters."""
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
