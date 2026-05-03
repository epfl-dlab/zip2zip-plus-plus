"""Lightweight LoRA implementation for zip2zip finetuning."""

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Linear layer with low-rank adapter: y = Wx + (BAx) * scaling."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        base_layer.weight.requires_grad_(False)
        if base_layer.bias is not None:
            base_layer.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out


def apply_lora(model: nn.Module, rank: int, alpha: float = 1.0, target_modules: list[str] | None = None):
    """Replace target Linear layers in the decoder with LoRA-wrapped versions.

    Only applies to decoder layers (model.layers.*), not hyper_encoder.

    Args:
        model: The Zip2ZipLlama3Model (unwrapped).
        rank: LoRA rank.
        alpha: LoRA alpha for scaling.
        target_modules: Which linear layer names to wrap. Defaults to attention projections.

    Returns:
        Number of LoRA parameters added.
    """
    if target_modules is None:
        target_modules = ["attention.wq", "attention.wk", "attention.wv", "attention.wo"]

    lora_params = 0
    for layer in model.layers.values():
        for name in target_modules:
            parts = name.split(".")
            parent = layer
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]

            base_linear = getattr(parent, attr)
            if not isinstance(base_linear, nn.Linear):
                continue

            lora_layer = LoRALinear(base_linear, rank=rank, alpha=alpha)
            setattr(parent, attr, lora_layer)
            lora_params += rank * (base_linear.in_features + base_linear.out_features)

    return lora_params
