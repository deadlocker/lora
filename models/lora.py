#!/usr/bin/env python3

import torch
import torch.nn as nn

'''
This function injects LORA into specific target layers of a model
'''
def inject_lora(model, rank, alpha, targets, dropout):
    for name, module in list(model.named_modules()):
        is_target = any(target in name for target in targets)
        is_linear = isinstance(module, nn.Linear) or 'Conv1D' in type(module).__name__

        if is_target and is_linear:
            name_parts = name.rsplit('.', 1)
            p_path = name_parts[0] if len(name_parts) > 1 else ''
            l_name = name_parts[-1]
            p_module = model
            if p_path:
                for part in p_path.split('.'):
                    p_module = getattr(p_module, part)

            lora_wrapper = LinearWithLora(module, rank, alpha, dropout)
            setattr(p_module, l_name, lora_wrapper)

'''
LORA layer implementation for the model
'''
class LORALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout):
        super().__init__()
        self.rank = rank
        self.scaling = float(alpha) / rank # scaling variance
        self.std_dev = 1 / torch.sqrt(torch.tensor(rank).float())

        self.A = nn.Parameter(torch.randn(in_features, rank) * self.std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.scaling * self.dropout(x @ self.A @ self.B)
        return x


'''
Linear Layer for LORA wrapper
'''
class LinearWithLora(nn.Module):
    def __init__(self, linear, rank, alpha, dropout):
        super().__init__()
        self.linear = linear
        if isinstance(linear, nn.Linear):
            in_features = linear.in_features
            out_features = linear.out_features
        else:
            in_features = linear.weight.shape[0]
            out_features = linear.weight.shape[1]

        # weight freezing
        self.linear.weight.requires_grad = False
        if hasattr(self.linear, "bias") and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_layer = LORALayer(
            in_features,
            out_features,
            rank,
            alpha,
            dropout
        )

    def forward(self, x):
        return self.linear(x) + self.lora_layer(x)
