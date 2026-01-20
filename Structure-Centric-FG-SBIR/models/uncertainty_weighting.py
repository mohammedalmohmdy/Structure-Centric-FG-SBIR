
import torch
import torch.nn as nn

class UncertaintyWeighting(nn.Module):
    """
    Section 3.3: Uncertainty-Aware Structural Weighting
    """
    def __init__(self, dim):
        super().__init__()
        self.var_head = nn.Linear(dim, 1)

    def forward(self, nodes):
        log_var = self.var_head(nodes)
        weight = torch.exp(-log_var)  # confidence weighting
        return nodes * weight, weight
