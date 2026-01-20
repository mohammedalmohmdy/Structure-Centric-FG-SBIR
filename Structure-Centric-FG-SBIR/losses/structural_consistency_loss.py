
import torch

class StructuralConsistencyLoss(torch.nn.Module):
    """
    Enforces topology preservation between sketch and photo graphs.
    """
    def forward(self, adj_s, adj_p):
        return torch.mean((adj_s - adj_p) ** 2)
