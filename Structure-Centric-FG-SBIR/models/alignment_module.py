
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAlignment(nn.Module):
    """
    Section 3.4: Structure-to-Structure Alignment
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, sketch_nodes, photo_nodes):
        Q = self.query(sketch_nodes)
        K = self.key(photo_nodes)
        V = self.value(photo_nodes)

        attn = torch.softmax(torch.bmm(Q, K.transpose(1,2)) / (Q.size(-1)**0.5), dim=-1)
        aligned = torch.bmm(attn, V)
        return aligned.mean(dim=1)
