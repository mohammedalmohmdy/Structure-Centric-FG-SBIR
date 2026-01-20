
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureGraphEncoder(nn.Module):
    """
    Section 3.2: Structure-Centric Graph Encoding
    Implements Eq.(3)-(6) in the paper.
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc_node = nn.Linear(in_dim, hidden_dim)
        self.fc_msg = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        nodes = feat_map.view(B, C, -1).permute(0, 2, 1)
        nodes = self.fc_node(nodes)  # Eq.(3)

        # adjacency via cosine similarity (Eq.4)
        norm = F.normalize(nodes, dim=-1)
        adj = torch.bmm(norm, norm.transpose(1, 2))

        msg = torch.bmm(adj, nodes) / (H * W)
        out = F.relu(self.fc_msg(msg))  # Eq.(6)
        return out, adj
