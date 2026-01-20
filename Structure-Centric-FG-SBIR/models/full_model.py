
import torch
import torch.nn as nn
from .backbone import ResNetBackbone
from .graph_encoder import StructureGraphEncoder
from .uncertainty_weighting import UncertaintyWeighting
from .alignment_module import GraphAlignment

class StructureCentricFGSBIR(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=256):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.graph = StructureGraphEncoder(feat_dim, hidden_dim)
        self.uncertainty = UncertaintyWeighting(hidden_dim)
        self.align = GraphAlignment(hidden_dim)

    def forward(self, sketch, photo):
        sk_feats = self.backbone(sketch)[-1]
        ph_feats = self.backbone(photo)[-1]

        sk_nodes, _ = self.graph(sk_feats)
        ph_nodes, _ = self.graph(ph_feats)

        sk_nodes, _ = self.uncertainty(sk_nodes)
        ph_nodes, _ = self.uncertainty(ph_nodes)

        emb = self.align(sk_nodes, ph_nodes)
        return emb
