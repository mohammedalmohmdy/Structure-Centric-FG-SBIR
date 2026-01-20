
import torch
import torch.nn.functional as F

def triplet_loss(anchor, pos, neg, margin=0.3):
    d_pos = F.pairwise_distance(anchor, pos)
    d_neg = F.pairwise_distance(anchor, neg)
    return torch.mean(F.relu(d_pos - d_neg + margin))
