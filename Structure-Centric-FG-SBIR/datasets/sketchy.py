
import torch
from torch.utils.data import Dataset
import random

class SketchyDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.length = 20

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sketch = torch.randn(3,224,224)
        photo = torch.randn(3,224,224)
        return sketch, photo
