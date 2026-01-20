
import torch
from torch.utils.data import DataLoader
from models.full_model import StructureCentricFGSBIR
from datasets.sketchy import SketchyDataset
from losses.triplet_loss import triplet_loss

model = StructureCentricFGSBIR()
dataset = SketchyDataset()
loader = DataLoader(dataset, batch_size=2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2):
    for sketch, photo in loader:
        emb = model(sketch, photo)
        loss = emb.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} finished")
