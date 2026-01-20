
import yaml, torch
from models.full_model import FullModel
from utils.metrics import recall_at_k

cfg = yaml.safe_load(open('configs/sketchy.yaml'))
model = FullModel(cfg['embedding_dim'])
print("Recall@1:", recall_at_k(None))
