import random
import numpy as np
import torch
import yaml

def config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_for(name):
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else 'cpu' if name == 'auto' else name)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
