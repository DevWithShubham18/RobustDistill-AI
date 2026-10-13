"""Original 8x8 sklearn-digits models, data split, and gradient attacks."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 2026
MODELS = ("plain", "kd", "robust_kd", "adv_student")


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(64, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 10))

    def forward(self, x):
        return self.layers(x)


def make_model(name: str) -> MLP:
    return MLP(128 if name == "teacher" else 48)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


@dataclass
class DataBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def get_data(seed: int = SEED) -> DataBundle:
    data = load_digits()
    x = torch.tensor(data.images / 16.0, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(data.target, dtype=torch.long)
    indices = np.arange(len(y))
    itrain, itest = train_test_split(indices, test_size=0.20, random_state=seed, stratify=y)
    itrain, ival = train_test_split(itrain, test_size=0.20, random_state=seed, stratify=y[itrain])
    return DataBundle(x[itrain], y[itrain], x[ival], y[ival], x[itest], y[itest],
        np.array(itrain), np.array(ival), np.array(itest))


def pgd(model, x, y, epsilon=0.15, steps=7, step_size=None, random_start=True):
    """Untargeted L-infinity PGD, maximizing CE in [0,1] input space."""
    if epsilon < 0 or steps < 1:
        raise ValueError("epsilon must be >=0 and steps >=1")
    step_size = epsilon / max(1, steps // 2) if step_size is None else step_size
    original = x.detach()
    if random_start and epsilon:
        adv = (original + torch.empty_like(original).uniform_(-epsilon, epsilon)).clamp(0, 1)
    else:
        adv = original.clone()
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(adv), y)
            grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + step_size * grad.sign()
        adv = torch.maximum(torch.minimum(adv, original + epsilon), original - epsilon).clamp(0, 1)
    return adv.detach()


def fgsm(model, x, y, epsilon=0.15):
    return pgd(model, x, y, epsilon=epsilon, steps=1, step_size=epsilon, random_start=False)
