"""Data loading for the Not Hotdog project.

This is exactly the code you wrote in the Phase 2 notebook, tidied into one place
so the later notebooks can `from data import get_dataloaders` instead of repeating it.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from datasets import load_dataset

# ImageNet normalization constants (standard for pretrained models in Phase 4).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training transform: random augmentation to fight memorization.
train_tf = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

# Eval transform (val + test): deterministic, no randomness.
eval_tf = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class HotdogDataset(Dataset):
    """Wraps a Hugging Face split + a transform, optionally over a subset of indices."""

    def __init__(self, hf_split, transform, indices=None):
        self.hf = hf_split
        self.transform = transform
        self.indices = indices if indices is not None else list(range(len(hf_split)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ex = self.hf[self.indices[i]]
        image = ex["image"].convert("RGB")
        return self.transform(image), ex["label"]


def get_dataloaders(batch_size=32, seed=42):
    """Return (train_loader, val_loader, test_loader, classes).

    Same 160 / 40 / 50 split as the Phase 2 notebook.
    """
    raw = load_dataset("truepositive/hotdog_nothotdog")
    classes = raw["train"].features["label"].names

    test_hf = raw["validation"]   # held-out test set
    full_hf = raw["train"]        # split into train + val

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_hf), generator=g).tolist()
    val_idx, train_idx = perm[:40], perm[40:]

    train_ds = HotdogDataset(full_hf, train_tf, train_idx)
    val_ds = HotdogDataset(full_hf, eval_tf, val_idx)
    test_ds = HotdogDataset(test_hf, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, classes


def denormalize(t):
    """Undo Normalize so a tensor can be shown with matplotlib."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0)
