from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
from utils.augmenter import apply_augmentation_pipeline

class avesDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.samples = []
            self.label_map = {}

            for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
                if not class_dir.is_dir():
                    continue

                self.label_map[class_dir.name] = idx

                for file in class_dir.glob("*.npy"):
                    self.samples.append((file, idx))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            file_path, label = self.samples[idx]

            spec = np.load(file_path) 

            # Normalização
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)

            spec = torch.tensor(spec, dtype=torch.float32)
            
            # Aplica transformações (Augmentation) apenas se definido
            if self.transform:
                spec = self.transform(spec)

            spec = spec.unsqueeze(0) 
            return spec, label

class SpectrogramAugment:
    """
    Wrapper que integra o augmenter.py ao pipeline do PyTorch.
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        has_channel = tensor.ndim == 3
        arr = tensor.squeeze(0).numpy() if has_channel else tensor.numpy()

        arr = apply_augmentation_pipeline(arr)

        result = torch.tensor(arr, dtype=torch.float32)
        return result.unsqueeze(0) if has_channel else result

class TransformedSubset(Dataset):
    """
    Wrapper que aplica um transform apenas em um Subset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
        
