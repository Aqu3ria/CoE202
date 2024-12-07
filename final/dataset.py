import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class YutDataset(Dataset):
    def __init__(self, features, labels):
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]