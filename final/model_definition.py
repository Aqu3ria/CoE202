import torch
import torch.nn as nn
import torch.nn.functional as F

class YutScoreModel(nn.Module):
    def __init__(self):
        super(YutScoreModel, self).__init__()
        self.fc1 = nn.Linear(5, 64)  # 5 input features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output score

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Linear activation
        return x