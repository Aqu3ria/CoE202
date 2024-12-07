import torch
import torch.nn as nn
import torch.nn.functional as F

class YutScoreModel(nn.Module):
    def __init__(self, input_size = 5, hidden_size = 64, num_features = 5):
        super(YutScoreModel, self).__init__()
        # Learnable feature weights
        self.feature_weights = nn.Parameter(torch.ones(num_features))
        
        # Define the rest of the network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.dropout = nn.Dropout(p=0.5)  # Dropout for regularization
        self.fc2 = nn.Linear(hidden_size, 1)  # Output layer for binary classification

    def forward(self, x):
        # Apply feature weights
        weighted_x = x * self.feature_weights
        
        # Pass through the network
        out = F.relu(self.bn1(self.fc1(weighted_x)))
        out = self.dropout(out)
        out = self.fc2(out)
        return out