from train_model import train_model
from data_collection import generate_training_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from model_definition import YutScoreModel
from dataset import YutDataset

for i in range(10):
    generate_training_data()
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    features = data['features']
    labels = data['labels']
    
    # Create dataset and dataloader
    dataset = YutDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train the model
    model = YutScoreModel()
    trained_model = train_model(model, dataloader, epochs=50, lr=0.001)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'yut_score_model.pth')
    print("Model trained and saved as 'yut_score_model.pth'.")
