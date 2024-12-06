import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from model_definition import YutScoreModel
from dataset import YutDataset

def train_model(model, dataloader, epochs=50, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()  # Suitable for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions.float() == y_batch).sum().item()
            total += y_batch.size(0)
        
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
    
    return model

if __name__ == "__main__":
    # Load training data
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