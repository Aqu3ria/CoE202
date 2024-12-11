import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from model_definition import YutScoreModel  # Ensure this matches your model definition
from dataset import YutDataset  # Ensure you have this implemented
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_model(model, dataloader, epochs=50, lr=0.001):
    """
    Trains the neural network model.

    Parameters:
    - model (nn.Module): The neural network model to train.
    - dataloader (DataLoader): DataLoader for training data.
    - epochs (int): Number of training epochs.
    - lr (float): Learning rate.

    Returns:
    - nn.Module: The trained model.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 regularization

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()

            # Calculate accuracy
            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions.float() == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = correct / total * 100
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

    return model

def validate_model(model, dataloader):
    """
    Validates the neural network model.

    Parameters:
    - model (nn.Module): The trained neural network model.
    - dataloader (DataLoader): DataLoader for validation data.

    Returns:
    - None
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.float().squeeze())
            val_loss += loss.item()

            # Calculate accuracy
            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions.float() == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total * 100
    logging.info(f"Validation Loss: {val_loss/len(dataloader):.4f}, Validation Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    # Load balanced training data
    with open('training_data_balanced.pkl', 'rb') as f:
        data = pickle.load(f)
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.float32)
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Create datasets and dataloaders
    train_dataset = YutDataset(X_train, y_train)
    val_dataset = YutDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)

    # Initialize and train the model
    input_size = features.shape[1]  # Number of features
    hidden_size = 64  # Adjust based on experimentation
    num_features = input_size  # Assuming one weight per feature
    model = YutScoreModel(input_size=input_size, hidden_size=hidden_size, num_features=num_features)

    trained_model = train_model(model, train_loader, epochs=50, lr=0.001)
    
    # Validate the model
    validate_model(trained_model, val_loader)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'yut_score_model_balanced.pth')
    logging.info("Trained model saved as 'yut_score_model_balanced.pth'.")