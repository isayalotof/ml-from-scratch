import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def load_iris_data(filepath):
    """Load Iris dataset from CSV"""
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv(filepath, names=columns)
    
    class_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data['class'] = data['class'].map(class_mapping)
    
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.astype(np.int64)
    
    return X, y


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    print("=== Neural Network Classifier (PyTorch) for Iris Dataset ===\n")
    
    # Load dataset
    X, y = load_iris_data('../data/iris.csv')
    print(f"Loaded {len(X)} samples with {X.shape[1]} features\n")
    
    # Split dataset (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Split dataset: {len(X_train)} train samples, {len(X_test)} test samples\n")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)
    
    # Create model (same architecture as C implementation: 4 -> 8 -> 3)
    model = NeuralNetwork(input_size=4, hidden_size=8, output_size=3)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    print("Created Neural Network: 4 -> 8 -> 3")
    print("Learning rate: 0.0100\n")
    
    # Training
    epochs = 1000
    losses = []
    
    print(f"Training neural network for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
    
    print("Training complete!\n")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        
        print("Evaluating neural network...")
        
        # Print first 10 predictions
        for i in range(min(10, len(X_test))):
            status = "[CORRECT]" if predicted[i] == y_test_tensor[i] else "[WRONG]"
            print(f"Sample {i}: Predicted={predicted[i].item()}, Actual={y_test_tensor[i].item()} {status}")
        
        accuracy = accuracy_score(y_test_tensor, predicted)
        correct = int(accuracy * len(y_test))
        print(f"\nAccuracy: {accuracy*100:.2f}% ({correct}/{len(y_test)} correct)\n")
    
    # Plot training loss
    print("=== Generating training loss plot ===")
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=1, alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Neural Network Training Loss (PyTorch)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('../results/nn_pytorch_loss.png', dpi=300, bbox_inches='tight')
    print("Saved plot to: ../results/nn_pytorch_loss.png\n")
    
    # Detailed classification report
    print("=== Classification Report ===")
    print(classification_report(y_test_tensor, predicted, 
                                target_names=['Setosa', 'Versicolor', 'Virginica']))
    
    print("=== Completed successfully! ===")


if __name__ == "__main__":
    main()