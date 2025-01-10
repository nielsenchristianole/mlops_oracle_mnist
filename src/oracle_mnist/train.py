import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Module
from typing import Optional
from model import create_model


def train_model(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    num_classes: int = 10,
) -> None:
    """
    Trains the model on the dataset.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (Optional[DataLoader]): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        num_classes (int): Number of output classes for classification.
    """
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Module = create_model(num_classes=num_classes)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation step
        if val_loader:
            validate_model(model, val_loader, device)

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth.")


def validate_model(model: Module, val_loader: DataLoader, device: torch.device) -> None:
    """
    Validates the model on the validation dataset.

    Args:
        model (Module): Trained PyTorch model.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run validation on.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # Replace with actual DataLoader instances
    train_loader: DataLoader = ...  # TODO: Load training data
    val_loader: Optional[DataLoader] = None  # TODO: Load validation data if available

    train_model(train_loader, val_loader)
