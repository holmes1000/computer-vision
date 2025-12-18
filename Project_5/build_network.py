"""
Samara Holmes
Spring 2025

Build and train a network to recognize digits using the MNIST database
"""
import torch
import torchvision
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# class definitions
class MyNetwork(nn.Module):
    def __init__(self, filter_size=5, pool_size=2, dropout_rate=0.25):
        """
        """
        super(MyNetwork, self).__init__()
        # A convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=filter_size)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=2)
        # A convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=filter_size)
        # A dropout layer with a dropout rate rate of your choice (anything from 5-50% is fine)
        self.dropout = nn.Dropout(p=dropout_rate)
        # A max pooling layer with a 2x2 window and a ReLU function applied
        self.pool2 = nn.MaxPool2d(kernel_size=pool_size, stride=2)
        # A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
        # self.fc1 = nn.Linear(20 * 4 * 4, 50)  # Assuming input images are MNIST 28x28

        # Need to do this after making filter_size and pool_size adjustable
        dim = 28  # MNIST image size
        dim = (dim - filter_size + 1) // pool_size  # After pool1
        dim = (dim - filter_size + 1) // pool_size  # After pool2

        _dim = 20 * dim * dim  # 20 channels output from conv2
        self.fc1 = nn.Linear(_dim, 50)

        # A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        """
        """
        # Pass through first convolutional layer, ReLU, and max pooling
        x = self.pool1(F.relu(self.conv1(x)))
        # print(f"Shape after conv1 and pool1: {x.shape}")
        # Pass through second convolutional layer, ReLU, dropout, and max pooling
        x = self.pool2(F.relu(self.conv2(x)))
        # print(f"Shape after conv2 and pool2: {x.shape}")
        x = self.dropout(x)
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # print(f"Flattened shape: {x.shape}")
        # Pass through the first fully connected layer and apply ReLU
        x = F.relu(self.fc1(x))
        # Pass through the final fully connected layer and apply log_softmax
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

def plot_MNIST(mnist_test):
    """
    Plot the first 6 MNIST digits
    """
    # Include a plot of the first six example digits from the test set in your report
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))
    for i in range(6):
        axes[i].imshow(mnist_test[i][0].squeeze(), cmap="gray")
        axes[i].set_title(f"Label: {mnist_test[i][1]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def train_network(dataloader, model, loss_fn, optimizer):
    """
    Train the neural network model
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    size = len(dataloader.dataset)
    model.train()
    acc = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return 100 * acc / size

def test_network(dataloader, model, loss_fn):
    """
    Test the neural network model
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = 100 * correct / size
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy


def plot_accuracies(train_accuracies, test_accuracies):
    """
    Plot training and testing accuracies over epochs.
    """
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, 'b-', label="Training Accuracy")
    plt.plot(epochs, test_accuracies, 'r-', label="Testing Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Testing Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()



def save_network(model, filepath):
    """
    Save the trained network to a file
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # Init hyperparams
    n_epochs = 20
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_accuracies = []
    test_accuracies = []

    # Get the dataset
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="./data/MNIST", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data/MNIST", train=False, download=True, transform=transform)     # Load the test set without shuffling

    # Plot the first 6 digits of the MNIST digit data test set
    plot_MNIST(mnist_test)

    # Create data loaders.
    train_dataloader = DataLoader(mnist_train, batch_size=batch_size_train)
    test_dataloader = DataLoader(mnist_test, batch_size=batch_size_test)

    # Build a network model
    model = MyNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # Train and test the model
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc = train_network(train_dataloader, model, loss_fn, optimizer)
        test_acc = test_network(test_dataloader, model, loss_fn)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    print("Done training and testing!")

    # Collect the accuracy scores and plot the training and testing accuracy in a graph
    plot_accuracies(train_accuracies, test_accuracies)

    # Save the network to a file
    save_network(model, filepath="mnist_model.pth")