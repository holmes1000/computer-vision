"""
Samara Holmes
Spring 2025

Evaluate the effect of changing different aspects of the network

The size of the convolution filters
The number of epochs of training
The batch size while training
The size of the pooling layer filters
"""
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from build_network import MyNetwork
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time  # For tracking time per epoch


def epoch_variation(num_epochs, model, test_loader, train_loader, criterion, optimizer):
    accuracy_per_epoch = []
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}...")
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()

            if batch_idx % 10 == 0:  # Log progress every 10 batches
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        # Set model to evaluation mode
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_per_epoch.append(accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # Plot the accuracy over epochs
    plt.plot(range(1, num_epochs + 1), accuracy_per_epoch, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy vs. Number of Epochs')
    plt.grid()
    plt.show()

def batch_size_variation(batch_sizes, num_epochs, model, test_data, training_data, criterion, optimizer):
    accuracy_for_batch_sizes = []
    for batch_size in batch_sizes:
        print(f"\nEvaluating Batch Size: {batch_size}")
        
        # Update data loaders with current batch size
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            print(f"  Starting Epoch {epoch + 1}/{num_epochs}...")
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                running_loss += loss.item()

        # Evaluate model performance
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_for_batch_sizes.append((batch_size, accuracy))
        print(f"Batch Size {batch_size}: Final Test Accuracy = {accuracy:.2f}%")
        
    batch_sizes, accuracies = zip(*accuracy_for_batch_sizes)  # Unpack batch sizes and accuracies
    plt.plot(batch_sizes, accuracies, marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Batch Size vs. Test Accuracy')
    plt.grid()
    plt.show()

def pool_size_variation(pool_sizes, num_epochs, test_data, training_data):
    accuracy_for_pool_sizes = []
    
    for pool_size in pool_sizes:
        print(f"\nEvaluating Pool Size: {pool_size}")
        
        # Initialize the model with the current pool size
        model = MyNetwork(filter_size=2, pool_size=pool_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Update data loaders
        train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                running_loss += loss.item()

            # Evaluate model performance
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_for_pool_sizes.append((pool_size, accuracy))
        print(f"Pool Size {pool_size}: Final Test Accuracy = {accuracy:.2f}%")

    # Plot pool size vs accuracy
    pool_sizes, accuracies = zip(*accuracy_for_pool_sizes)  # Unpack pool sizes and accuracies
    plt.plot(pool_sizes, accuracies, marker='o')
    plt.xlabel('Pool Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Pool Size vs. Test Accuracy')
    plt.grid()
    plt.show()

def filter_size_variation(filter_sizes, num_epochs, test_data, training_data, pool_size):
    accuracy_for_filter_sizes = []  
    for filter_size in filter_sizes:
        print(f"\nEvaluating Filter Size: {filter_size}")
        
        # Initialize the model with the current filter size
        model = MyNetwork(filter_size=filter_size, pool_size=pool_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Data loaders
        train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                running_loss += loss.item()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        # Calculate final accuracy after all epochs
        accuracy = 100 * correct / total
        accuracy_for_filter_sizes.append((filter_size, accuracy))
        print(f"Filter Size {filter_size}: Final Test Accuracy = {accuracy:.2f}%")

    # Plot filter size vs accuracy
    filter_sizes, accuracies = zip(*accuracy_for_filter_sizes)  # Unpack filter sizes and accuracies
    plt.plot(filter_sizes, accuracies, marker='o')
    plt.xlabel('Filter Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Filter Size vs. Test Accuracy')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 128
    filter_size = 3
    pool_size = 2

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Init the model
    model = MyNetwork(filter_size=filter_size, pool_size=pool_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Run for 10 epochs
    print("Running 10 Epochs")
    # epoch_variation(10, model, test_loader, train_loader, criterion, optimizer)

    num_epochs = 2
    batch_sizes = [64, 128, 256, 512]  # Batch sizes to evaluate
    # batch_size_variation(batch_sizes, num_epochs, model, test_data, training_data, criterion, optimizer)


    pool_sizes = [ 3, 4, 2]  # Pool sizes to evaluate
    # Evaluate and plot pool size vs accuracy
    # No need to take in model
    # pool_size_variation(pool_sizes, num_epochs, test_data, training_data)

    filter_sizes = [3, 5, 7]  # Filter sizes to evaluate
    filter_size_variation(filter_sizes, num_epochs, test_data, training_data, pool_size)


    

