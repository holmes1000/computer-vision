"""
Samara Holmes
Spring 2025

Examine your network and analyze how it processes the data. 
"""
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from build_network import *
import cv2

if __name__ == "__main__":
    # Read in the trained network as the first step and print the model
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    # Read an existing model from a file and load the pre-trained weights
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))

    # ------------- Analyze the first layer -------------------------
    # Access weights of the first convolutional layer
    weights = model.conv1.weight

    # Print weights and their shape
    print("Weights Shape:", weights.shape)
    print("Weights:", weights)

    # Show the effect of the filters
    # Set up a 3x4 grid for displaying the filters
    fig = plt.figure(figsize=(8, 6))

    for i in range(10):
        ax = fig.add_subplot(3, 4, i + 1)  # 3 rows, 4 columns
        ax.imshow(weights[i, 0].detach().numpy(), cmap="magma")  # Convert to NumPy for visualization
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_title(f"Filter {i + 1}")

    plt.show()

    # -------------- Show the effect of the filters ----------------

    with torch.no_grad():
    # put your code here
        weights = model.conv1.weight.detach().cpu().numpy()

    # Load the first training example from MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data/MNIST', train=False, transform=transform, download=True)

    first_image, _ = test_dataset[0]  # Get the first image and its label
    first_image = first_image.squeeze(0).numpy()

     # Apply filters to the first image

    filtered_images = []
    for i in range(10):  # First 10 filters
        kernel = weights[i, 0]  # Get the 2D filter for the first channel
        filtered_image = cv2.filter2D(first_image, -1, kernel)  # Apply OpenCV filter
        filtered_images.append((kernel, filtered_image))

    # Plot the filters and their corresponding filtered images
    fig, axes = plt.subplots(5, 4, figsize=(15, 10))  # Create 5 rows and 4 columns

    for i, (kernel, filtered_image) in enumerate(filtered_images):
        # Display the filter in the first column of the pair
        axes[i // 2, (i % 2) * 2].imshow(kernel, cmap='magma')  # Filter visualization
        axes[i // 2, (i % 2) * 2].set_xticks([])  # Remove x-axis ticks
        axes[i // 2, (i % 2) * 2].set_yticks([])  # Remove y-axis ticks

        # Display the filtered image in the second column of the pair
        axes[i // 2, (i % 2) * 2 + 1].imshow(filtered_image, cmap='gray')  # Filtered image visualization
        axes[i // 2, (i % 2) * 2 + 1].set_xticks([])  # Remove x-axis ticks
        axes[i // 2, (i % 2) * 2 + 1].set_yticks([])  # Remove y-axis ticks

    plt.tight_layout()
    plt.show()
