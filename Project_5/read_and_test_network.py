"""
Samara Holmes
Spring 2025

Read the network and run it on the test set
"""
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from build_network import MyNetwork
import os
from PIL import Image, ImageOps


def evaluate_handwriting(model):
    image_folder = './data/handwriting_2'
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),        # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to match MNIST preprocessing
    ]) 

    image_tensors = []
    file_names = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):  # Ensure only valid image files are processed
            file_names.append(filename)
            # print(filename)
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = ImageOps.invert(image) # Invert the colors
            # image_tensor = transform(image)
            image_tensor = transform(image).unsqueeze(0)
            image_tensors.append(image_tensor)

    # Combine all tensors into one batch
    images_batch = torch.cat(image_tensors)
    # images_batch = torch.stack(image_tensors)
    print(f"Loaded batch shape: {images_batch.shape}")

    # Create the DataLoader for the test dataset
    dataset = torch.utils.data.TensorDataset(images_batch)
    test_loader = torch.utils.data.DataLoader(images_batch, batch_size=10, shuffle=False)

    # images = next(iter(test_loader))
    images = next(iter(test_loader))

    # Run the model on the batch
    outputs = model(images)
    predicted_labels = outputs.argmax(dim=1).cpu().numpy()  # Predictions
    print(len(predicted_labels))
    # Print predictions for each image
    for i, filename in enumerate(file_names):
        print(f"{filename}: Predicted Label - {predicted_labels[i]}")
        output_values = outputs[i].detach().numpy()
        max_index = output_values.argmax()
        correct_label = predicted_labels[i].item()
        
        print(f"Example {i + 1}:")
        print(f"Network output values: {[round(val, 2) for val in output_values]}")
        print(f"Index of max output value: {max_index}")
        print(f"Correct label: {correct_label}")
        print()

    visualize_3x3_grid(images_batch, outputs)

def visualize_3x3_grid(images, outputs):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        axes[i // 3, i % 3].imshow(images[i].squeeze(), cmap='gray')
        axes[i // 3, i % 3].set_title(f"Pred: {outputs[i].detach().numpy().argmax()}")
        axes[i // 3, i % 3].axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    # Load the MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data/MNIST', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Read the network and run it on the test set
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()

    # # Test the network on new inputs
    images, labels = next(iter(test_loader))
    outputs = model(images)

    # Process and display results
    for i in range(10):
        output_values = outputs[i].detach().numpy()
        max_index = output_values.argmax()
        correct_label = labels[i].item()
        
        print(f"Example {i + 1}:")
        print(f"Network output values: {[round(val, 2) for val in output_values]}")
        print(f"Index of max output value: {max_index}")
        print(f"Correct label: {correct_label}")
        print()

    # Plot the first 9 digits with predictions
    visualize_3x3_grid(images, outputs)

    evaluate_handwriting(model)
