"""
Samara Holmes
Spring 2025

Transfer Learning on Greek Letters
"""
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from build_network import MyNetwork
import os
from PIL import Image, ImageOps
import torch.nn as nn
from build_network import *
from read_and_test_network import visualize_3x3_grid

class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        """
        Code that will transform the RGB images to grayscale, scale and crop them to the correct size, and invert the intensities to match the MNIST digits.
        """
        x = transforms.functional.rgb_to_grayscale( x )
        x = transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = transforms.functional.center_crop( x, (28, 28) )
        return transforms.functional.invert( x )
    
    
if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    # Read an existing model from a file and load the pre-trained weights
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    model.eval()

    # Freeze the network weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer with a new Linear layer with three nodes
    print("Model structure before:")
    print(model)

    if hasattr(model, 'fc1'):
        setattr(model, 'fc1', nn.Linear(model.fc1.in_features, 3))
        model.fc2 = nn.Linear(3, 5)
    else:
        raise AttributeError("Model does not have an fc1 layer")

    print("Model structure after:")
    print(model)

    # DataLoader for the Greek data set
    image_folder = './data/greek_train'
    dataset = datasets.ImageFolder(
        root=image_folder,
        transform=transforms.Compose([
            GreekTransform(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    # Create the DataLoader for the dataset
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=5,  # Adjust batch size as needed
        shuffle=True
    )

    # Print the class mappings
    print(f"Class mapping: {dataset.class_to_idx}")
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.train()

    # Init params
    n_epochs = 100
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01

    # Train
    train_accuracies = []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_acc = train_network(data_loader, model, loss_fn, optimizer)
        train_accuracies.append(train_acc)

    print("Done training!")
    
    # Load and process all images
    predictions = []
    for class_idx, class_name in idx_to_class.items():
        class_folder = os.path.join(image_folder, class_name)
        
        # Get one image file from the class folder
        for file_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, file_name)
            if os.path.isfile(image_path):
                # Process the image
                image = Image.open(image_path).convert('RGB')  # Convert to RGB
                transform = transforms.Compose([
                    GreekTransform(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
                
                # Get the prediction
                model.eval()
                with torch.no_grad():
                    output = model(image_tensor)
                    predicted_label_idx = torch.argmax(output, dim=1).item()
                    predicted_class = idx_to_class[predicted_label_idx]
                    predictions.append((class_name, predicted_class, file_name))

                break  # Process only one image per class

    for actual_class, predicted_class, file_name in predictions:
        print(f"Image '{file_name}' from Class '{actual_class}': Predicted - {predicted_class}")

    # Plot the accuracies
    plot_accuracies(train_accuracies, train_accuracies)
