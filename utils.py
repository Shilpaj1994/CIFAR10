#!/usr/bin/env python3
"""
Utility Script containing functions to be used for training
"""
# Standard Library Imports
from typing import NoReturn

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary


def get_summary(model: 'object of model architecture', input_size: tuple) -> NoReturn:
    """
    Function to get the summary of the model architecture
    :param model: Object of model architecture class
    :param input_size: Input data shape (Channels, Height, Width)
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    summary(network, input_size=input_size)


def get_misclassified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()
    
    # List to store misclassified Images
    misclassified_data = []
    
    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:
            
            # Migrate the data to the device
            data, target = data.to(device), target.to(device)
            
            # Extract single image, label from the batch
            for image, label in zip(data, target):
                
                # Add batch dimension to the image
                image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(image)
                
                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)
                
                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data


# -------------------- DATA STATISTICS --------------------
def get_mnist_statistics(data_set, data_set_type='Train'):
    """
    Function to return the statistics of the training data
    :param data_set: Training dataset
    :param data_set_type: Type of dataset [Train/Test/Val]
    """
    # We'd need to convert it into Numpy! Remember above we have converted it into tensors already
    train_data = data_set.train_data
    train_data = data_set.transform(train_data.numpy())

    print(f'[{data_set_type}]')
    print(' - Numpy Shape:', data_set.train_data.cpu().numpy().shape)
    print(' - Tensor Shape:', data_set.train_data.size())
    print(' - min:', torch.min(train_data))
    print(' - max:', torch.max(train_data))
    print(' - mean:', torch.mean(train_data))
    print(' - std:', torch.std(train_data))
    print(' - var:', torch.var(train_data))

    dataiter = next(iter(data_set))
    images, labels = dataiter[0], dataiter[1]

    print(images.shape)
    print(labels)

    # Let's visualize some of the images
    plt.imshow(images[0].numpy().squeeze(), cmap='gray')


def get_cifar_property(images, operation):
    """
    Get the property on each channel of the CIFAR
    :param images: Get the property value on the images
    :param operation: Mean, std, Variance, etc
    """
    param_r = eval('images[:, 0, :, :].' + operation + '()')
    param_g = eval('images[:, 1, :, :].' + operation + '()')
    param_b = eval('images[:, 2, :, :].' + operation + '()')
    return param_r, param_g, param_b


def get_cifar_statistics(data_set, data_set_type='Train'):
    """
    Function to get the statistical information of the CIFAR dataset
    :param data_set: Training set of CIFAR
    :param data_set_type: Training or Test data
    """
    # Images in the dataset
    images = [item[0] for item in data_set]
    images = torch.stack(images, dim=0).numpy()

    # Calculate mean over each channel
    mean_r, mean_g, mean_b = get_cifar_property(images, 'mean')

    # Calculate Standard deviation over each channel
    std_r, std_g, std_b = get_cifar_property(images, 'std')

    # Calculate min value over each channel
    min_r, min_g, min_b = get_cifar_property(images, 'min')

    # Calculate max value over each channel
    max_r, max_g, max_b = get_cifar_property(images, 'max')

    # Calculate variance value over each channel
    var_r, var_g, var_b = get_cifar_property(images, 'var')

    print(f'[{data_set_type}]')
    print(f' - Total {data_set_type} Images: {len(data_set)}')
    print(f' - Tensor Shape: {images[0].shape}')
    print(f' - min: {min_r, min_g, min_b}')
    print(f' - max: {max_r, max_g, max_b}')
    print(f' - mean: {mean_r, mean_g, mean_b}')
    print(f' - std: {std_r, std_g, std_b}')
    print(f' - var: {var_r, var_g, var_b}')

    # Let's visualize some of the images
    plt.imshow(np.transpose(images[1].squeeze(), (1, 2, 0)))
