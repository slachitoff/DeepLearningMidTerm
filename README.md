# CIFAR-10 Image Classification with ResNet

This repository contains the resources and links for a deep learning project focused on classifying images from the CIFAR-10 dataset using a modified Residual Network (ResNet). The primary goal was to fine-tune a ResNet model to achieve the highest possible test accuracy while keeping the number of parameters below 5 million.

## Repository Contents

- A direct link to the Google Colab notebook used for the project.
- The `model.pt` file containing the saved parameters from our most successful training run.

## Model Overview

The `model.pt` file is a PyTorch checkpoint containing the weights of a fine-tuned ResNet model that achieved a test accuracy of 95.76% on the CIFAR-10 dataset.

## Getting Started

### Google Colab Notebook

To access and interact with the project notebook:
1. Follow the provided link to the Google Colab environment.
2. The notebook can be run in your browser, allowing you to replicate our results or further experiment with the model.

## Using the Pre-Trained Model

To utilize the pre-trained model (`model.pt`), ensure you have PyTorch installed and then follow these steps:

1. Download the `model.pt` file from this repository.
2. Use the following code snippet in your PyTorch environment to load the model:

```python
import torch
from model_architecture import PreActResNet, PreActBlock  # Ensure you have the model architecture defined or imported

# Set the device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize an instance of the PreActResNet model
# The architecture is defined to match the saved checkpoint with <5 million parameters
model = PreActResNet(PreActBlock, [2, 1, 1, 1], num_classes=10)

# Load the pretrained model from a file
state_dict = torch.load('path/to/model.pt', map_location=device)
model.load_state_dict(state_dict)

# Move the model to the designated computing device
model.to(device)
model.eval()  # Switch the model to evaluation mode
