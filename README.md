# Indian Food Classification using CNN

This project implements a multi-class classification model to classify Indian food items using Convolutional Neural Networks (CNN). The model is built with **PyTorch** and trained on a dataset of Indian food images.

## Project Overview

The goal of this project is to build a model that can classify various types of Indian food. The classification is performed using a Convolutional Neural Network (CNN) architecture, which is trained to recognize and categorize images of Indian food items. 

## Key Features

- **Dataset**: A collection of Indian food images for multi-class classification.
- **Framework**: Built using **PyTorch**.
- **Data Augmentation**: Used **trivial augmentations** and **random horizontal transformations** to increase the robustness of the model.
- **Model Architecture**: A CNN model with approximately **1.3 million parameters**.
- **Optimizer**: **Stochastic Gradient Descent (SGD)** optimizer for model training.
- **Loss Function**: **CrossEntropyLoss** for multi-class classification.
- **Visualization**: Used **Matplotlib** to plot and visualize sample images during training.

## Requirements

- Python
- PyTorch
- Matplotlib
- NumPy
- PIL (Python Imaging Library)
