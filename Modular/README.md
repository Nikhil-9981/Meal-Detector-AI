**Modular Code for Meal-Detector-AI**
This repository contains code for training and evaluating a convolutional neural network (CNN) for image classification using the TinyVGG architecture. The project is structured to facilitate data loading, model building, and training processes.

Project Structure
data_setup.py: Contains functions for creating PyTorch DataLoaders for image classification data.

create_dataloaders: A function that takes in paths to training and testing directories, applies transformations, and returns DataLoaders for both sets along with class names.
engine.py: Provides utilities for training and evaluating models.

train_step: Performs a single training step.
test_step: Evaluates the model on a test dataset.
train: Orchestrates the training process over multiple epochs, including logging of metrics.
model_builder.py: Defines the TinyVGG model architecture.

TinyVGG: A class that builds a CNN with two convolutional blocks followed by a fully connected layer, designed for image classification tasks.
Getting Started
Prerequisites
Python 3.x
PyTorch
torchvision
Other libraries as needed (e.g., NumPy, Matplotlib)
Usage
Data Preparation: Organize your dataset into training and testing directories. Each class should have its own subdirectory.

Training:

Use create_dataloaders from data_setup.py to load your data.
Initialize the TinyVGG model from model_builder.py.
Use the train function from engine.py to train the model.
Evaluation:

Evaluate the trained model using the test_step function in engine.py.
