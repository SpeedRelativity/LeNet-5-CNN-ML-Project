# LeNet-5 Image Classification - MNIST Dataset

This project implements a **LeNet-5 Convolutional Neural Network (CNN)** to classify images from the **MNIST dataset**. The model achieves over **98% accuracy** in classifying handwritten digits (0-9). The project demonstrates essential deep learning concepts, such as convolutions, pooling, fully connected layers, and the use of PyTorch for training and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)

## Project Overview

This project focuses on building a convolutional neural network (CNN) based on the **LeNet-5** architecture for image classification. The model is trained and evaluated on the **MNIST dataset**, which contains 70,000 images of handwritten digits (60,000 for training and 10,000 for testing). 

The following steps are involved in the project:

1. **Data Preprocessing**: The MNIST dataset is loaded and transformed into tensors suitable for training. Normalization is applied to the data for better model convergence.
2. **Model Architecture**: The model consists of:
   - **Convolutional layers** to extract features from images.
   - **Average pooling layers** to downsample feature maps.
   - **Fully connected layers** for classification.
   - **Activation functions (Tanh)** to introduce non-linearity.
3. **Training**: The model is trained using the **Adam optimizer** and **Cross-Entropy Loss** for classification tasks.
4. **Evaluation**: The model's performance is evaluated on the test set, and predictions are visualized alongside true labels.

## Tech Stack

- **Python**: The primary programming language for the project.
- **PyTorch**: A deep learning framework used for building, training, and evaluating the neural network.
- **Matplotlib**: Used for visualizing predictions, true labels, and images.
- **MNIST Dataset**: A widely used dataset for training image classification models.

## Features

- Implements the **LeNet-5 CNN architecture** for handwritten digit classification.
- Achieves over **98% accuracy** on the MNIST test set.
- Visualizes **sample predictions** with images, predicted labels, and true labels.
- Shows the full training and validation pipeline in **PyTorch**, including model evaluation.



## Usage

1. Run the `main.py` script to train the model on the MNIST dataset:
    ```bash
    python main.py
    ```

2. The model will train for **2 epochs** and print the accuracy for each epoch.
   
3. After training, the script will display **sample predictions** with the images and their true labels using **Matplotlib**.

## Results

After training, you can expect the model to achieve **98%+ accuracy** on the MNIST test set. The visualizations will show sample images, predicted labels, and the corresponding true labels.
