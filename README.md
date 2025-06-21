# Deep_Neural_Network_Based_Image_Classification
# 🧠 Deep Neural Network Image Classifier

This repository contains a complete pipeline for building, training, and evaluating deep neural networks from scratch using NumPy. The goal is to classify images—such as distinguishing cats from non-cats—without relying on high-level machine learning frameworks. This is an educational project to deepen understanding of how deep learning models operate internally.

---

## 📌 Features

- End-to-end training of:
  - A **two-layer neural network**
  - An **L-layer deep neural network**
- Manual implementation of forward and backward propagation
- Visualization of training performance through cost plots
- Evaluation on real image data
- Support for user-supplied image predictions
- Dataset saving and loading using `pickle` for quick reuse

---

## 📁 Dataset

The dataset consists of labeled RGB images used for binary classification (e.g., "cat" vs. "non-cat"). The data is loaded from `.h5` files using helper utilities and consists of:

- Training images and labels
- Test images and labels
- Class labels

Each image is resized and flattened into a vector and normalized before being passed into the neural network.

---

## ⚙️ Architecture

### Two-Layer Model
- Input layer of size 12288 (for 64x64x3 images)
- One hidden layer with ReLU activation
- Output layer with sigmoid activation

### L-Layer Model
- Deep architecture with multiple hidden layers
- Fully connected layers using ReLU and sigmoid
- Layer sizes configurable via `layers_dims` list

---

## 🧠 Learning Process

Both models are trained using:
- Forward propagation
- Cost computation (binary cross-entropy)
- Backward propagation (gradient computation)
- Gradient descent for parameter updates

The training loop runs for a specified number of iterations and logs cost periodically. Cost is plotted to visualize convergence.

---

## 📊 Evaluation

After training, both models are evaluated using:
- Accuracy on training and test sets
- Visual inspection of mislabeled test images

---

## 🖼️ Predict on Your Own Image

The repository includes functionality to:
- Load an external image
- Preprocess it to match training data format
- Use the trained model to classify it
- Display the prediction and image

This allows you to test the model's performance on completely new data.

---

## 💾 Data Persistence with Pickle

To avoid reloading and preprocessing data every time, the project supports saving and loading datasets using Python’s `pickle` module:
- Save training and test datasets after preprocessing
- Load them directly for repeated experimentation

---

## 🛠️ Requirements

The project requires the following Python libraries:
- `numpy`
- `matplotlib`
- `scipy`
- `Pillow` (PIL)
- `h5py`

These can be installed using the included `requirements.txt` file.

---

## 📚 Educational Focus

This project is ideal for:
- Learners who want to understand how deep learning works under the hood
- Those looking to implement neural networks from scratch without using libraries like TensorFlow or PyTorch
- Practicing hands-on with data processing, forward/backward passes, and prediction logic

---

## 📦 Repository Contents

- **Model scripts**: Implementations of two-layer and deep neural networks
- **Utility functions**: For parameter initialization, activations, forward/backward propagation, cost computation
- **Data handling**: Dataset loading, reshaping, normalization
- **Prediction tools**: Visual feedback on model outputs
- **Pickle scripts**: Save and load datasets for quick reuse

---
