import numpy as np
import os
from torchvision.datasets import EMNIST

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

print("Downloading and preparing EMNIST Letters dataset...")
# Downloads EMNIST dataset via torchvision into ./data
emnist_train = EMNIST(root='./data', split='letters', download=True, train=True)
emnist_test = EMNIST(root='./data', split='letters', download=True, train=False)

# EMNIST images are transposed by default; transpose to standard (28, 28) orientation
X_train = emnist_train.data.numpy().transpose(0, 2, 1).reshape(-1, 784).astype(np.float32) / 255.0
# EMNIST letters targets are 1-indexed (1 to 26), map to 0-25
y_train = emnist_train.targets.numpy() - 1

X_test = emnist_test.data.numpy().transpose(0, 2, 1).reshape(-1, 784).astype(np.float32) / 255.0
y_test = emnist_test.targets.numpy() - 1

# Hyperparameters
input_size = 784
hidden_size = 128
output_size = 26
learning_rate = 0.1
epochs = 15
batch_size = 64

# Initialize weights (He initialization)
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))

num_samples = X_train.shape[0]

print("Training EMNIST Framework-Free MLP Network...")
for epoch in range(epochs):
    permutation = np.random.permutation(num_samples)
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    
    for i in range(0, num_samples, batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        
        # Forward pass
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)
        
        # One-hot encoding
        one_hot_y = np.zeros((len(y_batch), output_size))
        one_hot_y[np.arange(len(y_batch)), y_batch] = 1.0
        
        # Backpropagation
        dZ2 = (A2 - one_hot_y) / len(y_batch)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(X_batch.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # SGD Update
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
    # Evaluate epoch test accuracy
    Z1_test = np.dot(X_test, W1) + b1
    A1_test = relu(Z1_test)
    Z2_test = np.dot(A1_test, W2) + b2
    preds = np.argmax(softmax(Z2_test), axis=1)
    acc = np.mean(preds == y_test) * 100
    print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {acc:.2f}%")

# Save parameters and lightweight sample set for exploration
np.savez("emnist_model_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Saved 'emnist_model_weights.npz' successfully.")

# Save 1000 sample test images for the interactive Streamlit explorer
np.savez("emnist_samples.npz", X_test=X_test[:1000], y_test=y_test[:1000])
print("Saved 'emnist_samples.npz' successfully.")