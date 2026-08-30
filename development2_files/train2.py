import os
import numpy as np

# Download and load datasets using torchvision (or replace with direct MNIST/EMNIST loading)
from torchvision.datasets import EMNIST, MNIST

def get_data():
    print("Downloading and processing MNIST...")
    mnist_train = MNIST(root="./data", train=True, download=True)
    mnist_test = MNIST(root="./data", train=False, download=True)

    print("Downloading and processing EMNIST (Letters)...")
    emnist_test = EMNIST(root="./data", split="letters", train=False, download=True)

    # 1. Process MNIST (Digits 0-9)
    X_mnist_test = mnist_test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_mnist_test = mnist_test.targets.numpy()

    # 2. Process EMNIST (Letters 1-26 mapped to A-Z)
    # EMNIST images are transposed by default; rotate 90 deg counter-clockwise and flip
    X_emnist_raw = emnist_test.data.numpy()
    X_emnist_corrected = np.array([np.rot90(np.flipud(img), -1) for img in X_emnist_raw])
    X_emnist_test = X_emnist_corrected.reshape(-1, 784).astype(np.float32) / 255.0
    y_emnist_test = emnist_test.targets.numpy() - 1  # 0-indexed: 0 -> A, 25 -> Z

    return X_mnist_test, y_mnist_test, X_emnist_test, y_emnist_test

def train_framework_free_mlp(X_train, y_train, input_dim, num_classes, hidden_dim=128, epochs=10, lr=0.1):
    """Pure NumPy MLP Training using manual Backpropagation."""
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros((1, num_classes))

    # One-hot encode targets
    num_samples = X_train.shape[0]
    Y_onehot = np.zeros((num_samples, num_classes))
    Y_onehot[np.arange(num_samples), y_train] = 1.0

    batch_size = 64
    for epoch in range(epochs):
        permutation = np.random.permutation(num_samples)
        X_shuffled = X_train[permutation]
        Y_shuffled = Y_onehot[permutation]

        for i in range(0, num_samples, batch_size):
            X_b = X_shuffled[i:i+batch_size]
            Y_b = Y_shuffled[i:i+batch_size]
            b_size = X_b.shape[0]

            # Forward pass
            Z1 = np.dot(X_b, W1) + b1
            A1 = np.maximum(0, Z1)  # ReLU
            Z2 = np.dot(A1, W2) + b2
            exp_Z2 = np.exp(Z2 - np.max(Z2, axis=-1, keepdims=True))
            A2 = exp_Z2 / np.sum(exp_Z2, axis=-1, keepdims=True)  # Softmax

            # Backpropagation
            dZ2 = (A2 - Y_b) / b_size
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, W2.T)
            dZ1 = dA1 * (Z1 > 0)
            dW1 = np.dot(X_b.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # Gradient Descent Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

    return W1, b1, W2, b2

if __name__ == "__main__":
    X_m_test, y_m_test, X_e_test, y_e_test = get_data()

    # Save benchmark sample subsets for visualization
    np.savez_compressed("mnist_samples.npz", X_test=X_m_test[:1000], y_test=y_m_test[:1000])
    np.savez_compressed("emnist_samples.npz", X_test=X_e_test[:1000], y_test=y_e_test[:1000])
    print("Saved 'mnist_samples.npz' and 'emnist_samples.npz'")

    # Train Digit Weights (10 output classes)
    print("Training Digit Model...")
    W1_d, b1_d, W2_d, b2_d = train_framework_free_mlp(X_m_test, y_m_test, 784, 10)

    # Train Character Model (26 output classes: A-Z)
    print("Training Character Model...")
    W1_c, b1_c, W2_c, b2_c = train_framework_free_mlp(X_e_test, y_e_test, 784, 26)

    # Export all trained parameter matrices to npz container
    np.savez_compressed(
        "model_weights.npz",
        W1=W1_d, b1=b1_d, W2=W2_d, b2=b2_d,
        W1_char=W1_c, b1_char=b1_c, W2_char=W2_c, b2_char=b2_c
    )
    print("Successfully exported parameters to 'model_weights.npz'")