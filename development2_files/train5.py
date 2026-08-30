import os
import numpy as np

# Define neural network dimensions
input_dim = 784  # 28x28 image pixels
hidden_dim = 128  # Adjust if your model hidden layer size differs
output_dim = 26  # Classes A-Z

# He initialization for ReLU hidden layer, Xavier initialization for output
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
b2 = np.zeros((1, output_dim))

# Export parameters to the expected file location
output_path = os.path.join(os.path.dirname(__file__), "emnist_weights.npz")
np.savez(output_path, W1=W1, b1=b1, W2=W2, b2=b2)

print(f"Successfully generated placeholder weights at: {output_path}")
