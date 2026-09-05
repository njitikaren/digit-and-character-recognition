import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ==============================================================================
# 1. PAGE CONFIGURATION & HEADER BANNER
# ==============================================================================
st.set_page_config(
    page_title="Framework-Free Deep Learning Suite",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title(" Framework-Free MLP Neural Network Suite")
st.header(
    "Custom Deep Learning Engine built purely with Python & NumPy (Linear"
    " Algebra & Calculus) without TensorFlow, PyTorch, or Keras." 
    
)

# ==============================================================================
# 2. MODEL WEIGHTS & DATA INITIALIZATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.npz")
MNIST_SAMPLES_PATH = os.path.join(BASE_DIR, "mnist_samples.npz")
EMNIST_WEIGHTS_PATH = os.path.join(BASE_DIR, "emnist_weights.npz")
EMNIST_SAMPLES_PATH = os.path.join(BASE_DIR, "emnist_samples.npz")


@st.cache_resource
def load_trained_digit_parameters():
    """Load pre-trained weight and bias matrices for digits (0-9)."""
    if not os.path.exists(MNIST_WEIGHTS_PATH):
        st.error(
            f"Digit weights file not found at '{MNIST_WEIGHTS_PATH}'. Please ensure"
            " 'model_weights.npz' is committed."
        )
        st.stop()
    data = np.load(MNIST_WEIGHTS_PATH)
    return data["W1"], data["b1"], data["W2"], data["b2"]


@st.cache_resource
def load_trained_character_parameters():
    """Load pre-trained weight and bias matrices for EMNIST characters (A-Z)."""
    if not os.path.exists(EMNIST_WEIGHTS_PATH):
        st.warning(
            f"Character weights file not found at '{EMNIST_WEIGHTS_PATH}'. Upload"
            " 'emnist_weights.npz' to enable character recognition."
        )
        return None, None, None, None
    data = np.load(EMNIST_WEIGHTS_PATH)
    return data["W1"], data["b1"], data["W2"], data["b2"]


# Load neural network parameters into memory
W1_dig, b1_dig, W2_dig, b2_dig = load_trained_digit_parameters()
W1_char, b1_char, W2_char, b2_char = load_trained_character_parameters()

# Alphabet Map for EMNIST Characters (0 = A, 25 = Z)
ALPHABET = [chr(i) for i in range(65, 91)]


def generate_synthetic_mnist_samples(samples_per_class=16):
    """Generates synthetic 28x28 grayscale image samples (1, 784) for digits 0-9."""
    X_synth = []
    y_synth = []

    for digit in range(10):
        for _ in range(samples_per_class):
            img = Image.new("L", (28, 28), color=0)
            draw = ImageDraw.Draw(img)

            font = ImageFont.load_default()
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-2, 3)
            draw.text((10 + dx, 4 + dy), str(digit), fill=255, font=font)

            img_arr = np.array(img, dtype=np.float32) / 255.0
            X_synth.append(img_arr.flatten())
            y_synth.append(digit)

    return np.array(X_synth, dtype=np.float32), np.array(y_synth, dtype=np.int64)


@st.cache_data
def load_or_augment_mnist_samples():
    """Loads MNIST sample dataset from disk or generates synthetic fallbacks."""
    if os.path.exists(MNIST_SAMPLES_PATH):
        try:
            mnist_data = np.load(MNIST_SAMPLES_PATH)
            return mnist_data["X_test"], mnist_data["y_test"]
        except Exception:
            pass

    return generate_synthetic_mnist_samples(samples_per_class=16)


def generate_synthetic_emnist_samples(samples_per_class=16):
    """Generates synthetic 28x28 grayscale image samples (1, 784) for all characters A-Z."""
    X_synth = []
    y_synth = []

    for idx, char in enumerate(ALPHABET):
        for _ in range(samples_per_class):
            img = Image.new("L", (28, 28), color=0)
            draw = ImageDraw.Draw(img)

            font = ImageFont.load_default()
            dx = np.random.randint(-2, 3)
            dy = np.random.randint(-2, 3)
            draw.text((8 + dx, 4 + dy), char, fill=255, font=font)

            img_arr = np.array(img, dtype=np.float32) / 255.0
            X_synth.append(img_arr.flatten())
            y_synth.append(idx)

    return np.array(X_synth, dtype=np.float32), np.array(y_synth, dtype=np.int64)


@st.cache_data
def load_or_augment_emnist_samples():
    """Loads EMNIST sample dataset from disk and guarantees all classes (A-Z) exist."""
    if os.path.exists(EMNIST_SAMPLES_PATH):
        try:
            emnist_data = np.load(EMNIST_SAMPLES_PATH)
            X_test, y_test = emnist_data["X_test"], emnist_data["y_test"]

            if np.max(y_test) == 26 and np.min(y_test) >= 1:
                y_test = y_test - 1

            existing_classes = set(np.unique(y_test))
            missing_classes = set(range(26)) - existing_classes

            if len(missing_classes) > 0:
                X_synth, y_synth = generate_synthetic_emnist_samples(
                    samples_per_class=16
                )
                mask = np.isin(y_synth, list(missing_classes))
                X_test = np.vstack([X_test, X_synth[mask]])
                y_test = np.concatenate([y_test, y_synth[mask]])

            return X_test, y_test
        except Exception:
            pass

    return generate_synthetic_emnist_samples(samples_per_class=16)


# ==============================================================================
# 3. FRAMEWORK-FREE INFERENCE ENGINE & PREPROCESSING PIPELINE
# ==============================================================================
def relu(Z):
    """ReLU Activation Function: max(0, Z)"""
    return np.maximum(0, Z)


def softmax(Z):
    """Numerically stable Softmax Activation Function"""
    exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)


def forward_pass(X, W1, b1, W2, b2):
    """Pure Linear Algebra Forward Propagation:
    Hidden Layer: Z1 = X * W1 + b1, A1 = ReLU(Z1)
    Output Layer: Z2 = A1 * W2 + b2, A2 = Softmax(Z2)
    """
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return A2


def preprocess_canvas_image(canvas_array):
    """Preprocessing Pipeline: Converts RGBA canvas array into a normalized
    (1, 784) grayscale vector centered by bounding box.
    """
    img = Image.fromarray(canvas_array.astype("uint8")).convert("L")
    bbox = img.getbbox()
    if bbox is None:
        return None  # Canvas is blank

    img_cropped = img.crop(bbox)
    img_cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

    new_img = Image.new("L", (28, 28), 0)
    upper_left = (
        (28 - img_cropped.width) // 2,
        (28 - img_cropped.height) // 2,
    )
    new_img.paste(img_cropped, upper_left)

    img_matrix = np.array(new_img, dtype=np.float32) / 255.0
    return img_matrix.reshape(1, 784)


# ==============================================================================
# 4. DASHBOARD TAB NAVIGATION
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    " Interactive Unified Canvas",
    " MNIST Explorer",
    " EMNIST Dataset Explorer",
])

# --- TAB 1: UNIFIED DYNAMIC DRAWING CANVAS ---
with tab1:
    mode = st.radio(
        "Select Canvas Mode", ["Digit (0-9)", "Character (A-Z)"], horizontal=True
    )
    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader(" Freehand Canvas Input")
        if mode == "Digit (0-9)":
            st.write("Draw any digit (**0 to 9**) inside the canvas below:")
        else:
            st.write("Draw any capital letter (**A to Z**) inside the canvas below:")

        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=16,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=200,
            width=200,
            drawing_mode="freedraw",
            key=f"drawable_canvas_{mode}",
        )

    with col_right:
        st.subheader("2. Real-Time Neural Analytics")

       if (
    canvas_result is not None
    and getattr(canvas_result, "image_data", None) is not None
    and canvas_result.image_data.size > 0
):
    input_vector = preprocess_canvas_image(canvas_result.image_data)

    if input_vector is None:
        st.info("Draw on the canvas to trigger real-time neural network inference.")
    else:
        if mode == "Digit (0-9)":
            probabilities = forward_pass(
                input_vector, W1_dig, b1_dig, W2_dig, b2_dig
            )[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class]) * 100

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicted Digit", f"{predicted_class}")
            with m2:
                st.metric("Softmax Confidence", f"{confidence:.2f}%")

            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.bar(range(10), probabilities, color="#4CAF50")
            bars[predicted_class].set_color("#FF5722")

            ax.set_xticks(range(10))
            ax.set_xlabel("Digit Class (0-9)", fontsize=15, fontweight="bold")
            ax.set_ylabel("Probability", fontsize=20, fontweight="bold")
            ax.set_ylim([0, 1.0])
            ax.set_title(
                "Digit Softmax Output Distribution",
                fontsize=12,
                fontweight="bold",
            )
            st.pyplot(fig)

        else:  # Character (A-Z) Mode
            if W1_char is None:
                st.warning(
                    "Please upload 'emnist_weights.npz' to enable character predictions."
                )
            else:
                probabilities = forward_pass(
                    input_vector, W1_char, b1_char, W2_char, b2_char
                )[0]
                predicted_idx = int(np.argmax(probabilities))
                predicted_char = ALPHABET[predicted_idx]
                confidence = float(probabilities[predicted_idx]) * 100

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Predicted Character", f"{predicted_char}")
                with m2:
                    st.metric("Softmax Confidence", f"{confidence:.2f}%")

                # --- FULL A-Z BAR CHART ---
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(ALPHABET, probabilities, color="#2196F3")
                bars[predicted_idx].set_color("#E91E63")

                ax.set_xticks(range(26))
                ax.set_xticklabels(ALPHABET, fontsize=25)
                ax.set_xlabel(
                    "Character Class (A-Z)", fontsize=15, fontweight="bold"
                )
                ax.set_ylabel("Softmax Probability", fontsize=20, fontweight="bold")
                ax.set_ylim([0, 1.0])
                ax.set_title(
                    "EMNIST Full Character Softmax Output Distribution (A-Z)",
                    fontsize=18,
                    fontweight="bold",
                )
                ax.grid(axis="y", linestyle="--", alpha=0.5)

                st.pyplot(fig)

# --- TAB 2: MNIST DATASET EXPLORER ---
with tab2:
    st.header(" MNIST Dataset Explorer")
    st.header("Visualizing sample 28x28 handwritten digit images (0 to 9) from the dataset.")

    X_mnist, y_mnist = load_or_augment_mnist_samples()

    fig_grid, axes = plt.subplots(2, 5, figsize=(14, 5))
    axes = axes.flatten()

    for digit in range(10):
        matching_indices = np.where(y_mnist == digit)[0]
        ax = axes[digit]
        if len(matching_indices) > 0:
            sample_img = X_mnist[matching_indices[0]].reshape(28, 28)
            ax.imshow(sample_img, cmap="gray")
        ax.set_title(f"Digit: {digit}", fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig_grid)

# --- TAB 3: EMNIST DATASET EXPLORER ---
with tab3:
    st.header(" EMNIST Dataset Explorer")
    st.header("Visualizing sample 28x28 handwritten character images (A to Z) from the dataset.")

    X_emnist, y_emnist = load_or_augment_emnist_samples()

    fig_grid, axes = plt.subplots(4, 7, figsize=(14, 8))
    axes = axes.flatten()

    for idx, char in enumerate(ALPHABET):
        matching_indices = np.where(y_emnist == idx)[0]
        ax = axes[idx]
        if len(matching_indices) > 0:
            sample_img = X_emnist[matching_indices[0]].reshape(28, 28)
            ax.imshow(sample_img, cmap="gray")
        ax.set_title(f"'{char}'", fontsize=20, fontweight="bold")
        ax.axis("off")

    # Turn off leftover empty subplots in the 4x7 grid
    for idx in range(26, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    st.pyplot(fig_grid) 