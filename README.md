# Framework-Free Multi-Layer Perceptron Machine Learning Processing (MLP) Digit Classifier

 A deep learning engine classification web application built from scratch using  pure Python , NumPy (Linear Algebra and Calculus), complete with forward propagation, backpropagation, and custom activation functions without high-level Machine Learning frameworks, deployed as an interactive Streamlit web application to recognise and predicts digits from (0 to 9) using the MNIST datasets and character from (A to Z) using the EMNIST datasets.

## AUTHOR AND PROJECT INFORMATION

* **DEVELOPER**: NJITI NGAFOR KAREN
* **EMAIL:** njitikaren@gmail.com
* **GitHub Profile:** [@NJITI KAREN](https://github.com/NJITI KAREN)
* **Institution:** Higher Institute of Management Studies (HIMS) Buea
* **Department:** Software Engineering
* **Live Web App:** [Launch Streamlit Application](https://digit-and-character-recognition.streamlit.app)

## Key Features
* **Framework-Free Engine:** Custom Multi-Layer Perceptron (MLP) with manual forward propagation, backpropagation, ReLU, and Softmax activations.
* **Interactive Canvas:** Draw freehand digits and character ($0$–$9$) directly on the screen for real-time inference.
* **Dataset Explorer:** Browse raw $28 \times 28$ grayscale benchmark samples from the MNIST dataset(use for digits recognition) and EMNIST dataset(use for character recognition).

## Setup & Local Run
```bash
# Clone repository
git clone [https://github.com/NJITI KAREN/digit-and-character-recognition.git](https://github.com/NJITI_KAREN/digit-and-character-recognintion.git)

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run app.py