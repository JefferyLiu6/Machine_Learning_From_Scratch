# Machine Learning From Scratch


## 📌 Introduction
This repository contains implementations of fundamental machine learning algorithms **from scratch** using Python and NumPy. Each algorithm is organized into separate modules, making it easy to understand and modify.

### 🔥 Why this repo?
- No external ML libraries (like Scikit-Learn, TensorFlow, PyTorch)
- Pure Python + NumPy implementation
- Educational and beginner-friendly
- Well-structured code with clear documentation

## 🚀 Implemented Algorithms
This repository includes the following machine learning algorithms:

### 🔷 Supervised Learning
- **Linear Regression** (`linear_regression/LR.py`)
- **Logistic Regression** (`logistic_regression/LR.py`)
- **Support Vector Machine (SVM)** (`SVM/svm.py`)
- **Decision Tree** (`Decision_Tree/DT.py`)
- **Naive Bayes** (`NaiveBayes/nb.py`)
- **K-Nearest Neighbors (KNN)** (`kNN/knn.py`)
- **Perceptron** (`Perceptron/perceptron.py`)

### 🔷 Unsupervised Learning
- **K-Means Clustering** (`K-Mean/k_mean.py`)
- **Principal Component Analysis (PCA)** (`PCA/pca.py`)

## 📂 Project Structure
```
Machine_Learning_From_Scratch/
│── Decision_Tree/
│── K-Mean/
│── NaiveBayes/
│── PCA/
│── Perceptron/
│── SVM/
│── kNN/
│── linear_regression/
│── logistic_regression/
│── tools/
│── README.md
```

## 🛠 Installation
Clone this repository to get started:
```bash
git clone https://github.com/YourUsername/Machine_Learning_From_Scratch.git
cd Machine_Learning_From_Scratch
```
Ensure you have **Python 3.8+** and install dependencies:
```bash
pip install numpy
```

## 📌 Usage
Each algorithm can be tested individually. Example usage:

```python
from linear_regression.LR import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression(learning_rate=0.01, epochs=1000, normalize=True)
model.fit(X, y)
predictions = model.predict(X)
print("Predictions:", predictions)
```

## 📝 Contributing
Feel free to open issues, submit PRs, or suggest improvements. Contributions are always welcome!

## 📜 License
This project is **MIT licensed**. Feel free to use and modify it!

## 🙌 Acknowledgments
This project was inspired by the fundamental machine learning concepts and aims to provide a deeper understanding of how ML algorithms work internally.

🚀 Happy Coding!
