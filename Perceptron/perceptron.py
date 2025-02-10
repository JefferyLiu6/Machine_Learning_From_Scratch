import numpy as np
from ml_util import normalize_features

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000, normalize=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.normalize = normalize
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Train the Perceptron model.
        """
        if self.normalize:
            X = normalize_features(X)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y = np.where(y <= 0, -1, 1)  # Ensure labels are -1 or 1
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                update = self.learning_rate * y[idx]
                if y[idx] * (np.dot(x_i, self.weights) + self.bias) <= 0:
                    self.weights += update * x_i
                    self.bias += update
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        if self.normalize:
            X = normalize_features(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)
    
if __name__ == '__main__':
    # Sample dataset for Perceptron classification
    X_train = np.array([[2, 3], [1, 1], [4, 5], [6, 7], [1, 4], [7, 8]])
    y_train = np.array([1, -1, 1, 1, -1, 1])
    
    perceptron = Perceptron(learning_rate=0.01, epochs=1000, normalize=True)
    perceptron.fit(X_train, y_train)
    
    predictions = perceptron.predict(X_train)
    print("Perceptron Predictions:", predictions)
