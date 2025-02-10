import numpy as np
from ml_tools import mse
from ml_util import add_bias, normalize_features

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, normalize=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.normalize = normalize
        self.weights = None
        self.mean = None
        self.std = None
    
    def fit(self, X, y):
        """
        Train the Linear Regression model using Gradient Descent.
        """
        if self.normalize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = normalize_features(X)
        
        X = add_bias(X)  # Add bias term
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        
        for _ in range(self.epochs):
            y_pred = X @ self.weights
            gradient = -(2 / n_samples) * X.T @ (y - y_pred)
            self.weights -= self.learning_rate * gradient
    
    def predict(self, X):
        """
        Predict the target variable for given input features.
        """
        if self.normalize:
            X = normalize_features(X)
        X = add_bias(X)
        return X @ self.weights
    
    def evaluate(self, X, y):
        """
        Evaluates the model using MSE.
        """
        y_pred = self.predict(X)
        return mse(y, y_pred)

if __name__ == '__main__':
    # Sample dataset
    X = np.array([[1], [2], [3], [4], [5]])  # Single feature
    y = np.array([2, 4, 6, 8, 10])  # Linear relation y = 2x
    
    model = LinearRegression(learning_rate=0.1, epochs=1000, normalize=True)
    model.fit(X, y)
    
    predictions = model.predict(X)
    print("Predictions:", predictions)
    print("MSE:", model.evaluate(X, y))
