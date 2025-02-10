import numpy as np
from ml_util import add_bias, normalize_features

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, normalize=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.normalize = normalize
        self.weights = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train the Logistic Regression model using Gradient Descent.
        """
        if self.normalize:
            X = normalize_features(X)
        
        X = add_bias(X)  # Add bias term
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        
        for _ in range(self.epochs):
            y_pred = self.sigmoid(X @ self.weights)
            gradient = (1 / n_samples) * X.T @ (y_pred - y)
            self.weights -= self.learning_rate * gradient
    
    def predict_proba(self, X):
        """
        Predict probability estimates.
        """
        if self.normalize:
            X = normalize_features(X)
        X = add_bias(X)
        return self.sigmoid(X @ self.weights)
    
    def predict(self, X):
        """
        Predict class labels (0 or 1).
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
    
if __name__ == '__main__':
    # Sample dataset for Logistic Regression
    X_cls = np.array([[1], [2], [3], [4], [5]])  # Single feature
    y_cls = np.array([0, 0, 1, 1, 1])  # Binary labels
    
    log_model = LogisticRegression(learning_rate=0.1, epochs=1000, normalize=True)
    log_model.fit(X_cls, y_cls)
    
    log_predictions = log_model.predict(X_cls)
    print("Logistic Regression Predictions:", log_predictions)
