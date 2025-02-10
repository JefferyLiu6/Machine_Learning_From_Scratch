import numpy as np
from ml_util import normalize_features

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000, normalize=False):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.normalize = normalize
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Train the Support Vector Machine using gradient descent.
        """
        if self.normalize:
            X = normalize_features(X)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y = np.where(y <= 0, -1, 1)  # Ensure labels are -1 or 1
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.learning_rate * y[idx]
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        if self.normalize:
            X = normalize_features(X)
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
    
if __name__ == '__main__':
    # Sample dataset for SVM classification
    X_train = np.array([[2, 3], [1, 1], [4, 5], [6, 7], [1, 4], [7, 8]])
    y_train = np.array([1, -1, 1, 1, -1, 1])
    
    svm = SVM(learning_rate=0.01, lambda_param=0.1, epochs=1000, normalize=True)
    svm.fit(X_train, y_train)
    
    predictions = svm.predict(X_train)
    print("SVM Predictions:", predictions)
