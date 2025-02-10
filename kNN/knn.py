import numpy as np
from ml_tools import euclidean_distance

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store the training data.
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """
        Predict the class labels for the given input samples.
        """
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x):
        """
        Predict a single instance using K-Nearest Neighbors.
        """
        distances = np.array([euclidean_distance(x, x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Majority vote
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique[np.argmax(counts)]
    
if __name__ == '__main__':
    # Sample dataset for KNN classification
    X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    X_test = np.array([[3, 4], [5, 6]])
    predictions = knn.predict(X_test)
    print("KNN Predictions:", predictions)
