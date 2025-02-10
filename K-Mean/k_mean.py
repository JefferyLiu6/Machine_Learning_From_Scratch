import numpy as np
from ml_util import normalize_features
from ml_tools import euclidean_distance

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4, normalize=False):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.normalize = normalize
        self.centroids = None
    
    def fit(self, X):
        """
        Train the K-Means clustering model.
        """
        if self.normalize:
            X = normalize_features(X)
        
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        for _ in range(self.max_iters):
            clusters = self._assign_clusters(X)
            new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])
            
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            
            self.centroids = new_centroids
    
    def _assign_clusters(self, X):
        """
        Assign each sample to the nearest centroid using Euclidean distance.
        """
        distances = np.array([[euclidean_distance(x, centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample.
        """
        if self.normalize:
            X = normalize_features(X)
        return self._assign_clusters(X)
    
if __name__ == '__main__':
    # Sample dataset for K-Means clustering
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    kmeans = KMeans(k=2, max_iters=100, normalize=True)
    kmeans.fit(X)
    
    predictions = kmeans.predict(X)
    print("Cluster assignments:", predictions)
    print("Centroids:", kmeans.centroids)
