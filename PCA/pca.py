import numpy as np
from ml_util import normalize_features

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        """
        Compute the principal components using SVD.
        """
        # Normalize features
        X = normalize_features(X)
        
        # Compute covariance matrix
        covariance_matrix = np.cov(X, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]
    
    def transform(self, X):
        """
        Project data onto the principal components.
        """
        X = normalize_features(X)
        return np.dot(X, self.components)
    
if __name__ == '__main__':
    # Sample dataset for PCA
    X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])
    
    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca = pca.transform(X)
    print("Transformed Data:", X_pca)
