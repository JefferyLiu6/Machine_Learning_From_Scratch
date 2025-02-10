import numpy as np

def add_bias(X):
    """
    Adds a column of ones to the feature matrix X to account for the bias term.
    
    Parameters:
        X (np.ndarray): Input features of shape (n_samples, n_features)
        
    Returns:
        np.ndarray: Augmented matrix with a bias column.
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])

def normalize_features(X):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    
    Parameters:
        X (np.ndarray): Input features of shape (n_samples, n_features)
        
    Returns:
        np.ndarray: Normalized features.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std_replaced = np.where(std == 0, 1, std)
    return (X - mean) / std_replaced

def min_max_scale(X):
    """
    Scales features to a [0, 1] range.
    
    Parameters:
        X (np.ndarray): Input features of shape (n_samples, n_features)
        
    Returns:
        np.ndarray: Scaled features.
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    diff = np.where(X_max - X_min == 0, 1, X_max - X_min)  # Avoid division by zero
    return (X - X_min) / diff

def euclidean_distance(a, b):
    """
    Computes the Euclidean distance between two vectors.
    
    Parameters:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.
        
    Returns:
        float: Euclidean distance.
    """
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    """
    Computes the Manhattan distance between two vectors.
    
    Parameters:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.
        
    Returns:
        float: Manhattan distance.
    """
    return np.sum(np.abs(a - b))

def cosine_similarity(a, b):
    """
    Computes the cosine similarity between two vectors.
    
    Parameters:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.
        
    Returns:
        float: Cosine similarity.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

if __name__ == '__main__':
    # Test the utility functions
    X = np.array([[1, 2], [3, 4], [5, 6]])
    print("Original X:\n", X)
    print("X with bias term:\n", add_bias(X))
    print("Normalized X:\n", normalize_features(X))
    print("Min-max scaled X:\n", min_max_scale(X))
    
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print("\nEuclidean distance between a and b:", euclidean_distance(a, b))
    print("Manhattan distance between a and b:", manhattan_distance(a, b))
    print("Cosine similarity between a and b:", cosine_similarity(a, b))
