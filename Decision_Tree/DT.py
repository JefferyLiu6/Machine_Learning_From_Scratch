import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Train the decision tree classifier.
        """
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """
        Recursively builds the decision tree.
        """
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return unique_classes[0]
        
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()
        
        left_idx = X[:, best_feature] < best_threshold
        right_idx = ~left_idx
        
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {"feature": best_feature, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}

    def _find_best_split(self, X, y):
        """
        Finds the best feature and threshold to split on.
        """
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] < threshold
                right_idx = ~left_idx
                
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                
                gini = self._gini_impurity(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature, threshold
        
        return best_feature, best_threshold
    
    def _gini_impurity(self, left, right):
        """
        Computes the Gini impurity for a split.
        """
        n = len(left) + len(right)
        return sum((len(group) / n) * (1 - sum((np.sum(group == c) / len(group)) ** 2 for c in np.unique(group))) for group in [left, right])
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse the decision tree to make a prediction.
        """
        if isinstance(node, dict):
            if x[node["feature"]] < node["threshold"]:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        return node

if __name__ == '__main__':
    # Sample dataset for Decision Tree classification
    X = np.array([[2, 3], [1, 1], [4, 5], [6, 7], [1, 4], [7, 8]])
    y = np.array([0, 0, 1, 1, 0, 1])
    
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    
    predictions = tree.predict(X)
    print("Decision Tree Predictions:", predictions)
