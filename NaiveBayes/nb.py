import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None
        self.variances = None
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        """
        self.classes = np.unique(y)
        self.priors = {c: np.mean(y == c) for c in self.classes}
        self.means = {c: np.mean(X[y == c], axis=0) for c in self.classes}
        self.variances = {c: np.var(X[y == c], axis=0) for c in self.classes}
    
    def _gaussian_pdf(self, x, mean, var):
        """
        Compute the Gaussian Probability Density Function.
        """
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
    
    def predict(self, X):
        """
        Predict class labels for given input samples.
        """
        posteriors = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self._gaussian_pdf(x, self.means[c], self.variances[c])))
                class_probs[c] = prior + likelihood
            posteriors.append(max(class_probs, key=class_probs.get))
        return np.array(posteriors)
    
if __name__ == '__main__':
    # Sample dataset for Naive Bayes classification
    X_train = np.array([[1.0, 2.1], [1.5, 1.8], [5.0, 6.1], [6.2, 5.9], [1.1, 2.2], [6.8, 6.4]])
    y_train = np.array([0, 0, 1, 1, 0, 1])
    
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    
    X_test = np.array([[1.3, 2.0], [5.5, 6.0]])
    predictions = nb.predict(X_test)
    print("Naive Bayes Predictions:", predictions)
