import numpy as np

def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE) for regression.
    
    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        
    Returns:
        float: MSE value.
    """
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE) for regression.
    
    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        
    Returns:
        float: MAE value.
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    Coefficient of Determination (R^2 score) for regression.
    
    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        
    Returns:
        float: R^2 score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def accuracy(y_true, y_pred):
    """
    Computes the classification accuracy.
    
    Parameters:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        
    Returns:
        float: Accuracy as a proportion.
    """
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    """
    Computes precision for binary classification.
    
    Parameters:
        y_true (np.ndarray): True class labels (0 or 1).
        y_pred (np.ndarray): Predicted class labels (0 or 1).
        
    Returns:
        float: Precision value.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    """
    Computes recall for binary classification.
    
    Parameters:
        y_true (np.ndarray): True class labels (0 or 1).
        y_pred (np.ndarray): Predicted class labels (0 or 1).
        
    Returns:
        float: Recall value.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    """
    Computes the F1 score for binary classification.
    
    Parameters:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        
    Returns:
        float: F1 score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

if __name__ == '__main__':
    # Test the regression metrics
    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8])
    
    print("Regression Metrics:")
    print("MSE:", mse(y_true_reg, y_pred_reg))
    print("MAE:", mae(y_true_reg, y_pred_reg))
    print("R^2 Score:", r2_score(y_true_reg, y_pred_reg))
    
    # Test the classification metrics
    y_true_cls = np.array([1, 0, 1, 1, 0, 1, 0])
    y_pred_cls = np.array([1, 0, 1, 0, 0, 1, 1])
    
    print("\nClassification Metrics:")
    print("Accuracy:", accuracy(y_true_cls, y_pred_cls))
    print("Precision:", precision(y_true_cls, y_pred_cls))
    print("Recall:", recall(y_true_cls, y_pred_cls))
    print("F1 Score:", f1_score(y_true_cls, y_pred_cls))
