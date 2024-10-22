import numpy as np
from utils import *

def confusion_matrix(y_true, y_pred):
    """Compute the confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        
    Returns:
        numpy.ndarray: The confusion matrix.
    """
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    matrix = np.array([[TN, FP], [FN, TP]])
    return (matrix/np.sum(matrix))

def accuracy(y_true, y_pred):
    """
    Compute the accuracy.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    
    Returns:
        float: Accuracy.
    """
    matrix = confusion_matrix(y_true, y_pred)
    TN = matrix[0, 0]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TP = matrix[1, 1]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return accuracy

def precision(y_true, y_pred):
    """
    Compute the precision. 
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    
    Returns:
        float: precision.
    """
    matrix = confusion_matrix(y_true, y_pred)
    TP = matrix[1, 1]
    FP = matrix[0, 1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

def recall(y_true, y_pred):
    """
    Compute the recall (sensitivity).
    
    Args:
       y_true (np.ndarray): True labels.
       y_pred (np.ndarray): Predicted labels.
    
    Returns:
        float: Recall.
    """
    matrix = confusion_matrix(y_true, y_pred)
    TP = matrix[1, 1]
    FN = matrix[1, 0]
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall

def f1_score(y_true, y_pred):
    """
    Compute the F1 score.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    
    Returns:
        float: F1 score.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return f1

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using accuracy, precision, recall, and F1 score.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    
    Returns:
        dict: Dictionary containing the evaluation metrics.
    """
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }