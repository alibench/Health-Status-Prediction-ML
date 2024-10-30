# -*- coding: utf-8 -*-
"""Gradient Descent
"""

import numpy as np


def compute_loss_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        loss: the value of the loss (a scalar), corresponding to the input parameters w.
    """

    # Compute the error vector at w
    error = y - tx @ w

    # Compute the loss
    loss = (1 / (2 * y.shape[0])) * error.T @ error

    return loss


def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    # Compute the error vector at w
    error = y - tx @ w

    # Compute the gradient based on the error vector
    new_w = (-1 / y.shape[0]) * tx.T @ error

    return new_w


def logistic(t):
    """Apply sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))


def compute_loss_mle(y, tx, w):
    """Compute the negative log-likelihood loss for logistic regression (MLE).

    Args:
        y (_type_): numpy array of shape (N, ). Actual labels (0 or 1).
        tx (_type_): numpy array of shape (N, D+1). Input features with bias term.
        w (_type_): numpy array of shape (D+1, ). Weight vector.
<<<<<<< HEAD:utils.py

    Returns:
        loss (scalar): The mean negative log-likelihood loss.
    """
    # Compute the predicted probabilities
    y_pred = tx @ w

    # Apply the logistic function (sigmoid)
    y_pred_prob = 1 / (1 + np.exp(-y_pred))

    # Compute the log-likelihood loss
    loss = -np.mean(y * np.log(y_pred_prob) + (1 - y) * np.log(1 - y_pred_prob))

    return loss


def compute_gradient_mle(y, tx, w):
    """Compute the gradient of the negative log-likelihood loss for logistic regression.

    Args:
        y (_type_): numpy array of shape=(N,)
        tx (_type_): numpy array of shape=(N, D+1)
        w (_type_): numpy array of shape=(D+1, )

    Returns:
        gradient: numpy array of shape=(D+1,)
    """

    # Compute the predicted probabilities
    pred = logistic(tx @ w)

    # Compute the gradient of the loss
    gradient = tx.T @ (pred - y) / y.shape[0]

    return gradient

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """
    # set seed
    np.random.seed(seed)
    
    # shuffle indices
    shuffled = np.random.permutation(len(y))
    train_size = int(np.floor(ratio * len(y)))    

    # Split indices into training and testing
    train_indices = shuffled[:train_size]
    test_indices = shuffled[train_size:]
    
    # Use indices to split the data and labels
    x_tr = x[train_indices]
    x_te = x[test_indices]
    y_tr = y[train_indices]
    y_te = y[test_indices]
    
    return x_tr, x_te, y_tr, y_te