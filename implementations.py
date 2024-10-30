import numpy as np
from utils import compute_loss_mse, compute_gradient_mse, compute_loss_mle, compute_gradient_mle, split_data
from metrics import f1_score


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression algorithm using gradient descent.

    Args:
        y (_type_): numpy array of shape=(N, )
        tx (_type_): numpy array of shape=(N, D+1)
        initial_w (_type_): numpy array of shape=(D+1, ). The initialization for the model parameters
        max_iters (_type_): a scalar denoting the total number of iterations of GD
        gamma (_type_): a scalar denoting the stepsize.
        
    Returns:
        loss: loss value (scalar), corresponding to the input parameters w
        w: model parameters as numpy array of shape (D+1, ).
    """
    
    # Initialize the model parameters
    w = initial_w
    
    # Iterate over the number of max iterations
    for n_iter in range(max_iters):
        # Compute the gradient with respect to w
        gradient = compute_gradient_mse(y, tx, w)
        
        # Update the model parameters
        w = w - gamma * gradient
        
    # Compute the final loss after all iterations
    loss = compute_loss_mse(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression algorithm using stochastic gradient descent.

    Args:
        y (_type_): numpy array of shape=(N, )
        tx (_type_): numpy array of shape=(N, D+1)
        initial_w (_type_): numpy array of shape=(D+1, ). The initialization for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters (_type_): a scalar denoting the total number of iterations of GD
        gamma (_type_): a scalar denoting the stepsize.
        
    Returns:
        loss: loss value (scalar), corresponding to the input parameters w
        w: model parameters as numpy array of shape (D+1, ).
    """
    
    # Initialize the model parameters
    w = initial_w
    
    # Number of samples N
    N = len(y)
    
    # Iterate over the number of max iterations
    for n_iter in range(max_iters):
        # Select a random data sample i among the N
        i = np.random.randint(0, N)
        
        # Compute the gradient with respect to w using the single data sample i
        gradient = compute_gradient_mse(np.array([y[i]]), np.array([tx[i]]), w)
        
        # Update the model parameters
        w = w - gamma * gradient
        
    # Compute the final loss after all iterations
    loss = compute_loss_mse(y, tx, w)

    return w, loss
    
    
def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y (_type_): numpy array of shape (N, )
        tx (_type_): numpy array of shape (N, D+1)
        
    Returns:
        w: optimal weights, numpy array of shape(D+1,)
    """
    
    # Compute the optimal weights using the normal equations
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    
    # Compute the MSE loss with thoptimal weights
    loss = compute_loss_mse(y, tx, w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y (_type_): numpy array of shape=(N, )
        tx (_type_): numpy array of shape=(N, D+1)
        lambda_ (_type_): a scalar denoting the regularization (penalty) term
    """
    
    # Compute the optimal weights using the normal equations
    w = np.linalg.solve(tx.T @ tx + lambda_* 2 * y.shape[0] * np.identity(tx.shape[1]), tx.T @ y)
    
    # Compute the MSE loss with thoptimal weights
    loss = compute_loss_mse(y, tx, w)
    
    return w, loss
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression algorithm using gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D+1)
        initial_w: numpy array of shape=(D+1, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize.
        
    Returns:
        loss: loss value (scalar), corresponding to the input parameters w
        w: model parameters as numpy array of shape (D+1, ).
    """
    
    # Initialize the model parameters
    w = initial_w
    
    # Iterate over the number of max iterations
    for n_iter in range(max_iters):
        # Compute the gradient with respect to w
        gradient = compute_gradient_mle(y, tx, w)
        
        # Update the model parameters
        w = w - gamma * gradient
    
    # Compute the final loss after all iterations
    loss = compute_loss_mle(y, tx, w)
    
    return w, loss
    
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y (numpy array): Labels of shape (N, ) where N is the number of data points
        tx (numpy array): Features of shape (N, D+1) where D is the number of features (with bias term)
        lambda_ (float): Regularization parameter (penalty term)
        initial_w (numpy array): Initial weights of shape (D+1,)
        max_iters (int): Number of iterations for gradient descent
        gamma (float): Step size (learning rate)
        
    Returns:
        w (numpy array): The optimized weight vector of shape (D+1,)
        loss (float): The regularized logistic loss (cross-entropy) after all iterations
    """
    
    # Initialize weights
    w = initial_w
    
    # Iterate over the number of max iterations
    for n_iter in range(max_iters):
        # Compute the gradient of the regularized logistic loss
        gradient = compute_gradient_mle(y, tx, w) + 2 * lambda_ * w
        
        # Update the weights
        w = w - gamma * gradient
    
    # Compute the final loss
    loss = compute_loss_mle(y, tx, w)
    
    return w, loss

def poly_features(tx, degree):
    tx_poly = tx
    for d in range(2, degree + 1):
        tx_poly = np.c_[tx_poly, np.power(tx, d)]
        
    return tx_poly

def cross_validation_ridge_regression(yb, tx, lambdas, degrees):
    """
    Cross-validate ridge regression with different hyperparameters.

    Args:
        yb (np.ndarray): Labels of the data.
        tx (np.ndarray): Features of the data.
        lambdas (list): List of regularization parameters to test.
        degrees (list): List of polynomial degrees to test.

    Returns:
        dict: Dictionary containing the best hyperparameters, F1 score, and weights.
    """
    best_lambda = None
    best_degree = None
    best_f1_score = -1
    best_w = None

    for degree in degrees:
        # Generate polynomial features
        x_poly = poly_features(tx, degree)

        # Split data for training and testing
        tx_train, tx_test, y_train, y_test = split_data(x_poly, yb, 0.8, seed=1)

        for lambda_ in lambdas:
            # Train the model
            w, _ = ridge_regression(y_train, tx_train, lambda_)

            # Predict and calculate F1 score
            y_pred = np.where(tx_test @ w >= 0.5, 1, -1)
            f1 = f1_score(y_test, y_pred)

            # Update best hyperparameters if this F1 score is higher
            if f1 > best_f1_score:
                best_lambda = lambda_
                best_degree = degree
                best_f1_score = f1
                best_w = w

    # Return best hyperparameters and associated metrics
    return {
        "lambda": best_lambda,
        "degree": best_degree,
        "f1_score": best_f1_score,
        "weights": best_w,
    }