import numpy as np
from helpers import batch_iter

# ---- Internal utility functions ----

def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    N = np.shape(tx)[0]
    e = y - np.dot(tx, w)
    loss = 1 / (2*N) * np.dot(e, e)
    # ***************************************************
    return loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    N = np.shape(tx)[0]
    e = y - np.dot(tx, w)
    grad = -1/N * np.dot(np.transpose(tx), e)
    # ***************************************************
    return grad


# ---- Required methods ----

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: a numpy arrays of shape (2, ), for the last iteration of GD
        loss: the loss value (scalar) for the last iteration of GD
    """

    # Define parameters to store w and loss
    w = initial_w
    loss = compute_mse(y, tx, w)

    for n_iter in range(max_iters):

        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_mse(y, tx, w)

        # Show w and loss
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The SGD algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
                w: a numpy arrays of shape (2, ), for the last iteration of SGD
        loss: the loss value (scalar) for the last iteration of SGD
    """

    batch_size = 1
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_mse(y, tx, w)

    for n_iter in range(max_iters):
         for minibatch_y, minibatch_tx in batch_iter(
                 y, tx, batch_size=batch_size, num_batches=1
         ):

            grad = compute_gradient(minibatch_y, minibatch_tx, w)

            w = w - gamma * grad
            loss = compute_mse(y, tx, w)

            # Show w and loss
            print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss

def least_squares(y, tx):
    """Least squares algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)

    Returns:
        w: a numpy arrays of shape (2, ), of the corresponding loss
        loss: the loss value (scalar) of the Least squares
    """
    A = np.transpose(tx) @ tx
    b = np.transpose(tx) @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)

    return w, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        lambda_: a scalar denoting the regularization parameter

    Returns:
        w: a numpy arrays of shape (2, ), of the corresponding loss
        loss: the loss value (scalar) of  the Ridge regression
    """
    A = np.transpose(tx) @ tx + 2 * y.shape[0] * lambda_ * np.eye(tx.shape[1])
    b = np.transpose(tx) @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of Logistic Regression
        gamma: a scalar denoting the stepsize

    Returns:
        w: a numpy arrays of shape (2, ), for the last iteration of Logistic regression
        loss: the loss value (scalar) for the last iteration of Logistic regression
    """
    # TODO

    return w, loss

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    """Regularized Logistic Regression algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        lambda_: a scalar denoting the regularization parameter
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of Logistic Regression
        gamma: a scalar denoting the stepsize

    Returns:
        w: a numpy arrays of shape (2, ), for the last iteration of Logistic regression
        loss: the loss value (scalar) for the last iteration of Logistic regression
    """
    # TODO

    return w, loss