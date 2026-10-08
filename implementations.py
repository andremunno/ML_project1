import numpy as np

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


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return 1 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)

    return float(loss)

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    N = y.shape[0]
    # ***************************************************
    grad = 1 / N * np.transpose(tx) @ (sigmoid(tx @ w) - y)
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
    w = initial_w
    loss = compute_logistic_loss(y, tx, w)
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_logistic_loss(y, tx, w)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

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
    w = initial_w
    loss = compute_logistic_loss(y, tx, w)
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
        loss = compute_logistic_loss(y, tx, w)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))


    return w, loss