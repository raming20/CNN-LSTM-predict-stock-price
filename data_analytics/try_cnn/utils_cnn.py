import numpy as np
from sklearn.preprocessing import OneHotEncoder


def output_size(X_shape, F_shape, s, p_row, p_col) -> tuple:
    output_row = int((X_shape[0] + 2*p_row - F_shape[0]) / s) + 1
    output_col = int((X_shape[1] + 2*p_col - F_shape[1]) / s) + 1
    return output_row, output_col


def convolution(X, F, s=1, p_row=0, p_col=0) -> np.ndarray:
    assert len(X.shape) == len(
        F.shape), "X and F must have the same number of dimensions."

    X_padded = np.zeros(
        (X.shape[0] + 2*p_row, X.shape[1] + 2*p_col) + X.shape[2:])

    X_padded[p_row:X_padded.shape[0]-p_row, p_col:X_padded.shape[1]-p_col] = X

    output_row, output_col = output_size(X.shape, F.shape, s, p_row, p_col)
    output = np.zeros((int(output_row), int(output_col)))

    start_row = 0
    output_row = 0
    while start_row + F.shape[0] <= X_padded.shape[0] and output_row < output.shape[0]:
        start_col = 0
        output_col = 0
        while start_col + F.shape[1] <= X_padded.shape[1] and output_col < output.shape[1]:
            o = X_padded[start_row:start_row + F.shape[0],
                         start_col:start_col + F.shape[1]] * F
            result = np.sum(o)
            output[output_row][output_col] = result
            start_col += s
            output_col += 1

        start_row += s
        output_row += 1

    return output


def convolution_multiple_kernel(X, Fs, Bs, s=1, p_row=0, p_col=0) -> np.ndarray:
    
    output_row, output_col = output_size(X.shape, Fs[0].shape, s, p_row, p_col)
    output = np.zeros((int(output_row), int(output_col), len(Fs)))
    
    for i, F in enumerate(Fs):
        output[:,:,i] = convolution(X, F, s, p_row, p_col) + Bs[i]
    
    return output
        
  
def pooling(X, size, operator=np.max, s=1) -> np.ndarray:
    assert len(size) == 2, "size must be 2 dimensional"

    output_row = int((X.shape[0] - size[0]) / s) + 1
    output_col = int((X.shape[1] - size[1]) / s) + 1

    output = np.zeros((int(output_row), int(output_col)) + X.shape[2:])

    start_row = 0
    output_row = 0
    while start_row + size[0] <= X.shape[0] and output_row < output.shape[0]:
        start_col = 0
        output_col = 0
        while start_col + size[1] <= X.shape[1] and output_col < output.shape[1]:
            o = operator(X[start_row:start_row + size[0],
                         start_col:start_col + size[1]], axis=(0, 1))
            output[output_row][output_col] = o
            start_col += s
            output_col += 1

        start_row += s
        output_row += 1

    return output


def glorot_uniform_initialization(size) -> np.ndarray:
    fan_in = size[0]
    fan_out = size[1]
    
    limit = np.sqrt(6 / (fan_in + fan_out))
    
    weights = np.random.uniform(-limit, limit, size=size)
    return weights

def he_initialization(size) -> np.ndarray:
    fan_in = size[0]
    
    stddev = np.sqrt(2 / fan_in)
    
    weights = np.random.normal(0, stddev, size=size)
    return weights

def init_kernel(size, method=glorot_uniform_initialization) -> np.ndarray:
    return method(size)

def init_bias_kernel() -> int:
    return 0

def ReLU(X) -> np.ndarray:
    return np.maximum(X, 0)

def init_weights(size, method=he_initialization) -> np.ndarray:
    return method(size)

def init_bias(size) -> np.ndarray:
    return np.zeros(shape=size)

def softmax(Z: np.ndarray) -> np.ndarray:
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    softmax_Z = exp_Z / exp_Z.sum(axis=0)
    return softmax_Z

def one_hot(y: np.ndarray) -> np.ndarray:
    one_hot = OneHotEncoder(sparse_output=False)
    y_one_hot = one_hot.fit_transform(y.reshape(-1, 1))
    return y_one_hot.T

def cross_entropy_loss(y: np.ndarray, y_predict: np.ndarray):
    """ Deprecated, use calculate_cross_entropy_loss_without_softmax_value instead. """
    loss = (-1/len(y)) * np.sum(y * np.log(y_predict))
    return loss

def calculate_cross_entropy_loss_without_softmax_value(y: np.ndarray, Z: np.ndarray):
    N = y.shape[1]
    
    Z_minimize = Z - np.max(Z, axis=0, keepdims=True)
    e_Z = np.exp(Z_minimize)
    sum_e_Z_col = np.sum(e_Z, axis=0, keepdims=True)
    log_A = Z_minimize - np.log(sum_e_Z_col)
    return (-1 / N) * np.sum(y * log_A)

def derivative_ReLU(X: np.ndarray):
    return np.where(X > 0, 1, 0)


def derivative_layer_before_pooling(C: np.ndarray, P: np.ndarray, dL_dP: np.ndarray, size: tuple, s: int):
    """ Deprecated, use derivative_layer_before_max_pooling_for_multiple_point instead. """
    
    assert len(size) == 2, "size must be 2 dimensional"

    output = np.zeros(C.shape)

    start_row = 0
    pool_row = 0
    while start_row + size[0] <= C.shape[0] and pool_row < output.shape[0]:
        start_col = 0
        pool_col = 0
        while start_col + size[1] <= C.shape[1] and pool_col < output.shape[1]:

            slide_window = C[start_row:start_row +
                              size[0], start_col:start_col + size[1]]
            output[start_row:start_row + size[0], start_col:start_col + size[1]] = np.where(
                slide_window == P[pool_row][pool_col], dL_dP[pool_row][pool_col], 0)

            start_col += s
            pool_col += 1

        start_row += s
        pool_row += 1

    return output


def derivative_layer_before_max_pooling_for_multiple_point(list_C: list[np.ndarray], list_P: list[np.ndarray], list_dL_dP: list[np.ndarray], size: tuple, s: int):
    assert len(size) == 2, "size must be 2 dimensional"

    N = len(list_C)
    
    output = np.zeros([N] + list(list_C[0].shape))
    
    C_row = list_C[0].shape[0]
    C_col = list_C[0].shape[1]
    
    P_row = list_P[0].shape[0]
    P_col = list_P[0].shape[1]

    start_row = 0
    pool_row = 0
    while start_row + size[0] <= C_row and pool_row < P_row:
        start_col = 0
        pool_col = 0
        while start_col + size[1] <= C_col and pool_col < P_col:

            slide_window = list_C[:, 
                                  start_row:start_row + size[0], 
                                  start_col:start_col + size[1]
                                ]
            output[:, 
                   start_row:start_row + size[0], 
                   start_col:start_col + size[1]
                ] = np.where(
                    slide_window == list_P[:, pool_row:pool_row+1, pool_col:pool_col+1], 
                    list_dL_dP[:, pool_row:pool_row+1, pool_col:pool_col+1], 
                    0
                )

            start_col += s
            pool_col += 1

        start_row += s
        pool_row += 1

    return output
    
  