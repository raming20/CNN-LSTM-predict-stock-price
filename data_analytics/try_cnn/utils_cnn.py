import numpy as np
from sklearn.preprocessing import OneHotEncoder


def output_size_of_convolution(X_shape: tuple, F_shape: tuple, stride: int, padding_row: int, padding_col: int) -> tuple:
    output_row = int((X_shape[0] + 2*padding_row - F_shape[0]) / stride) + 1
    output_col = int((X_shape[1] + 2*padding_col - F_shape[1]) / stride) + 1
    return output_row, output_col


def add_padding(X, padding_row, padding_col):
    """
    Add padding to tensor X.
    
    X be assumed to having size: (R, C, D) with:
        R: number of rows in each face of tensor
        C: number of cols in each face of tensor
        D: deep of tensor or number of faces in tensor
        
    X also can be a matrix with size: (R, C) or (R, C, 1)
    """
    
    X_padded = np.zeros(
        (
            X.shape[0] + 2*padding_row, 
            X.shape[1] + 2*padding_col
        ) + X.shape[2:]
    )
    
    X_padded[
        padding_row : X_padded.shape[0]-padding_row,
        padding_col : X_padded.shape[1]-padding_col
    ] = X
    
    return X_padded


def add_padding_multiple_point(list_X: list[np.ndarray], padding_row: int, padding_col: int):
    """
    Add padding to list of tensors.
    
    This funciton loops over each of tensor in list and adds padding to the them.
    
    list_X be assumed to having size: (N, R, C, D) with:
        N: number of tensors in list
        R: number of rows in each face of a tensor
        C: number of cols in each face of a tensor
        D: deep of a tensor or number of faces in each tensor
    """
    
    N = len(list_X)
    
    first_point_X = list_X[0]
    number_X_row = first_point_X.shape[0]
    number_X_col = first_point_X.shape[1]
    
    list_X_padded = np.zeros(
        [N] +
        [
            number_X_row + 2*padding_row, 
            number_X_col + 2*padding_col
        ] + list(first_point_X.shape[2:])
    )
    
    first_point_X_padded = list_X_padded[0]
    number_X_padded_row = first_point_X_padded.shape[0]
    number_X_padded_col = first_point_X_padded.shape[1]
    
    list_X_padded[
        :,
        padding_row : number_X_padded_row-padding_row,
        padding_col : number_X_padded_col-padding_col
    ] = list_X
    
    return list_X_padded


def add_dilation(X: np.ndarray, dilation: int):
    """
    Add dilation to tensor X.
    
    X be assumed to having size: (R, C, D) with:
        R: number of rows in each face of tensor
        C: number of cols in each face of tensor
        D: deep of tensor or number of faces in tensor
        
    X also can be a matrix with size: (R, C) or (R, C, 1)
    """
    if dilation == 0: return X
    
    number_X_row = X.shape[0]
    number_X_col = X.shape[1]
    
    output = np.zeros(
        (
            number_X_row + (number_X_row - 1) * dilation,
            number_X_col + (number_X_col - 1) * dilation
        ) + X.shape[2:]
    )
    
    number_output_row = output.shape[0]
    number_output_col = output.shape[1]
    
    output_row_run = 0
    X_row_run = 0
    while output_row_run < number_output_row and X_row_run < number_X_row:
        output_col_run = 0
        X_col_run = 0
        while output_col_run < number_output_col and X_col_run < number_X_col:
            output[
                output_row_run : output_row_run + 1,
                output_col_run : output_col_run + 1
            ] = X[
                X_row_run : X_row_run + 1,
                X_col_run : X_col_run + 1,
            ]
            
            output_col_run += dilation + 1
            X_col_run += 1
        
        output_row_run += dilation + 1
        X_row_run += 1
    return output


def add_dilation_multiple_point(list_X: list[np.ndarray], dilation: int):
    """
    Add dilation to list of tensors.
    
    This funciton loops over each of tensor in list and adds dilation to the them.
    
    list_X be assumed to having size: (N, R, C, D) with:
        N: number of tensors in list
        R: number of rows in each face of a tensor
        C: number of cols in each face of a tensor
        D: deep of a tensor or number of faces in each tensor
    """
    
    N = len(list_X)
    
    first_point_X = list_X[0]
    number_X_row = first_point_X.shape[0]
    number_X_col = first_point_X.shape[1]
    
    list_output = np.zeros(
        [N] +
        [
            number_X_row + (number_X_row - 1) * dilation,
            number_X_col + (number_X_col - 1) * dilation
        ] + list(first_point_X.shape[2:])
    )
    
    number_output_row = list_output[0].shape[0]
    number_output_col = list_output[0].shape[1]
    
    output_row_run = 0
    X_row_run = 0
    while output_row_run < number_output_row and X_row_run < number_X_row:
        output_col_run = 0
        X_col_run = 0
        while output_col_run < number_output_col and X_col_run < number_X_col:
            list_output[
                :,
                output_row_run : output_row_run + 1,
                output_col_run : output_col_run + 1
            ] = list_X[
                :,
                X_row_run : X_row_run + 1,
                X_col_run : X_col_run + 1,
            ]
            
            output_col_run += dilation + 1
            X_col_run += 1
        
        output_row_run += dilation + 1
        X_row_run += 1
    return list_output


def rotate_matrix_180_row(X: np.ndarray):
    """
    Rotate each face of tensor X vertically (along the rows) that mean in each face: 
        new_face[last_row] = X_face[first_row], 
        new_face[last_row - 1] = X_face[first_row + 1],
        ...
    
    X be assumed to having size: (R, C, D) with:
        R: number of rows in each face of tensor
        C: number of cols in each face of tensor
        D: deep of tensor or number of faces in tensor
        
    X also can be a matrix with size: (R, C) or (R, C, 1)
    
    After rotation the size of tensor X is not changed.
    """
    
    output_rotate_row = np.zeros(X.shape)
    
    number_X_row = X.shape[0]
    
    first_row_run = 0
    last_row_run = number_X_row - 1
    while first_row_run < number_X_row:
        output_rotate_row[first_row_run, :] = X[last_row_run, :]
        
        first_row_run += 1
        last_row_run -= 1
        
    return output_rotate_row

def rotate_matrix_180_col(X: np.ndarray):
    """
    Rotate each face of tensor X horizontally (along the cols) that mean in each face: 
        new_face[last_col] = X_face[first_col], 
        new_face[last_col - 1] = X_face[first_col + 1],
        ...
        
    X be assumed to having size: (R, C, D) with:
        R: number of rows in each face of tensor
        C: number of cols in each face of tensor
        D: deep of tensor or number of faces in tensor
        
    X also can be a matrix with size: (R, C) or (R, C, 1)
    
    After rotation the size of tensor X is not changed.
    """
    
    output_rotate_col = np.zeros(X.shape)
    
    number_X_col = X.shape[1]
    
    first_col_run = 0
    last_col_run = number_X_col - 1
    while first_col_run < number_X_col:
        output_rotate_col[:, first_col_run] = X[:, last_col_run]
        
        first_col_run += 1
        last_col_run -= 1
        
    return output_rotate_col


def rotate_matrix_180(X):
    """
    Rotate each face of tensor X vertically first (along the rows) and after that rotate each face horizontally (along the cols).
    
    X be assumed to having size: (R, C, D) with:
        R: number of rows in each face of tensor
        C: number of cols in each face of tensor
        D: deep of tensor or number of faces in tensor
        
    X also can be a matrix with size: (R, C) or (R, C, 1)
    
    After rotation the size of tensor X is not changed.
    """
    return rotate_matrix_180_col(rotate_matrix_180_row(X))
    

def convolution(X: np.ndarray, Filter: np.ndarray, stride=1, padding_row=0, padding_col=0) -> np.ndarray:
    """
    Calculate convolution of tensor X and filter Filter (Filter is also a tensor)
    
    X be assumed to have size: (Rx, Cx, D) with:
        Rx: number of rows in each face of tensor X
        Cx: number of cols in each face of tensor X
        D: deep of tensor or number of faces in tensor X
        
    Filter be assumed to have size: (Rf, Cf, D) with:
        Rf: number of rows in each face of tensor Filter
        Cf: number of cols in each face of tensor Filter
        D: deep of tensor or number of faces in tensor Filter (is also in X)
        
    Number of faces in tensor X must be equal to number of faces in tensor Filter: D
    
    Output shape is a matrix (tensor with deep = 1): (Ro, Co, 1) or (Ro, Co) with:
        Ro: number of rows of matrix output 
        Co: number of cols of matrix output 
        Ro and Co be calculated by function output_size_of_convolution
    """
    
    assert len(X.shape) == len(
        Filter.shape), "X and F must have the same number of dimensions."

    X_padded = np.zeros(
        (
            X.shape[0] + 2*padding_row, 
            X.shape[1] + 2*padding_col
        ) + X.shape[2:]
    )

    X_padded[
        padding_row : X_padded.shape[0]-padding_row,
        padding_col : X_padded.shape[1]-padding_col
    ] = X

    output_row, output_col = output_size_of_convolution(X_padded.shape, Filter.shape, stride, padding_row, padding_col)
    output = np.zeros((int(output_row), int(output_col)))

    start_row = 0
    output_row = 0
    while start_row + Filter.shape[0] <= X_padded.shape[0] and output_row < output.shape[0]:
        start_col = 0
        output_col = 0
        while start_col + Filter.shape[1] <= X_padded.shape[1] and output_col < output.shape[1]:
            o = X_padded[
                        start_row:start_row + Filter.shape[0],
                        start_col:start_col + Filter.shape[1]
                        ] * Filter
            result = np.sum(o)
            output[output_row][output_col] = result
            start_col += stride
            output_col += 1

        start_row += stride
        output_row += 1

    return output


def convolution_multiple_kernel(X: np.ndarray, list_Filter: list[np.ndarray], list_Bias: list[int], stride=1, padding_row=0, padding_col=0) -> np.ndarray:
    """
    Calculate convolution of tensor X and list of filters (each filter is also a tensor).
    
    X be assumed to have size: (Rx, Cx, Dx) with:
        Rx: number of rows in each face of tensor X
        Cx: number of cols in each face of tensor X
        Dx: deep of tensor or number of faces in tensor X
        
    list_Filter is a list of filters that apply to tensor X. 
    
    We loop through each filter in list and calculate the 
    convolution between X and each filter, the result is saved to a face of tensor output.
    
    list_Filter be assumed to have size: (Nf, Rf, Cf, Df) or list[(Rf, Cf, Df)] with:
        Nf: number of filters to apply
        Rf: number of rows in each face of tensor Filter
        Cf: number of cols in each face of tensor Filter
        Df: deep of tensor or number of faces in tensor Filter
        
    Number of faces in tensor X must be equal to number of faces in each tensor filter: Dx = Df = D
    
    Output shape is a tensor with size: (Ro, Co, Nf) with:
        Ro: number of rows of matrix output 
        Co: number of cols of matrix output
        Nf: deep of tensor output or number of faces in tensor output and it is also number of filters in list_Filter
    
    Ro and Co be calculated by function output_size_of_convolution
    each face in output is result of convolution between X and each filter.
    """
    
    output_row, output_col = output_size_of_convolution(
        X.shape, list_Filter[0].shape, stride, padding_row, padding_col)
    output = np.zeros((int(output_row), int(output_col), len(list_Filter)))

    for i, F in enumerate(list_Filter):
        output[:, :, i] = convolution(
            X, F, stride, padding_row, padding_col) + list_Bias[i]

    return output
        
  
def pooling(X: np.ndarray, size: int, operator=np.max, stride=1) -> np.ndarray:
    """
    Apply pooling to tensor X.
    
    X be assumed to have size: (Rx, Cx, Dx) with:
        Rx: number of rows in each face of tensor X
        Cx: number of cols in each face of tensor X
        Dx: deep of tensor or number of faces in tensor X
        
    This function apply pooling to each face of tensor X, the result is a tensor has size: (Ro, Co, Dx) with:
        Ro: number of rows in each face of tensor output
        Co: number of cols in each face of tensor output
        Dx: deep of tensor or number of faces in tensor X
    """
    
    assert len(size) == 2, "size must be 2 dimensional"

    output_row = int((X.shape[0] - size[0]) / stride) + 1
    output_col = int((X.shape[1] - size[1]) / stride) + 1

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
            start_col += stride
            output_col += 1

        start_row += stride
        output_row += 1

    return output


def glorot_uniform_initialization(size) -> np.ndarray:
    """
    Initialize the values, usually for kernel (filter) in CNN.
    """
    
    fan_in = size[0]
    fan_out = size[1]
    
    limit = np.sqrt(6 / (fan_in + fan_out))
    
    weights = np.random.uniform(-limit, limit, size=size)
    return weights


def he_initialization(size) -> np.ndarray:
    """
    Initialize the values, usually for Weights in fully connected in CNN.
    """
    
    fan_in = size[0]
    
    stddev = np.sqrt(2 / fan_in)
    
    weights = np.random.normal(0, stddev, size=size)
    return weights


def init_kernel(size, method=glorot_uniform_initialization) -> np.ndarray:
    """
    Initialize the kernel (filter) values, default method is glorot_uniform_initialization.
    """
    
    return method(size)


def init_bias_kernel() -> int:
    """
    Initialize the bias kernel (filter) values.
    """
    return 0


def ReLU(X: np.ndarray) -> np.ndarray:
    """
    Calculate the ReLU value for all elements of the tensor (or matrix) X.
    The output tensor (or matrix) have same size as X.
    """
    return np.maximum(X, 0)


def init_weights(size, method=he_initialization) -> np.ndarray:
    """
    Initialize values for the weights in fully connected layers of CNN.
    """
    return method(size)


def init_bias(size) -> np.ndarray:
    """
    Initialize values for the bias in fully connected layers of CNN.
    """
    return np.zeros(shape=size)


def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Calculate the softmax value for each column in matrix Z (tensor with deep = 1).
    
    Each column in matrix Z represents a point of data.
    
    Z is assumed to have size: (Rz, Cz, 1) or (Rz, Cz), with:
        Rz: number of rows of matrix Z
        Cz: number of cols of matrix Z
        
    The output of this function has same size with Z: (Rz, Cz).
    """
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    softmax_Z = exp_Z / exp_Z.sum(axis=0)
    return softmax_Z


def one_hot(y: np.ndarray) -> np.ndarray:
    """
    One-hot coding for array y. Each element of the array is a point of data.
    
    The output is a matrix (tensor with deep = 1) with size: (Ro, Co), with:
        Ro: number of different values in array y. Example: y = [1, 2, 2, 3] => Ro = 3 because y have 3 different values: 1, 2, 3
        Co: has same value with len(y), represent number of points data
        
    Each column in the output matrix is a point of data that is one-hoted.
    """
    one_hot = OneHotEncoder(sparse_output=False)
    y_one_hot = one_hot.fit_transform(y.reshape(-1, 1))
    return y_one_hot.T


def cross_entropy_loss(Y_real: np.ndarray, Y_predict: np.ndarray):
    """ 
    Deprecated, use calculate_cross_entropy_loss_without_softmax_value instead. 
    
    When the values in matrix Y is very small, it can crash the np.log function (raise error and can't be calculated).
    
    Y_predict is the output layer of neural network. 
    Y_predict is predict value and is assumed to be a matrix with size: (R, C) or (R, C, 1), with:
        R: number of rows of matrix Y_predict
        C: number of cols of matrix Y_predict
    
    Y_real is real values and is assumed to be a matrix with size: (R, C) or (R, C, 1), with:
        R: number of rows of matrix Y_real (is also in Y_predict)
        C: number of cols of matrix Y_real (is also in Y_predict)
    
    Number of rows of matrix Y_predict must be equal to number of rows of matrix Y_real and represents amounts of datas.
    Each column in Y_predict corresponds to a column in Y_real and represents one point of data.
    
    Number of cols of matrix Y_predict must be equal to number of cols of matrix Y_real and represents the category into which data is classified.
    
    The output is a number of cross entropy loss.
    """
    
    N = len(Y_real)
    loss = (-1/N) * np.sum(Y_real * np.log(Y_predict))
    return loss


def calculate_cross_entropy_loss_without_softmax_value(Y_real: np.ndarray, Z: np.ndarray):
    """
    Calculates the cross entropy loss.
    
    Z is a matrix that belongs to the last layer of the neural network and softmax(Z) = Y_predict, 
    
    the size of Z is: (R, C) or (R, C, 1), with:
        R: number of rows in Z, it is the number of category features 
        into which data is classified.
        
        C: number of cols in Z, it is the amounts of point data.
    
    Y_real is real values and is assumed to be a matrix with size: (R, C) or (R, C, 1), with:
        R: number of rows in Y_real, it is the number of category features 
        into which data is classified (is also in Z)
        
        C: number of cols in Y_real, it is the amounts of point data (is also in Z)
    
    The output is a number of cross entropy loss.
    """
    
    N = Y_real.shape[1]
    
    Z_minimize = Z - np.max(Z, axis=0, keepdims=True)
    e_Z = np.exp(Z_minimize)
    
    # Sum all the values in e_Z in each column. The output is a array and each element of array is correspond to a column in e_Z.
    sum_e_Z = np.sum(e_Z, axis=0, keepdims=True)
    
    log_Y_predict = Z_minimize - np.log(sum_e_Z)
    return (-1 / N) * np.sum(Y_real * log_Y_predict)


def derivative_ReLU(X: np.ndarray):
    """
    X is a tensor.
    """
    return np.where(X > 0, 1, 0)


def derivative_layer_before_max_pooling_one_point(C: np.ndarray, P: np.ndarray, dL_dP: np.ndarray, size: tuple, stride: int):
    """ 
    Condition: pooling(C, operator=np.max) = P
    
    C is layer before layer P in the CNN, C is a tensor of size: (Rc, Cc, D), with:
        Rc: number of rows in each face of tensor C
        Cc: number of cols in each face of tensor C
        D: deep of tensor or number of faces in tensor C
    
    We apply max-pooling in each face of C and the output is tensor P with size: (Rp, Cp, D), with:
        Rp: number of rows in each face of tensor P
        Cp: number of cols in each face of tensor P
        D: deep of tensor or number of faces in tensor P (is also in C)
        
    dL_dP is derivative of loss value for P and it is a tensor with same size with P.
    
    This function loop through each slide window of tensor C (through all face of C) and 
    compare elements's value to corresponding value in P (face to face), assign the elements 
    that have values equals to corresponding value in P value of dL_dP, assigning 0 to other elements.
    
    The output tensor has the same size as C.
    """
    
    assert len(size) == 2, "size must be 2 dimensional"

    output = np.zeros(C.shape)

    C_row_run = 0
    pool_row_run = 0
    while C_row_run + size[0] <= C.shape[0] and pool_row_run < output.shape[0]:
        C_col_run = 0
        pool_col_run = 0
        while C_col_run + size[1] <= C.shape[1] and pool_col_run < output.shape[1]:

            slide_window = C[
                            C_row_run : C_row_run + size[0], 
                            C_col_run : C_col_run + size[1]
                            ]
            output[
                C_row_run : C_row_run + size[0], 
                C_col_run : C_col_run + size[1]
                ] = np.where(
                    slide_window == P[pool_row_run][pool_col_run], 
                    dL_dP[pool_row_run][pool_col_run], # Assigning elements in output that have values
                                                       # equal to corresponding values in P the values of dL_dP
                    0 # Assigning other elements in output value 0
                    )

            C_col_run += stride
            pool_col_run += 1

        C_row_run += stride
        pool_row_run += 1

    return output


def derivative_layer_before_max_pooling_for_multiple_point(list_C: list[np.ndarray], list_P: list[np.ndarray], list_dL_dP: list[np.ndarray], size: tuple, stride: int):
    """
    This function applies the derivative for all points of data 
    (this version is extention of function derivative_layer_before_max_pooling_one_point for a chunk of points data)
    
    list_C is a list of tensor C (C in here same C in function derivative_layer_before_max_pooling_one_point) and assumed to have a size: (N, Rc, Cc, D)
    list_P is a list of tensor P (P in here same P in function derivative_layer_before_max_pooling_one_point) and assumed to have a size: (N, Rp, Cp, D)
    list_dL_dP is a list of tensor dL_dP (dL_dP in here same dL_dP in function derivative_layer_before_max_pooling_one_point) and assumed to have a size: (N, Rp, Cp, D)
    """
    
    assert len(size) == 2, "size must be 2 dimensional"

    N = len(list_C) # Number of points data len(list_C) = len(list_P) = len(list_dL_dP) = N
    
    first_point_C = list_C[0]
    output = np.zeros([N] + list(first_point_C.shape))
    
    C_row = first_point_C.shape[0]
    C_col = first_point_C.shape[1]
    
    first_point_P = list_P[0]
    P_row = first_point_P.shape[0]
    P_col = first_point_P.shape[1]

    C_row_run = 0
    pool_row_run = 0
    while C_row_run + size[0] <= C_row and pool_row_run < P_row:
        C_col_run = 0
        pool_col_run = 0
        while C_col_run + size[1] <= C_col and pool_col_run < P_col:

            slide_window = list_C[:, 
                                  C_row_run : C_row_run + size[0], 
                                  C_col_run : C_col_run + size[1]
                                ]
            output[:, 
                   C_row_run : C_row_run + size[0], 
                   C_col_run : C_col_run + size[1]
                ] = np.where(
                    slide_window == list_P[
                                            :, 
                                            pool_row_run : pool_row_run+1, 
                                            pool_col_run : pool_col_run+1
                                        ], 
                    list_dL_dP[:, 
                               pool_row_run : pool_row_run+1, 
                               pool_col_run : pool_col_run+1
                            ],  # Assigning elements in output that have values
                                # equal to corresponding values in P the values of dL_dP
                    0 # Assigning other elements in output value 0 
                )

            C_col_run += stride
            pool_col_run += 1

        C_row_run += stride
        pool_row_run += 1

    return output


def derivative_for_filter_one_point(P: np.ndarray, dL_dZk: np.ndarray, filter_size: tuple, stride=1, padding_row=0, padding_col=0):

    number_P_row = P.shape[0]
    number_P_col = P.shape[1]
    
    P_padded: np.ndarray = np.zeros(
        [
            number_P_row + 2*padding_row, 
            number_P_col + 2*padding_col
        ] + list(P.shape[2:])
    )
    
    number_P_padded_row = P_padded.shape[0]
    number_P_padded_col = P_padded.shape[1]
    P_padded[
        padding_row : number_P_padded_row-padding_row,
        padding_col : number_P_padded_col-padding_col
    ] = P
    
    output = np.zeros(filter_size)
    
    
    first_point_dL_dZk = dL_dZk[0]
    number_dL_dZk_row = first_point_dL_dZk.shape[0]
    number_dL_dZk_col = first_point_dL_dZk.shape[1]
    number_output_row = output.shape[0]
    number_output_col = output.shape[1]
    
    deep_of_filter = filter_size[-1]
    P_row_run = 0
    output_row_run = 0
    while P_row_run + number_dL_dZk_row <= number_P_padded_row and output_row_run < number_output_row:
        P_col_run = 0
        output_col_run = 0
        while P_col_run + number_dL_dZk_col <= number_P_padded_col and output_col_run < number_output_col:
            for i in range(deep_of_filter):
                o = P_padded[
                            :,
                            P_row_run:P_row_run + number_dL_dZk_row,
                            P_col_run:P_col_run + number_dL_dZk_col,
                            i
                            ] * dL_dZk
                result = np.sum(o, axis=(1, 2))
                output[output_row_run, output_col_run, i] = np.sum(result, axis=0)
            P_col_run += stride
            output_col_run += 1

        P_row_run += stride
        output_row_run += 1

    return output


def derivative_for_filter_multiple_point(list_P: list[np.ndarray], list_dL_dZk: list[np.ndarray], filter_size: tuple, stride=1, padding_row=0, padding_col=0):
    N = len(list_P) # Number of points data, len(list_P) = len(list_dL_dZk) = N
    first_point_P = list_P[0]
    number_P_row = first_point_P.shape[0]
    number_P_col = first_point_P.shape[1]
    
    assert len(list_P[0].shape) == len(
        list_dL_dZk.shape), "X and F must have the same number of dimensions."
    
    list_P_padded: list[np.ndarray] = np.zeros(
        [N] + [
            number_P_row + 2*padding_row, 
            number_P_col + 2*padding_col
        ] + list(first_point_P.shape[2:])
    )
    
    first_point_P_padded = list_P_padded[0]
    number_P_padded_row = first_point_P_padded.shape[0]
    number_P_padded_col = first_point_P_padded.shape[1]
    list_P_padded[
        :,
        padding_row : number_P_padded_row-padding_row,
        padding_col : number_P_padded_col-padding_col
    ] = list_P
    
    output = np.zeros(filter_size)
    
    
    first_point_dL_dZk = list_dL_dZk[0]
    number_dL_dZk_row = first_point_dL_dZk.shape[0]
    number_dL_dZk_col = first_point_dL_dZk.shape[1]
    number_output_row = output.shape[0]
    number_output_col = output.shape[1]
    
    deep_of_filter = filter_size[-1]
    P_row_run = 0
    output_row_run = 0
    while P_row_run + number_dL_dZk_row <= number_P_padded_row and output_row_run < number_output_row:
        P_col_run = 0
        output_col_run = 0
        while P_col_run + number_dL_dZk_col <= number_P_padded_col and output_col_run < number_output_col:
            for i in range(deep_of_filter):
                o = list_P_padded[
                            :,
                            P_row_run:P_row_run + number_dL_dZk_row,
                            P_col_run:P_col_run + number_dL_dZk_col,
                            i
                            ] * list_dL_dZk
                result = np.sum(o, axis=(1, 2))
                output[output_row_run, output_col_run, i] = np.sum(result, axis=0)
            P_col_run += stride
            output_col_run += 1

        P_row_run += stride
        output_row_run += 1

    return output


def derivative_for_bias_kernel(list_dL_dZk: list[np.ndarray]):
    return np.sum(list_dL_dZk)


def convolution_two_list_matrix(list_X: list[np.ndarray], list_filter: list[np.ndarray], stride: int):
    pass


def derivative_layer_input_convolution(P: np.ndarray, list_Filter: list[np.ndarray], dL_dZk: np.ndarray, stride: int):
    """
        P: is layer input for convolution
        list_Filter: list of filter applied to P
        dL_dZk: derivative of layer output of convolution, here is Zk
        
        We have:
            conv(P, list_Filter) = Zk
    """
    
    output = np.zeros(P.shape)
    
    P_copy = P.copy()
    
    # Convert matrix to tensor with deep = 1
    if len(P.shape) == 2:
        P_copy = P_copy.reshape(list(P.shape) + [1])
    
    first_filter = list_Filter[0]
    dL_dZk_padded_dilationed = add_padding(add_dilation(dL_dZk, stride-1), first_filter.shape[0] - 1, first_filter.shape[1] - 1)
    list_Filter = [rotate_matrix_180(filter) for filter in list_Filter]
    
    def convert_list_filter_to_tensor(list_filter):
        first_filter = list_Filter[0]
        output = np.zeros([len(list_filter)] + list(first_filter.shape))
        for i, filter in enumerate(list_Filter):
            output[i] = filter
            
    list_Filter = convert_list_filter_to_tensor(list_Filter)
    
    deep_of_P_copy = P_copy.shape[-1]
    for i in range(deep_of_P_copy):
        pass
        
  