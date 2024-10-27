import numpy as np

from utils_cnn import (
	output_size_of_convolution,
	add_padding,
	add_padding_multiple_point,
	add_dilation,
	add_dilation_multiple_point,
	rotate_matrix_180_row,
	rotate_matrix_180_col,
	rotate_matrix_180,
	convolution,
	convolution_multiple_kernel,
	pooling,
	glorot_uniform_initialization,
	he_initialization,
	init_kernel,
	init_bias_kernel,
	ReLU,
	init_weights,
	init_bias,
	softmax,
	one_hot,
	cross_entropy_loss,
	calculate_cross_entropy_loss_without_softmax_value,
	convert_list_tensor_to_tensor,
    Sigmoid,
)


def derivative_ReLU(X: np.ndarray):
    """
    X is a tensor.
    Calculates the derivative of ReLU function for each elements in X.
    """
    return np.where(X > 0, 1, 0)


def derivative_Sigmoid(X: np.ndarray):
    """
    X is a tensor.
    Calculates the derivative of Sigmoid function for each elements in X.
    """
    s = Sigmoid(X)
    return s * (1 - s)


def derivative_layer_before_max_pooling_one_point(
        C: np.ndarray,
        P: np.ndarray,
        dL_dP: np.ndarray,
        size_pooling: tuple,
        stride_pooling: int):
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

    assert len(size_pooling) == 2, "size must be 2 dimensional"

    output = np.zeros(C.shape)

    C_row_run = 0
    pool_row_run = 0
    while C_row_run + size_pooling[0] <= C.shape[0] and pool_row_run < output.shape[0]:
        C_col_run = 0
        pool_col_run = 0
        while C_col_run + size_pooling[1] <= C.shape[1] and pool_col_run < output.shape[1]:

            slide_window = C[
                C_row_run: C_row_run + size_pooling[0],
                C_col_run: C_col_run + size_pooling[1]
            ]
            output[
                C_row_run: C_row_run + size_pooling[0],
                C_col_run: C_col_run + size_pooling[1]
            ] = np.where(
                slide_window == P[pool_row_run][pool_col_run],
                # Assigning elements in output that have values
                dL_dP[pool_row_run][pool_col_run],
                # equal to corresponding values in P the values of dL_dP
                0  # Assigning other elements in output value 0
            )

            C_col_run += stride_pooling
            pool_col_run += 1

        C_row_run += stride_pooling
        pool_row_run += 1

    return output


def derivative_layer_before_max_pooling_for_multiple_point(
        list_C: list[np.ndarray],
        list_P: list[np.ndarray],
        list_dL_dP: list[np.ndarray],
        size_pooling: tuple,
        stride_pooling: int):
    """
    This function applies the derivative for all points of data 
    (this version is extention of function derivative_layer_before_max_pooling_one_point for a chunk of points data)
    
    list_C is a list of tensor C (C in here same C in function derivative_layer_before_max_pooling_one_point) and assumed to have a size: (N, Rc, Cc, D)
    
    list_P is a list of tensor P (P in here same P in function derivative_layer_before_max_pooling_one_point) and assumed to have a size: (N, Rp, Cp, D)
    
    list_dL_dP is a list of tensor dL_dP (dL_dP in here same dL_dP in function derivative_layer_before_max_pooling_one_point) and assumed to have a size: (N, Rp, Cp, D)
    """

    assert len(size_pooling) == 2, "size must be 2 dimensional"

    # Number of points data len(list_C) = len(list_P) = len(list_dL_dP) = N
    N = len(list_C)

    first_point_C = list_C[0]
    output = np.zeros([N] + list(first_point_C.shape))

    C_row = first_point_C.shape[0]
    C_col = first_point_C.shape[1]

    first_point_P = list_P[0]
    P_row = first_point_P.shape[0]
    P_col = first_point_P.shape[1]

    C_row_run = 0
    pool_row_run = 0
    while C_row_run + size_pooling[0] <= C_row and pool_row_run < P_row:
        C_col_run = 0
        pool_col_run = 0
        while C_col_run + size_pooling[1] <= C_col and pool_col_run < P_col:

            slide_window = list_C[:,
                                  C_row_run: C_row_run + size_pooling[0],
                                  C_col_run: C_col_run + size_pooling[1]
                                  ]
            output[:,
                   C_row_run: C_row_run + size_pooling[0],
                   C_col_run: C_col_run + size_pooling[1]
                   ] = np.where(
                        slide_window == list_P[
                            :,
                            pool_row_run: pool_row_run+1,
                            pool_col_run: pool_col_run+1
                        ],
                        list_dL_dP[:,
                                pool_row_run: pool_row_run+1,
                                pool_col_run: pool_col_run+1
                                ],  # Assigning elements in output that have values
                        # equal to corresponding values in P the values of dL_dP
                        0  # Assigning other elements in output value 0
                    )

            C_col_run += stride_pooling
            pool_col_run += 1

        C_row_run += stride_pooling
        pool_row_run += 1

    return output


def derivative_for_filter_one_point(
        P: np.ndarray,
        dL_dZk_face_correspond_filter: np.ndarray,
        filter_size: tuple,
        stride_kernel=1,
        padding_row=0,
        padding_col=0):
    """ 
    Condition: convolution(P, filter) + bias = Zk
    
    P is layer before layer Zk in the CNN, P is a tensor of size: (Rp, Cp, Dp), with:
        Rp: number of rows in each face of tensor P
        Cp: number of cols in each face of tensor P
        Dp: deep of tensor or number of faces in tensor P
    
    We apply list of filters (D filters, each filter have size (Rf, Cf, Dp)) with P and 
    the output is tensor Zk with size: (Rzk, Czk, D), with:
    
        Rzk: number of rows in each face of tensor Zk
        Czk: number of cols in each face of tensor Zk
        D: deep of tensor or number of faces in tensor Zk
    
    So each face of Zk will correspond to each filter.
        
    dL_dZk_face_correspond_filter is derivative of loss value for a face of tensor 
    Zk that correspond to filter for which we want to calculate derivative. And 
    it is a matrix with size (Rzk, Czk).
    
    This function will duplicate dL_dZk_face_correspond_filter to convolution 
    it with each face of tensor P. 
    
    dL/d(one face of filter) = convolution(
                                    one face of P correspond to face of filter, 
                                    dL_dZk_face_correspond_filter
                                ).
    
    The output tensor has the same size as filter_size.
    """

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
        padding_row: number_P_padded_row-padding_row,
        padding_col: number_P_padded_col-padding_col
    ] = P

    output = np.zeros(filter_size)

    number_dL_dZk_row = dL_dZk_face_correspond_filter.shape[0]
    number_dL_dZk_col = dL_dZk_face_correspond_filter.shape[1]
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
                    P_row_run:P_row_run + number_dL_dZk_row,
                    P_col_run:P_col_run + number_dL_dZk_col,
                    i
                ] * dL_dZk_face_correspond_filter
                result = np.sum(o, axis=(0, 1))
                output[output_row_run, output_col_run,
                       i] = np.sum(result, axis=0)
            P_col_run += stride_kernel
            output_col_run += 1

        P_row_run += stride_kernel
        output_row_run += 1

    return output


def derivative_for_filter_multiple_point(
        list_P: list[np.ndarray],
        list_dL_dZk_face_correspond_filter: list[np.ndarray],
        filter_size: tuple,
        stride_kernel=1,
        padding_row=0,
        padding_col=0):
    """
    This function applies the derivative for all points of data 
    (this version is extention of function derivative_for_filter_one_point for a chunk of points data)
    
    list_P is a list of tensor P (P in here same P in function derivative_for_filter_one_point) and assumed to have a size: (N, Rp, Cp, Dp)
    
    list_dL_dZk_face_correspond_filter is a list of matrix dL_dZk_face_correspond_filter (dL_dZk_face_correspond_filter in here same 
    dL_dZk_face_correspond_filter in function derivative_for_filter_one_point) and assumed to have a size: (N, Rzk, Czk)
    
    This function calculates the derivative for each face of filter by calculate 
    convolution(one face of P correspond to face of filter, dL_dZk_face_correspond_filter) 
    in each point of data and then sum up all the result into a single matrix.
    
    The output tensor has the same size as filter_size.
    """

    N = len(list_P)  # Number of points data, len(list_P) = len(list_dL_dZk) = N
    first_point_P = list_P[0]
    number_P_row = first_point_P.shape[0]
    number_P_col = first_point_P.shape[1]

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
        padding_row: number_P_padded_row-padding_row,
        padding_col: number_P_padded_col-padding_col
    ] = list_P

    output = np.zeros(filter_size)

    first_dL_dZk_face_correspond_filter = list_dL_dZk_face_correspond_filter[0]
    number_dL_dZk_row = first_dL_dZk_face_correspond_filter.shape[0]
    number_dL_dZk_col = first_dL_dZk_face_correspond_filter.shape[1]
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
                ] * list_dL_dZk_face_correspond_filter
                result = np.sum(o, axis=(1, 2))  # The convolution result
                # Sum up convolution result of all points of data
                output[output_row_run, output_col_run,
                       i] = np.sum(result, axis=0)
            P_col_run += stride_kernel
            output_col_run += 1

        P_row_run += stride_kernel
        output_row_run += 1

    return output


def derivative_for_bias_kernel_one_point(dL_dZk_face_correspond_bias: np.ndarray):
    """
    Condition: convolution(P, filter) + bias = Zk 
    
    So each face of Zk will correspond to each bias.
        
    dL_dZk_face_correspond_bias is derivative of loss value for a face of tensor 
    Zk that correspond to bias for which we want to calculate derivative. And 
    it is a matrix with size (Rzk, Czk).
    
    This function will sum up all elements in a face of dL_dZk_face_correspond_bias that correspond to bias.
    
    dL/d_bias = np.sum(dL_dZk_face_correspond_bias).
    
    The output is a number.
    """

    return np.sum(dL_dZk_face_correspond_bias)


def derivative_for_bias_kernel_multiple_point(list_dL_dZk_face_correspond_bias: list[np.ndarray]):
    """
    This function applies the derivative for all points of data 
    (this version is extention of function derivative_for_bias_kernel_one_point for a chunk of points data)
    
    list_dL_dZk_face_correspond_bias is a list of matrix dL_dZk_face_correspond_bias in each point data (dL_dZk_face_correspond_bias in here same 
    dL_dZk_face_correspond_bias in function derivative_for_bias_kernel_one_point) and assumed to have a size: (N, Rzk, Czk)
    
    This function will sum up all elements of the list_dL_dZk_face_correspond_bias.
    
    dL/d_bias = np.sum(list_dL_dZk_face_correspond_bias).
    
    The output is a number.
    """

    return np.sum(list_dL_dZk_face_correspond_bias)


def derivative_layer_input_convolution_one_point(
        P: np.ndarray,
        list_Filter: list[np.ndarray],
        dL_dZk: np.ndarray,
        stride_default: int=1):
    """
	Condition: convolution(P, filter) + bias = Zk
    
    P is layer before layer Zk in the CNN, P is a tensor of size: (Rp, Cp, Dp), with:
        Rp: number of rows in each face of tensor P
        Cp: number of cols in each face of tensor P
        Dp: deep of tensor or number of faces in tensor P
        
    list_Filter is list of filters and assumed to have size: list[(Rf, Cf, Dp)] or (D, Rf, Cf, Dp).
    
    We apply list_Filter with P and the output is tensor Zk with size: (Rzk, Czk, D), with:
        Rzk: number of rows in each face of tensor Zk
        Czk: number of cols in each face of tensor Zk
        D: deep of tensor or number of faces in tensor Zk and also is number of filters in list_Filter
    
    So each face of Zk will correspond to each filter.
    
    Each face in each filter will be corresponding to each face of P.
        
    dL_dZk is derivative of loss value for Zk,
    it is a tensor with same size with Zk.
    
    This function will loop through each face of P and calculate the derivative of that face by
    equaltion:
    
    dL/d(one face of P) = np.sum(
        convolution( 
            padding( 
                dilation(each face in dL_dZk) 
            ), 
            rotate_matrix_180(
                each face in each filter (filter that correspond to face of dL_dZk) that corresponds to face of P 
            )
        )
    )
    
    The output tensor has the same size as P.
    """

    output = np.zeros(P.shape)

    first_filter = list_Filter[0]
    number_filter_row = first_filter.shape[0]
    number_filter_col = first_filter.shape[1]
    deep_of_filter = first_filter.shape[-1]
    number_faces_in_list_Filter = deep_of_filter * \
        len(list_Filter)  # total all faces in each filter in list

    dL_dZk_dilationed_padded = add_padding(add_dilation(
        dL_dZk, stride_default-1), number_filter_row - 1, number_filter_col - 1)
    list_Filter = [rotate_matrix_180(filter) for filter in list_Filter]

    list_Filter_concat = np.concatenate(list_Filter, axis=2)

    number_dL_dZk_dilationed_padded_row = dL_dZk_dilationed_padded.shape[0]
    number_dL_dZk_dilationed_padded_col = dL_dZk_dilationed_padded.shape[1]
    number_output_row = output.shape[0]
    number_output_col = output.shape[1]

    deep_of_P = P.shape[-1]
    dL_dZk_dilationed_padded_row_run = 0
    output_row_run = 0
    while dL_dZk_dilationed_padded_row_run + number_filter_row <= number_dL_dZk_dilationed_padded_row \
            and output_row_run < number_output_row:
        dL_dZk_dilationed_padded_col_run = 0
        output_col_run = 0
        while dL_dZk_dilationed_padded_col_run + number_filter_col <= number_dL_dZk_dilationed_padded_col \
                and output_col_run < number_output_col:
            for i in range(deep_of_P):
                faces_corresponding_in_list_Filter = list_Filter_concat[:,
                                                                        :,
                                                                        [e for e in range(
                                                                            i,
                                                                            number_faces_in_list_Filter,
                                                                            deep_of_filter)]
                                                                        ]
                o = dL_dZk_dilationed_padded[
                    dL_dZk_dilationed_padded_row_run:dL_dZk_dilationed_padded_row_run + number_filter_row,
                    dL_dZk_dilationed_padded_col_run:dL_dZk_dilationed_padded_col_run + number_filter_col,
                ] * faces_corresponding_in_list_Filter  # convolution
                output[output_row_run, output_col_run, i] = np.sum(o)

            dL_dZk_dilationed_padded_col_run += stride_default
            output_col_run += 1

        dL_dZk_dilationed_padded_row_run += stride_default
        output_row_run += 1

    return output


def derivative_layer_input_convolution_multiple_point(
        list_P: list[np.ndarray],
        list_Filter: list[np.ndarray],
        list_dL_dZk: list[np.ndarray],
        stride_default: int=1):
    """
    This function applies the derivative for all points of data 
    (this version is extention of function derivative_layer_input_convolution_one_point for a chunk of points data)
    
    list_P is a list of tensor P (P in here same P in function derivative_layer_input_convolution_one_point) 
    and assumed to have a size: (N, Rp, Cp, Dp)
    
    list_Filter is list of filters and assumed to have size: list[(Rf, Cf, Dp)] or (D, Rf, Cf, Dp).
    
    list_dL_dZk is a list of tensors dL_dZk (dL_dZk in here same 
    dL_dZk in function derivative_layer_input_convolution_one_point) and assumed to have a size: (N, Rzk, Czk)
    
    This function calculates the derivative for each face of P.
    
    The output tensor has the same size as P.
    """
    N = len(list_P)
    
    first_P = list_P[0]
    output = np.zeros([N] + list(first_P.shape))

    first_filter = list_Filter[0]
    number_filter_row = first_filter.shape[0]
    number_filter_col = first_filter.shape[1]
    deep_of_filter = first_filter.shape[-1]
    # total all faces in each filter in list
    number_faces_in_list_Filter = deep_of_filter * len(list_Filter)

    list_dL_dZk_dilationed_padded: list[np.ndarray] = add_padding_multiple_point(
        add_dilation_multiple_point(
            list_dL_dZk,
            stride_default-1
        ),
        number_filter_row - 1,
        number_filter_col - 1
    )
    list_Filter = [rotate_matrix_180(filter) for filter in list_Filter]

    list_Filter_concat = np.concatenate(list_Filter, axis=2)

    first_dL_dZk_dilationed_padded = list_dL_dZk_dilationed_padded[0]
    number_dL_dZk_dilationed_padded_row = first_dL_dZk_dilationed_padded.shape[0]
    number_dL_dZk_dilationed_padded_col = first_dL_dZk_dilationed_padded.shape[1]
    number_output_row = output.shape[0]
    number_output_col = output.shape[1]

    deep_of_P = first_P.shape[-1]
    dL_dZk_dilationed_padded_row_run = 0
    output_row_run = 0
    while dL_dZk_dilationed_padded_row_run + number_filter_row <= number_dL_dZk_dilationed_padded_row \
            and output_row_run < number_output_row:
        dL_dZk_dilationed_padded_col_run = 0
        output_col_run = 0
        while dL_dZk_dilationed_padded_col_run + number_filter_col <= number_dL_dZk_dilationed_padded_col \
                and output_col_run < number_output_col:
            for i in range(deep_of_P):
                list_faces_corresponding_in_list_Filter = list_Filter_concat[:,
                                                                        :,
                                                                        [e
                                                                         for e in
                                                                         range(i,
                                                                               number_faces_in_list_Filter,
                                                                               deep_of_filter
                                                                               )
                                                                         ]
                                                                        ]
                o = list_dL_dZk_dilationed_padded[
                    :,
                    dL_dZk_dilationed_padded_row_run:dL_dZk_dilationed_padded_row_run + number_filter_row,
                    dL_dZk_dilationed_padded_col_run:dL_dZk_dilationed_padded_col_run + number_filter_col,
                ] * list_faces_corresponding_in_list_Filter
                sum_in_each_face_of_dL_dZk_dilationed_padded = np.sum(o, axis=(1, 2), keepdims=True)
                output[:, 
                       output_row_run : output_row_run + 1, 
                       output_col_run : output_col_run + 1, 
                       i : i + 1] = np.sum(
                    sum_in_each_face_of_dL_dZk_dilationed_padded, 
                    axis=3,
                    keepdims=True
                )

            dL_dZk_dilationed_padded_col_run += stride_default
            output_col_run += 1

        dL_dZk_dilationed_padded_row_run += stride_default
        output_row_run += 1

    return output
