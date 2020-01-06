#!/usr/bin/env python3


import numpy as np


# Returns the p.m.f. estimation of a certain feature of a dataset
def pmf_feature(data_matrix, matrix_axis):          # Axis of the feature
    rows, columns = data_matrix.shape               # Number of rows and columns of data_matrix
    unique_rows_array, pmf_vector = np.unique(data_matrix[:, matrix_axis], axis=0, return_counts=True)
    return unique_rows_array, pmf_vector/rows
