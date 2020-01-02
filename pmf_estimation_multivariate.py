#!/usr/bin/env python3


import numpy as np


def pmf_multivariate(data_matrix):  # axis_index indicates axis to operate on
    rows, columns = data_matrix.shape           # Number of rows and columns of data_matrix
    unique_rows_array, pmf_vector = np.unique(data_matrix, axis=0, return_counts=True)
    return unique_rows_array, pmf_vector/rows
