#!/usr/bin/env python3


import numpy as np


# Returns the joint p.m.f. estimation of a certain dataset
def pmf_multivariate(data_matrix):
    rows, columns = data_matrix.shape           # Number of rows and columns of data_matrix
    unique_rows_array, pmf_vector = np.unique(data_matrix, axis=0, return_counts=True)
    return unique_rows_array, pmf_vector/rows
