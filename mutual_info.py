#!usr/bin/env python3


# Returns the mutual information between two features, unique_rows contains the unique rows inside the two features,
# joint is the joint pmf and marginal_x and marginal_y are the pmf of the features treated

def mutual_information(joint, unique_rows, marginal_x, marginal_y):

    from numpy import log2 as log2

    mutual_info = 0                       # Mutual information of x and y
    rows, columns = joint.shape           # Number of rows and columns of data_matrix

    for i in range(0, rows):
        mutual_info += joint[i] * log2(joint[i] /
                                       (marginal_x[search_pmf(unique_rows[0], marginal_x)] *
                                        marginal_y[search_pmf(unique_rows[1], marginal_y)]))

    return mutual_info


def search_pmf(unique_row_value, unique_rows_a):
    for i in range(0, len(unique_rows_a)):
        if unique_rows_a[i] == unique_row_value:
            return i
