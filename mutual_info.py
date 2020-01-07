#!usr/bin/env python3


# Returns the mutual information between two features. Unique_rows contains the unique rows inside the two features
# sub-matrix. Unique_rows_x and unique_rows_y contains the unique values inside the feature vector. Joint is the joint
# pmf and marginal_x and marginal_y are the pmf of the features treated
def mutual_information(joint, unique_rows, marginal_x, unique_rows_x, marginal_y, unique_rows_y):

    from numpy import log2 as log2
    mutual_info = 0                       # Mutual information of x and y

    # unique_rows, unique_rows_x and unique_rows_y have different dimensions, so a search for the correct index of these
    # last two is needed
    for i in range(0, len(joint)):
        # The index of marginals needs to be searched according to the value inside the two features submatrix
        mutual_info += joint[i] * log2(joint[i] /
                                       (marginal_x[search_pmf(unique_rows[i][0], unique_rows_x)] *
                                        marginal_y[search_pmf(unique_rows[i][1], unique_rows_y)]))
    return mutual_info


# Searches inside unique_rows_x of a certain feature the value equals to a certain element of the unique_rows sub-matrix
def search_pmf(unique_row_value, unique_rows_a):
    for i in range(0, len(unique_rows_a)):
        if unique_rows_a[i] == unique_row_value:
            return i
