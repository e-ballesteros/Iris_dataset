#!/usr/bin/env python3


from sklearn import datasets
from pmf_estimation_multivariate import pmf_multivariate
from pmf_estimation_feature import pmf_feature

iris = datasets.load_iris()
data_matrix, class_vector = iris.data, iris.target
data_matrix_int = (10*data_matrix[:, 0:4]).astype(int)

unique_rows, pmf = pmf_multivariate(data_matrix_int)

unique_rows_0, pmf_0 = pmf_feature(data_matrix_int, 0)
unique_rows_1, pmf_1 = pmf_feature(data_matrix_int, 1)
unique_rows_2, pmf_2 = pmf_feature(data_matrix_int, 2)
unique_rows_3, pmf_3 = pmf_feature(data_matrix_int, 3)

print(data_matrix_int)
print(pmf_0)
print(unique_rows_0)
#print(pmf_1)
#print(pmf_2)
#print(pmf_3)


