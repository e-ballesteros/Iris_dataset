#!/usr/bin/env python3


from sklearn import datasets
from pmf_estimation_multivariate import pmf_multivariate

iris = datasets.load_iris()
data_matrix, class_vector = iris.data, iris.target
data_matrix_int = (10*data_matrix[:, 0:4]).astype(int)

#unique_rows, pmf = pmf_multivariate(data_matrix_int)
# Es absurdo, solo hay uno que se repita, HAY QUE CALCULAR PMF DE CADA FEATURE POR SEPARADO
unique_rows_0, pmf_0 = pmf_multivariate(data_matrix_int, 0)
unique_rows_1, pmf_1 = pmf_multivariate(data_matrix_int, 1)
#unique_rows_2, pmf_2 = pmf_multivariate(data_matrix_int, 2)
#unique_rows_3, pmf_3 = pmf_multivariate(data_matrix_int, 3)

print(data_matrix_int)
#print(pmf)


