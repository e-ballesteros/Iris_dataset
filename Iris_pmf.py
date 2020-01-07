#!/usr/bin/env python3


from sklearn import datasets

from pmf_estimation_multivariate import pmf_multivariate
from pmf_estimation_feature import pmf_feature
from pmf_joint import pmf_joint

from entropy import entropy
from mutual_info import mutual_information


iris = datasets.load_iris()
data_matrix, class_vector = iris.data, iris.target
data_matrix_int = (10*data_matrix[:, 0:4]).astype(int)      # Discretization of samples as integers

# Computation of joint p.m.f. and unique rows inside the discrete matrix
unique_rows, pmf = pmf_multivariate(data_matrix_int)

# Computation of p.m.f. of each feature and unique elements inside each feature
unique_rows_0, pmf_0 = pmf_feature(data_matrix_int, 0)
unique_rows_1, pmf_1 = pmf_feature(data_matrix_int, 1)
unique_rows_2, pmf_2 = pmf_feature(data_matrix_int, 2)
unique_rows_3, pmf_3 = pmf_feature(data_matrix_int, 3)

print('The p.m.f. of the Sepal Length feature is:\n',
      pmf_0,
      '\ncorresponding to the unique values of the feature:\n',
      unique_rows_0)

print('\nThe p.m.f. of the Sepal Width feature is:\n',
      pmf_1,
      '\ncorresponding to the unique values of the feature:\n',
      unique_rows_1)

print('\nThe p.m.f. of the Petal Length feature is:\n',
      pmf_2,
      '\ncorresponding to the unique values of the feature:\n',
      unique_rows_2)

print('\nThe p.m.f. of the Petal Width feature is:\n',
      pmf_3,
      '\ncorresponding to the unique values of the feature:\n',
      unique_rows_3)

print('\nThe joint p.m.f. of the Iris dataset is:\n',
      pmf,
      '\ncorresponding to the unique rows:\n',
      unique_rows)

# Computation of entropies of each feature
print('\nThe entropy of the Sepal Length feature is: ', entropy(pmf_0))
print('\nThe entropy of the Sepal Width feature is: ', entropy(pmf_1))
print('\nThe entropy of the Petal Length feature is: ', entropy(pmf_2))
print('\nThe entropy of the Petal Width feature is: ', entropy(pmf_3))

# Computation of joint entropy
print('\nThe joint entropy of the the Iris dataset is: ', entropy(pmf))

# Computation of joint p.m.f. and unique rows inside the discrete matrix, of all the possible pairs of features
unique_rows_0_1, pmf_0_1 = pmf_joint(data_matrix_int, 0, 1)
unique_rows_0_2, pmf_0_2 = pmf_joint(data_matrix_int, 0, 2)
unique_rows_0_3, pmf_0_3 = pmf_joint(data_matrix_int, 0, 3)
unique_rows_1_2, pmf_1_2 = pmf_joint(data_matrix_int, 1, 2)
unique_rows_1_3, pmf_1_3 = pmf_joint(data_matrix_int, 1, 3)
unique_rows_2_3, pmf_2_3 = pmf_joint(data_matrix_int, 2, 3)

print('\nThe mutual information of the Sepal Length and Sepal Width features is: ',
      mutual_information(pmf_0_1, unique_rows_0_1, pmf_0, unique_rows_0, pmf_1, unique_rows_1))
print('\nThe mutual information of the Sepal Length and Petal Length features is: ',
      mutual_information(pmf_0_2, unique_rows_0_2, pmf_0, unique_rows_0, pmf_2, unique_rows_2))
print('\nThe mutual information of the Sepal Length and Petal Width features is: ',
      mutual_information(pmf_0_3, unique_rows_0_3, pmf_0, unique_rows_0, pmf_3, unique_rows_3))
print('\nThe mutual information of the Sepal Width and Petal Length features is: ',
      mutual_information(pmf_1_2, unique_rows_1_2, pmf_1, unique_rows_1, pmf_2, unique_rows_2))
print('\nThe mutual information of the Sepal Width and Petal Width features is: ',
      mutual_information(pmf_1_3, unique_rows_1_3, pmf_1, unique_rows_1, pmf_3, unique_rows_3))
print('\nThe mutual information of the Petal Length and Petal Width features is: ',
      mutual_information(pmf_2_3, unique_rows_2_3, pmf_2, unique_rows_2, pmf_3, unique_rows_3))



