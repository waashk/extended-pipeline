
from scipy.spatial import distance
import math
import numpy as np

def Dist(arr1, arr2):
	return distance.euclidean(arr1, arr2)

def log2(n):
	return math.log(n)/math.log(2)


def csr_matrix_equal2(a1, a2):
	return all((np.array_equal(a1.indptr, a2.indptr),
				np.array_equal(a1.indices, a2.indices),
				np.array_equal(a1.data, a2.data)))


	