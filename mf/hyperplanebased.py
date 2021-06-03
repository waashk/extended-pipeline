
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.neighbors import NearestNeighbors

#TODO: Refatorar 
class MFHyperPlaneErrors(object):
	"""
	Implementation of 
	"""
	def __init__(self, metric, k):

		self.k = k
		self.metric = metric


	def fit(self, X, y):
		#
		self.X_train = X
		self.y_train = y
		
		#
		self.classes = sorted(map(int, list(set(self.y_train))))
		self.n_classes = len(self.classes)

		return self


	def csr_matrix_equal2(self, a1, a2):
		return all((np.array_equal(a1.indptr, a2.indptr),
					np.array_equal(a1.indices, a2.indices),
					np.array_equal(a1.data, a2.data)))

	#modificar
	def transform(self, X):

		#
		istrain = True if self.csr_matrix_equal2(self.X_train, X) else False
		#print(istrain)
		n_neighbors = self.k+1 if istrain else self.k

		metafeatures = []
		scores = {}

		for j in self.classes:
			if self.metric == "l1" or self.metric == "l2":
				scores[j] = 0.0 + self.knn_by_class[j].kneighbors(X, n_neighbors, return_distance=True)[0]
			if self.metric == "cosine":
				scores[j] = 1.0 - self.knn_by_class[j].kneighbors(X, n_neighbors, return_distance=True)[0]

		#
		for i, doc in enumerate(X):
			for j in self.classes:
				if istrain:
					if self.y_train[i] == j:
						metafeatures += list(scores[j][i][1:])
					else:
						metafeatures += list(scores[j][i][:-1])
				else:
					metafeatures += list(scores[j][i])


		return np.array(metafeatures).reshape((X.shape[0],self.k*self.n_classes))


#TODO: Refatorar
class MFHyperPlaneAcima(object):
	"""
	Implementation of 
	"""
	def __init__(self, metric, k):

		self.k = k
		self.metric = metric


	def fit(self, X, y):
		#
		self.X_train = X
		self.y_train = y
		
		#
		self.classes = sorted(map(int, list(set(self.y_train))))
		self.n_classes = len(self.classes)

		return self


	def csr_matrix_equal2(self, a1, a2):
		return all((np.array_equal(a1.indptr, a2.indptr),
					np.array_equal(a1.indices, a2.indices),
					np.array_equal(a1.data, a2.data)))

	#modificar
	def transform(self, X):

		#
		istrain = True if self.csr_matrix_equal2(self.X_train, X) else False
		#print(istrain)
		n_neighbors = self.k+1 if istrain else self.k

		metafeatures = []
		scores = {}

		for j in self.classes:
			if self.metric == "l1" or self.metric == "l2":
				scores[j] = 0.0 + self.knn_by_class[j].kneighbors(X, n_neighbors, return_distance=True)[0]
			if self.metric == "cosine":
				scores[j] = 1.0 - self.knn_by_class[j].kneighbors(X, n_neighbors, return_distance=True)[0]

		#
		for i, doc in enumerate(X):
			for j in self.classes:
				if istrain:
					if self.y_train[i] == j:
						metafeatures += list(scores[j][i][1:])
					else:
						metafeatures += list(scores[j][i][:-1])
				else:
					metafeatures += list(scores[j][i])


		return np.array(metafeatures).reshape((X.shape[0],self.k*self.n_classes))



