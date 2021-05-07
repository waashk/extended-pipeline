
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

class MFCent(object):
	"""
	Implementation of
	"""
	def __init__(self, metric):

		self.metric = metric
		if metric == 'l2':
			self.metric = 'euclidean'



	def fit(self, X, y):

		self.X_train = X
		self.y_train = y
		
		#
		self.classes = sorted(map(int, list(set(self.y_train))))
		self.n_classes = len(self.classes)

		#
		self.docs_by_class = [len(np.where(self.y_train == i)[0]) for i in self.classes]

		#
		self.centroids = [np.mean(self.X_train[np.where(self.y_train == i)],axis=0,dtype=np.float64) for i in self.classes]

		#
		self.centroids_sum = [np.sum(self.X_train[np.where(self.y_train == i)],axis=0,dtype=np.float64) for i in self.classes]

		return self

	def csr_matrix_equal2(self, a1, a2):
		return all((np.array_equal(a1.indptr, a2.indptr),
					np.array_equal(a1.indices, a2.indices),
					np.array_equal(a1.data, a2.data)))

	def transform(self, X):

		#
		istrain = True if self.csr_matrix_equal2(self.X_train, X) else False

		#
		metafeatures = []

		#
		for i, doc in enumerate(X):
			for j in self.classes:
				#
				if (istrain and self.y_train[i] == j and self.docs_by_class[j]-1 > 1):
					c = (self.centroids_sum[j] - self.X_train[i])/(self.docs_by_class[j]-1)
				else:
					c = self.centroids[j]
				#
				if self.metric == 'l1' or self.metric == 'euclidean' or self.metric == 'l2':
					metafeatures += [cdist(X[i].toarray() ,c,metric=self.metric)[0][0]]
				if self.metric == 'cosine':
					metafeatures += [1.0 - cdist(X[i].toarray() ,c,metric=self.metric)[0][0]]
			

		return np.array(metafeatures).reshape((X.shape[0],self.n_classes))


