
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.neighbors import NearestNeighbors

class MFKnn(object):
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

		#
		self.docs_by_class = [len(np.where(self.y_train == i)[0]) for i in self.classes]

		#
		self.X_by_class = []
		self.knn_by_class = []
		#self.scores = {}

		#
		njobs=-1
		if self.metric == 'l1':
			njobs=1


		for i in self.classes:
			X_tmp = self.X_train[np.where(self.y_train == i)]
			#print ("xtmp"+str(X_tmp.shape[0])+" class: "+str(i))
			data=[]
			data.append(0)
			ind=[]
			ind.append(0)
			
			auxf=csr_matrix((data, (ind,ind)),     shape=(1,self.X_train.shape[1]),dtype=np.float64) #zero a linha
			if X_tmp.shape[0]<self.k+1:
				newxtmp=[]
				for iww in list(range(X_tmp.shape[0])):
					newxtmp.append(X_tmp[iww])
				for iww in list(range(self.k+1-X_tmp.shape[0])):
					newxtmp.append(auxf)
				X_tmp=vstack(newxtmp)

			knn = NearestNeighbors(n_neighbors=self.k+1, algorithm="brute", metric=self.metric, n_jobs=njobs)

			knn.fit(X_tmp)
			self.knn_by_class.append(knn)

		return self



	def csr_matrix_equal2(self, a1, a2):
		return all((np.array_equal(a1.indptr, a2.indptr),
					np.array_equal(a1.indices, a2.indices),
					np.array_equal(a1.data, a2.data)))

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


