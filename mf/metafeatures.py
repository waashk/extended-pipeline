
import numpy as np
from tqdm import tqdm

from mf.knnbased import MFKnn
from mf.centbased import MFCent
from mf.hyperplanebased import MFHyperPlaneErrors, MFHyperPlaneAcima


class MetaFeatures(object):
	"""
	Implementation of 
	"""
	
	def __init__(self, groups, k: int = -1): #cosknn, coscent, l2knn, l2cent, k_neig: int = -1):

		self.groups = groups
		self.k = k

		if len(groups) == 0:
			raise("Select at least one mf group")

		for (metric, approach) in groups:
			if approach == "knn":
				if self.k == -1:
					raise("Please define the k neighbor parameter")

	def fit(self, X, y):

		self.estimators = []

		for (metric, approach) in self.groups:
			
			if approach == 'knn':
				estimator = MFKnn(metric, self.k)
			elif approach == 'cent':
				estimator = MFCent(metric)
			#TODO: Remover apos refatorar codigo mf err
			elif approach == 'err':
				continue
			#elif approach == 'hyperplane_errors':
			#	#cos_knn_treino,cos_knn_teste = hyperplane_errors(X_treino, y_treino, X_teste, y_teste, n_classes, f, k, 'cosine', 'original', save=False)
			#	estimator = MFHyperPlaneErrors(metric, self.k)			
			#elif approach == 'hyperplane_acima':
        	#	#cos_knn_treino,cos_knn_teste = hyperplane_acima(X_treino, y_treino, X_teste, y_teste, n_classes, f, k, 'cosine', 'original', save=False)
			#	estimator = MFHyperPlaneAcima(metric, self.k)	


			self.estimators.append(estimator.fit(X,y))

	def transform(self, X):

		X_list = []

		for estimator in self.estimators:
			X_list.append(estimator.transform(X))

		X_all = X_list[0]

		for t in X_list:
			print(t.shape)


		if len(X_all)>1:
			for i in range(1,len(X_list)):
				X_all = np.hstack((X_all, X_list[i]))

		print(X_all.shape)

		print(type(X_all))
		#print(X_all[0])
		#print(np.nan in X)
		array_sum = np.sum(X_all)
		array_has_nan = np.isnan(array_sum)
		if array_has_nan:
			X_all = np.nan_to_num(X_all)
		
		return X_all