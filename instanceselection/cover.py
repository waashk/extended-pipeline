
import numpy as np
from instanceselection.aux import Dist, log2, csr_matrix_equal2
import mlpack
from scipy.sparse.csr import csr_matrix
import copy 

class Cover(object):
	"""
	Implementation of 
	"""
	def __init__(self, percentsel): #cosknn, coscent, l2knn, l2cent, k_neig: int = -1):

		self.percentsel = percentsel


	def emst(self, X):

		#Round for prevent Seg Fault
		#for line in X_train:
		#	X.append([round(float(x), 7) for x in line])
		X_round = [[round(float(x), 7) for x in line] for line in X]

		#emst computation
		emstfile = mlpack.emst(input=X_round)  
		return emstfile['output']


	def fit(self, X, y):

		self.X_train = X
		self.y_train = y 

		try:
			self.X_train = self.X_train.toarray()
		except:
			print("Already in array.")

		self.selsize = int(self.percentsel*self.X_train.shape[0])
		
		shape = self.X_train.shape
		self.n_docs = shape[0]
		self.n_features = shape[1]

		#TODO
		self.emstfile = self.emst(self.X_train)
		#np.save("teste/emst0", self.emstfile)
		#self.emstfile = np.load("teste/emst0.npy")

		pass

	def transform(self, X):

		try:
			self.X_train = self.X_train.toarray()
		except:
			print("Already in array.")

		if isinstance(X, np.ndarray) and not np.allclose(self.X_train, X):
			raise("Cover must to be applied only in the train set")

		if isinstance(X, csr_matrix) and not csr_matrix_equal2(self.X_train, X):
			raise("Cover must to be applied only in the train set")

		resultfolder = "./"
		row = 0
		unlabeledset = [] # unlabeled set with only feature values (this makes it easier to calculate distances later on)
		clustermap   = {} # % this hash makes it easier to know which cluster an instance belongs to during the cluster merge process
		clusters     = [] # an array of pointers to the clusters' list
		last         = [] # array containing a pointer to the last instance of each cluster (for easier cluster merging)
		size         = [] # array with the size of each cluster (makes it simple to decide the fastest way to merge 2 clusters; i.e. merge smaller cluster into larger one)
		labels       = [] # we keep an extra array with the labels
		qids         = [] # and the qids for each training set instance.    

		# read the training file and create arrays with the coordinates for each instance
		while row < len(X):

			allfeat = []
			for f in (range(self.n_features)):
				allfeat.append(X[row][f])

	        # array of arrays with every instance coordinates
			unlabeledset.append(allfeat)
	        # we keep the label
			labels.append(self.y_train[row])        
	        # Initially, each instance is a cluster. Here we initialize the arrays we need
			clusters.append([row, None])
	 		#initialize last instance pointer array
			last.append([row, None])
	 		#initialize cluster size array
			size.append(1)
	       #map the current instance to its own cluster
			clustermap[row] = row

			row+=1
	       
			if row % 1000 == 0:
				print("{} docs processed".format(row))


		print ("Total instances {}".format(len(X)))

		# a few control variables
		distances = []                   # @ contains the size of the last edge added to a cluster 
		maxd = 0						 # $
		mind = np.inf    				 # $
		clustercount = len(clustermap.keys())
		deletedclusters = {}			 # %
		percent = 100					 # $


		for linedist in self.emstfile:
		
			inst1, inst2, dist = int(linedist[0]), int(linedist[1]), linedist[2]

			if (clustermap[inst1] != clustermap[inst2]):
				source = 0
				dest = 0
				
				if size[clustermap[inst1]] >= size[clustermap[inst2]]:
					source = clustermap[inst2]
					dest = clustermap[inst1]
				else:
					source = clustermap[inst1]
					dest = clustermap[inst2]

				current = clusters[source]
				while current != None:
					clustermap[current[0]] = dest
					current = current[1]

				last[dest][1] = clusters[source]
				last[dest] = last[source]
				size[dest] += size[source]
		
				# delete $source cluster reference
				deletedclusters[source] = 1
				#undef($clusters[$source]);
				clusters[source] = [None, None]


			currsize = len(clusters)+1 - len(deletedclusters.keys())



			if currsize == self.selsize + 1:

				distances.append(dist)
				percent = int(currsize*100/len(clusters))
				centroids = [None for i in range(self.n_docs)]

				clustrel = []
				for i in range(len(clusters)):
					clustrel.append(-1)

				for i in range(len(clusters)):
					if(clusters[i][0] != None):
						if (clustrel[i] == -1):
							clustrel[i] = 0

						zerocentroid = []

						for j in range(self.n_features):
							zerocentroid.append(0)

						centroids[i] = copy.copy(zerocentroid)

						current = clusters[i]
						while current != None:
							if (labels[current[0]] > 0): 
								clustrel[i]+=1 # instance with label > 0
						
							for j in range(self.n_features):
								centroids[i][j] += unlabeledset[current[0]][j]

							current = current[1]

				X_train_new = np.zeros((self.selsize, self.n_features))
				y_train_new = []


				c1 = -1
				for i in range(len(clusters)):
					if clusters[i][0] != None:
						c1+=1

						for j in range(self.n_features):
							centroids[i][j] = centroids[i][j]/size[i]
						
						mindist = np.inf
						mininst = -1

						current = clusters[i]
						while (current != None):
							dist = Dist(unlabeledset[current[0]], centroids[i])
							if (dist < mindist):
								mindist = dist
								mininst = current[0]
							
							current = current[1]
						
						centroids[i][0] = None
						clustnrel = size[i] - clustrel[i] # calculate how many instances with label = 0
						
						y_train_new.append(int(labels[mininst]))

						# recreate instance in the original format using the feature values
						for j in range(self.n_features):
							feat = j
							X_train_new[c1][feat] = unlabeledset[mininst][j]
				
				centroids = None

			if (currsize < self.selsize + 1): 
				print("Cover: ", currsize)
				return X_train_new, y_train_new
