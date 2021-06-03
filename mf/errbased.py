


from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_selection import chi2



from operator import itemgetter

import random
from sklearn.utils import resample


	
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss

from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from scipy.sparse import coo_matrix, vstack
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
#from neighbors import cuKNeighborsSparseClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import dump_svmlight_file
from scipy.spatial.distance import cdist
import getopt
import sys
import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
import scipy
from sklearn import preprocessing
import numpy
import scipy.sparse
import hashlib
#import cPickle as pickle
#data_pickle = pickle.dumps(data)
#data_md5 = hashlib.md5(data_pickle).hexdigest()
from threading import Thread
from sklearn.preprocessing import MinMaxScaler


c_param=1
c_param_reg=[]
cent_param=[]

splits=5
regression_bias=1.1
classification_bias=0.3
reg_mult_bias=1

def get_data(dataset,tamanho):
	if tamanho != 0:
		data = load_svmlight_file(dataset, dtype=np.float64, n_features=tamanho)
	else:
		data = load_svmlight_file(dataset, dtype=np.float64)
	return data[0], data[1]


def get_sigmoid(score,sigmoid_param):
	E=np.exp(sigmoid_param*score)
	return 1/(1+E)


def ajusta_treino(X_train, y_train, n_classes, min):
	classes = list(range(n_classes))

	X_train2=X_train.copy()
	y_train2=y_train.copy()

	data=[]
	data.append(0)
	ind=[]
	ind.append(0)
	auxf=csr_matrix((data, (ind,ind)),     shape=(1,X_train.shape[1]),dtype=np.float64) #zero a linha
	
	for j in classes:

		num=len(np.where(y_train == j)[0])
		  
		if (num<min):
			print ("menor q o minimo")
			print( num)
			print( j)
			addaux=[]
			yaux=[]
			for i in range(0,min-num):
				addaux.append(auxf)
				yaux=yaux+[j]
			nullsaux=sparse.vstack(addaux)

			X_train2=sparse.vstack([X_train2, nullsaux], format='csr', dtype=np.float64)

			y_train2=list(y_train2)+yaux
			y_train2=np.array(y_train2).reshape((len(y_train2),))
			print( y_train)
			print( y_train2)
			print( y_train.shape)
			print( y_train2.shape)

	return X_train2, y_train2



def generate_RF(X_treino, y_treino,n_classes,classifier,params):


	#data_md5 = hashlib.md5(pickle.dumps(X_treino)).hexdigest()
	
	#if not os.path.exists("md5"+classifier+str(regcent)):
	#	os.makedirs("md5"+classifier+str(regcent))
	
	#data_md5="md5"+classifier+str(regcent)+"/"+data_md5
	
	#try:
	#	forests=pickle.load(open(data_md5, 'rb'))
	#	print( "aproveitando treino!")
	#	
	#except: 
	classes = list(range(n_classes))
	forests=[]
	
	X_treino2=X_treino.copy()

	for i in classes:


		y_tmp=[]
		y_regtmp=[]         
		for l in y_treino:
			if l==i:
				y_tmp.append(1)
				y_regtmp.append(regression_bias)

			else:
				y_tmp.append(0)

				y_regtmp.append(1)

		seed = 1
		if classifier=='rfreg':
			print( "gerando rf regression para classe "+str(i))
			forest = RandomForestRegressor(n_estimators = 200,max_features=0.3, oob_score=True, n_jobs=-1, random_state=seed, min_samples_leaf=1)
			forest.fit(X_treino2, y_tmp)
			forests.append(forest)

		if classifier=='svr':
			print( "gerando svr regression para classe "+str(i)+" c_param_reg: "+str(c_param_reg[i]))

			forest = LinearSVR(C=c_param_reg[i],epsilon=0.0001, dual=True)
			forest.fit(X_treino2, y_tmp)
			
			forests.append(forest)

		if classifier=='rf':
			print( "gerando rf classification para classe "+str(i))
			forest = RandomForestClassifier(n_estimators = 100,max_features=0.3, oob_score=True, n_jobs=-1, random_state=seed, min_samples_leaf=1)
			forest.fit(X_treino2, y_tmp)
			forests.append(forest)

		if classifier=='svm':
			print( "gerando svm para classe "+str(i)+" c_param "+str(c_param))

			forest2 = LinearSVC(C=c_param, random_state=0)

			forest2.fit(X_treino2, y_tmp)
			forests.append(forest2)

		if classifier=='svm':
			print( "gerando svm para classe "+str(i)+" c_param "+str(c_param))

			forest2 = LinearSVC(C=c_param, random_state=0)

			forest2.fit(X_treino2, y_tmp)
			forests.append(forest2)

		if classifier=='svmcentrf':
			print( "gerando svm para classe "+str(i)+" c_param "+str(params))
			forest2 = LinearSVC(C=params, random_state=0, class_weight='balanced')

			forest = CalibratedClassifierCV(forest2, method='sigmoid', cv=5)

			forest.fit(X_treino2, y_tmp)
			forests.append(forest)

		if classifier=='svmbagging':
			print( "gerando svm para classe "+str(i)+" c_param "+str(c_param))

			r=0
			forestvec=[]
			
			for ck in range(0,len(datasetbaggingids)):

				X2,y2=resample(X_treino2,y_tmp,random_state=r)

				cols_to_keep=datasetbaggingids[ck]
				X3 = X2[:, cols_to_keep]

				
				r=r+1
				forest = LinearSVC(C=c_param,random_state=r, class_weight='balanced')
				forest.fit(X3,y2)
				forestvec.append(forest)

			forests.append(forestvec)
			
			
	#pickle.dump(forests, open(data_md5, 'wb'))
	return forests



def hyperplane_errors(X_treino, y_treino, X_teste, k, metric = 'cosine'):#,grupos = 'original'):

	n_classes = len(sorted(map(int, list(set(y_treino)))))
	
	stackingtr, stackingte  = metric_cent_RF(X_treino, y_treino, X_teste, n_classes, 'svmcentrf')

	classes = list(range(n_classes))

	cos_knn_treino_ids, cos_knn_treino_scores, cos_knn_teste_ids,cos_knn_teste_scores, originalids = metric_knn(X_treino, y_treino, X_teste, n_classes, k, metric)
	
	auxmf=[]

	for i, doc in enumerate(X_treino):

		l=list(cos_knn_treino_scores[i])

		l.sort()

		avgerror=0.5
		counterror=1.0
		avgcorrect=0.5
		countcorrect=1.0
		propacerto=0.0
		tot=0.0
		magerro=0.0
		for j in classes:
			auxmf2=[]

			for z in list(range(j*k,(j+1)*k)):

				try:

					docind= originalids[j][cos_knn_treino_ids[i,z]]

					if cos_knn_treino_scores[i,z]<=l[k]: #se esta entyre os k mais proximos

						if stackingtr[docind,j] >0.5: # se eh um acerto
							propacerto=propacerto+1.0
						else:
							magerro=magerro+0.5-stackingtr[docind,j]
						tot=tot+1

				except:
					print( "probably no knn neighbors for class "+str(j))

		auxmf.append(propacerto/tot)

	auxmfte=[]

	for i, doc in enumerate(X_teste):
		l=list(cos_knn_teste_scores[i])

		l.sort()

		avgerror=0.5
		counterror=1.0
		avgcorrect=0.5
		countcorrect=1.0
		propacerto=0.0
		tot=0.0
		magerro=0.0

		for j in classes:
			auxmfte2=[]

			for z in list(range(j*k,(j+1)*k)):

				docclassification=-0.0666
				try:
					docind= originalids[j][cos_knn_teste_ids[i,z]]


					if cos_knn_teste_scores[i,z]<=l[k]:
						if stackingtr[docind,j] >0.5: # se eh um acerto
							propacerto=propacerto+1.0
						else:
							magerro=magerro+0.5-stackingtr[docind,j]

						tot=tot+1
						

				except:
					print ("probably no knn neighbors for class "+str(j))

		auxmfte.append(propacerto/tot)

	return  np.array(auxmf).reshape((X_treino.shape[0],1)),  np.array(auxmfte).reshape((X_teste.shape[0],1))




def hyperplane_acima(X_treino, y_treino, X_teste, k, metric = 'cosine'):#,grupos = 'original'):

	n_classes = len(sorted(map(int, list(set(y_treino)))))

	stackingtr, stackingte  = metric_cent_RF(X_treino, y_treino, X_teste, n_classes, 'svmcentrf')
	classes = list(range(n_classes))
	cos_knn_treino_ids, cos_knn_treino_scores, cos_knn_teste_ids,cos_knn_teste_scores, originalids = metric_knn(X_treino, y_treino, X_teste, n_classes, k, metric)
	
	auxmf=[]

	for i, doc in enumerate(X_treino):
		acima=0.0
		for j in classes:
			auxmf2=[]
			for z in list(range(j*k,(j+1)*k)):
				
				try:
					docind= originalids[j][cos_knn_treino_ids[i,z]]
					if stackingtr[i,j]>stackingtr[docind,j]:
						acima=acima+1.0         
				except:
					print( "probably no knn neighbors for class "+str(j))
			auxmf.append(acima/float(k))
			

	auxmfte=[]

	for i, doc in enumerate(X_teste):
		acima=0.0
		for j in classes:

			
			for z in list(range(j*k,(j+1)*k)):

				try:
					docind= originalids[j][cos_knn_teste_ids[i,z]]
					if stackingte[i,j]>stackingtr[docind,j]:
						acima=acima+1.0
				except:
					print( "probably no knn neighbors for class "+str(j))

			auxmfte.append(acima/float(k))

	return  np.array(auxmf).reshape((X_treino.shape[0],n_classes)),  np.array(auxmfte).reshape((X_teste.shape[0],n_classes))




def metric_knn(X_treino2, y_treino, X_teste2, n_classes, k, metric):
	classes = list(range(n_classes))

	X_treino=X_treino2
	X_teste=X_teste2

	docs_by_class = []
	X_by_class = []
	knn_by_class = []
	scores={}
	ids={}
	originalids=[]

	for i in classes:
		X_tmp = X_treino[np.where(y_treino == i)]

		originalids.append(list(np.where(y_treino == i)[0]))

		print( "xtmp"+str(X_tmp.shape[0])+" class: "+str(i))
		data=[]
		data.append(0)
		ind=[]
		ind.append(0)
		auxf=csr_matrix((data, (ind,ind)),     shape=(1,X_treino.shape[1]),dtype=np.float64) #zero a linha
		if X_tmp.shape[0]<k+1:
			newxtmp=[]
			for iww in list(range(X_tmp.shape[0])):
				newxtmp.append(X_tmp[iww])
			for iww in list(range(k+1-X_tmp.shape[0])):
				newxtmp.append(auxf)
			X_tmp=vstack(newxtmp)

		knn = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric, n_jobs=-1)
		knn.fit(X_tmp,[0]*X_tmp.shape[0])
		knn_by_class.append(knn)
		
	

		
	
	metafeatures = []
	metafeaturesids=[]


	for j in classes:

		scoresaux =   knn_by_class[j].kneighbors(X_treino, k+1, return_distance=True)
		scores[j]=scoresaux[0]
		ids[j]=scoresaux[1]

	#classes deve comecar com 0
	for i, doc in enumerate(X_treino):
		for j in classes:

			if y_treino[i] == j:  
				metafeatures += list(scores[j][i][1:])
				metafeaturesids += list(ids[j][i][1:])

			else:

				metafeatures += list(scores[j][i][:-1])
				metafeaturesids += list(ids[j][i][:-1])
				


	for j in classes:

		scoresaux =  knn_by_class[j].kneighbors(X_teste, k, return_distance=True)
		scores[j] =scoresaux[0]
		ids[j]=scoresaux[1]

	
	metafeatures_teste = []
	metafeatures_teste_ids = []

	for i, doc in enumerate(X_teste):
		for j in classes:

			metafeatures_teste += list(scores[j][i])
			metafeatures_teste_ids += list(ids[j][i])

	return np.array(metafeaturesids).reshape((X_treino.shape[0],k*n_classes)),np.array(metafeatures).reshape((X_treino.shape[0],k*n_classes)),  np.array(metafeatures_teste_ids).reshape((X_teste.shape[0],k*n_classes)),np.array(metafeatures_teste).reshape((X_teste.shape[0],k*n_classes)), originalids


def metric_cent_RF_singlesplit(X_treino, y_treino,X_teste, n_classes, classifier, params):
	classes = list(range(n_classes))
	docs_by_class = []
	forests=[]
	forests= generate_RF(X_treino, y_treino,n_classes,classifier,params)
	forestscv=[]

	mf_per_class=[]
	for j in classes:
		testintersect=[]
		for i, doc in enumerate(X_teste):
			testintersect.append(X_teste[i])

		vecttestintersect=vstack(testintersect)

		print ("classificando lista da classe"+str(j)+" dimensao "+str(vecttestintersect.shape))
		if classifier=='rf':
			mf_per_class.append(list(forests[j].predict_proba(vecttestintersect)[:,1] )) #pega os scores para os vetores de teste
		if classifier=='rfreg':
			mf_per_class.append(list(forests[j].predict(vecttestintersect) )) #pega os scores para os vetores de teste
		if classifier=='svr':
			mf_per_class.append(list(forests[j].predict(vecttestintersect) )) #pega os scores para os vetores de teste
		if classifier=='svm':
			mf_per_class.append(list(get_sigmoid(forests[j].decision_function(vecttestintersect)) , -1.0)) #pega os scores para os vetores de teste
		if classifier=='svmcentrf':
			mf_per_class.append(list(forests[j].predict_proba(vecttestintersect)[:,1] )) #pega os scores para os vetores de teste

	auxmf=[]
	for j in classes:
		if classifier!='svmbagging':
			auxj= np.array(mf_per_class[j]).reshape((X_teste.shape[0],1))

			auxmf.append(auxj)
		if classifier=='svmbagging':
			for ir in range(0,len(mf_per_class[j])):
				auxj= np.array(mf_per_class[j][ir]).reshape((X_teste.shape[0],1))
				
				auxmf.append(auxj)
				

	metafeatures_teste=np.hstack(auxmf)
	return  metafeatures_teste



	  
def metric_cent_RF(X_treino, y_treino, X_teste, n_classes, classifier):

	#if correcterrors==True:
	#	errortr,errorte = hyperplane_errors(X_treino, y_treino, X_teste, y_teste, n_classes, k, metric)

	outtr=np.zeros((X_treino.shape[0],1*n_classes))

	kf = StratifiedKFold(n_splits=splits,random_state=0)


	X_treino2=X_treino.copy()
	X_teste2=X_teste.copy()

	cos_cent_treino=[]
	cos_cent_teste=[]

	'''
	if grupos=='tudo':
		trids, cos_knnmf_treino,teids, cos_knnmf_teste, originalids = metric_knn(X_treino, y_treino, X_teste, y_teste, n_classes, f, k, metric, save=False)
		cos_knnmf2_treino, cos_knnmf2_teste = cos_cent_l2(X_treino, y_treino, X_teste, y_teste, n_classes, f, k, metric, save=False)
		cos_cent_treino, cos_cent_teste = metric_cent(X_treino, y_treino, X_teste, y_teste, n_classes, f, metric, save=False)
		X_treino2=sparse.hstack([ csr_matrix(cos_cent_treino),csr_matrix(cos_knnmf_treino),csr_matrix(cos_knnmf2_treino), X_treino2],format='csr')
		X_teste2=sparse.hstack([csr_matrix(cos_cent_teste),csr_matrix(cos_knnmf_teste),csr_matrix(cos_knnmf2_teste), X_teste2],format='csr')
	elif grupos=='knn':
		trids, cos_knnmf_treino,teids, cos_knnmf_teste, originalids = metric_knn(X_treino, y_treino, X_teste, y_teste, n_classes, f, k, metric, save=False)
		X_treino2=csr_matrix(cos_knnmf_treino)
		X_teste2= csr_matrix(cos_knnmf_teste)
	elif grupos=='coscentl2':
		cos_knnmf2_treino, cos_knnmf2_teste = cos_cent_l2(X_treino, y_treino, X_teste, y_teste, n_classes, f, k, metric, save=False)
		X_treino2=csr_matrix(cos_knnmf2_treino)
		X_teste2=csr_matrix(cos_knnmf2_teste)
	elif grupos=='cent':
		cos_cent_treino, cos_cent_teste = metric_cent(X_treino, y_treino, X_teste, y_teste, n_classes, f, metric, save=False)
		X_treino2= csr_matrix(cos_cent_treino)
		X_teste2=csr_matrix(cos_cent_teste)
	'''
	#elif grupos=='original':
	X_treino2=X_treino.copy()
	X_teste2=X_teste.copy()



	params=float(get_svm_params3(X_treino2, y_treino))
	
	for train_index, test_index in kf.split(X_treino2,y_treino):

		X_traincv, X_testcv = X_treino2[train_index], X_treino2[test_index]
		y_traincv, y_testcv = y_treino[train_index], y_treino[test_index]


		out1=metric_cent_RF_singlesplit(X_traincv, y_traincv, X_testcv, n_classes, classifier,  params)
		outtr[test_index]=out1


	out1=metric_cent_RF_singlesplit(X_treino2, y_treino, X_teste2, n_classes, classifier, params)

	'''
	if correcterrors==True:
	  
		for i in list(range(outtr.shape[0])):

			outtr[i]=outtr[i]-0.5
			outtr[i]=outtr[i]*errortr[i][0]     
			outtr[i]=outtr[i]+0.5

		for i in list(range(out1.shape[0])):
			out1[i]=out1[i]-0.5
			out1[i]=out1[i]*errortr[i][0]       
			out1[i]=out1[i]+0.5
	if correcterrors==True:
		outtr=np.concatenate([outtr, errortr],axis=1)
		out1=np.concatenate([out1, errorte],axis=1)
	'''
	return  outtr, out1


'''
def metric_cent(X_treino, y_treino,X_teste, y_teste, n_classes, f,metric,save=False):
	

	classes = list(range(n_classes))
	
	docs_by_class = []
	
	for i in classes:
		docs_by_class.append(len(np.where(y_treino == i)[0]))
	
	centroids = []
	for i in classes:
		centroids.append(csr_matrix.mean(X_treino[np.where(y_treino == i)],axis=0,dtype=np.float64))
	print( "ok1")

	centroids_sum = []
	for i in classes:
		centroids_sum.append(csr_matrix.sum(X_treino[np.where(y_treino == i)],axis=0,dtype=np.float64))
	#Treino
	print( "ok2")
	metafeatures_centroid = []
	for i, doc in enumerate(X_treino):
		for j in classes:
			if y_treino[i] == j:
				c = (centroids_sum[j] - X_treino[i])/(docs_by_class[j]-1)
			else:
				c = centroids[j]

			metafeatures_centroid += [ cdist(X_treino[i].toarray() ,c,metric=metric)[0][0]]

	metafeatures_centroid_teste = []
	

	for i, doc in enumerate(X_teste):
		for j in classes:
	
			c = centroids[j]

			metafeatures_centroid_teste += [  cdist(X_teste[i].toarray() ,c,metric=metric)[0][0]]


	return np.nan_to_num(np.array(metafeatures_centroid).reshape((X_treino.shape[0],n_classes))), np.nan_to_num(np.array(metafeatures_centroid_teste).reshape((X_teste.shape[0],n_classes)))


def get_svm_params (X_train, y_train, n_classes,f,k):

	best_score=0

	params=[]
	best=[]

	cos_cent_treino, cos_cent_teste = metric_cent(X_train, y_train, X_train, y_train, n_classes, f, 'cosine', save=False)
	trids, cos_knn_treino,teids, cos_knn_teste, originalids = metric_knn(X_train, y_train, X_train, y_train, n_classes, f, k, 'cosine', save=False)
	cos_knn2_treino, cos_knn2_teste = cos_cent_l2(X_train, y_train, X_train, y_train, n_classes, f, k, 'cosine', save=False)
	
	
	X_treino2=sparse.hstack([X_train, csr_matrix(cos_cent_treino),csr_matrix(cos_knn_treino),csr_matrix(cos_knn2_treino)],format='csr')

	tuned_parameters = [{'C': [ 0.03125, .25000, 2, 16, 128, 1024, 8192]}] 

	clf = GridSearchCV (LinearSVC(random_state=0), tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1)
	clf.fit(X_treino2, y_train)

	print(  clf.best_params_['C'])
	c_param=clf.best_params_['C']

	return c_param
'''


def get_svm_params3 (X_train, y_train):
	params=1
	tuned_parameters = [{'C': [  0.00390, 0.01, 0.03125, 0.1, .25000, 1.01, 2, 8, 16, 128]}] 
	clf = GridSearchCV (LinearSVC(random_state=0), tuned_parameters, cv=3, scoring='accuracy', n_jobs=-1)
	clf.fit(X_train, y_train)
	params=clf.best_params_['C']
	print(params)
	#pickle.dump(params, open(data_md5, 'wb'))
	return params

def main():
	folds=5

	try:
		opts, args = getopt.getopt(sys.argv[1:],"d:k:u:r:",["d=","k=","u=","r="])
	except getopt.GetoptError:
			exit()

	for opt, arg in opts:
		if opt == '-d':
			dataset = arg
		if opt == '-k':
			k = int(arg)
		if opt == '-u':
			use = arg
		if opt == '-r':
			regression_bias = float(arg)



	os.system("mkdir datasets")
	os.system("mkdir datasets/saves")
	os.system("mkdir datasets/saves/"+dataset)
	os.system("mkdir datasets/lowmf")
	os.system("mkdir datasets/lowmf/"+dataset)
	os.system("mkdir datasets/lowmf/"+dataset+"/"+use)


	tim = []

	#for f in range(folds):
	for f in range(1):

		t1 = time.time()
		print(f)


		X_treino2, y_treino2 = get_data("original/"+dataset+"/treino"+str(f)+"_orig",0)
		X_teste, y_teste = get_data("original/"+dataset+"/teste"+str(f)+"_orig",0)


		
		
		if (X_treino2.shape[1]>X_teste.shape[1]):
			X_teste, y_teste = get_data("original/"+dataset+"/teste"+str(f)+"_orig",X_treino2.shape[1])
		else:
			X_treino2, y_treino2 = get_data("original/"+dataset+"/treino"+str(f)+"_orig",X_teste.shape[1])
			
		n_classes=int(max(np.max(y_treino2),np.max(y_teste)))+1
		print( "n_classes: "+str(n_classes))
		
		print("Shape")


		X_treino, y_treino=ajusta_treino(X_treino2, y_treino2,n_classes, splits*2)
			

		train = []
		test = []

		if use[7] == '1':
			print("hyperplane_errors")
			trname="datasets/saves/"+dataset+"/hwerror_tr"+str(f)+".npz"
			tename="datasets/saves/"+dataset+"/hwerror_te"+str(f)+".npz"
			#try:
			#    loadedtr = np.load(trname)
			#    loadedte = np.load(tename)
			#    train.append(loadedtr['x'])
			#    test.append(loadedte['x'])
			#except: 
			print("Generating features...")

			t_init = time.time()
			#cos_knn_treino,cos_knn_teste = hyperplane_errors(X_treino, y_treino, X_teste, y_teste, n_classes, f, k, 'cosine', 'original', save=False)
			train.append(cos_knn_treino)
			test.append(cos_knn_teste)
			print(time.time() - t_init,"seconds")
			np.savez_compressed(trname, x=train[-1])
			np.savez_compressed(tename, x=test[-1])


		if use[10] == '1':
			print("hyperplane_acima")
			trname="datasets/saves/"+dataset+"/hwacima_tr"+str(f)+".npz"
			tename="datasets/saves/"+dataset+"/hwacima_te"+str(f)+".npz"
			#try:
			#    loadedtr = np.load(trname)
			#    loadedte = np.load(tename)
			#    train.append(loadedtr['x'])
			#    test.append(loadedte['x'])
			#except: 
			print("Generating features...")

			t_init = time.time()
			#cos_knn_treino,cos_knn_teste = hyperplane_acima(X_treino, y_treino, X_teste, y_teste, n_classes, f, k, 'cosine', 'original', save=False)
			train.append(cos_knn_treino)
			test.append(cos_knn_teste)
			print(time.time() - t_init,"seconds")
			np.savez_compressed(trname, x=train[-1])
			np.savez_compressed(tename, x=test[-1])



		treino_orig_all = np.concatenate(train,axis=1)
		teste_orig_all = np.concatenate(test,axis=1)

		if(np.any(np.isnan(treino_orig_all))):
			print('Nan1')
			treino_orig_all[np.isnan(treino_orig_all)] = float(0.0)
		if(np.any(np.isnan(teste_orig_all))):
			print('Nan2')
			teste_orig_all[np.isnan(teste_orig_all)] = float(0.0)

		tim.append(time.time() - t1)

		dump_svmlight_file(treino_orig_all,y_treino, f="datasets/lowmf/"+dataset+"/"+use+"/treino"+str(f)+"_"+use, zero_based=False)
		dump_svmlight_file(teste_orig_all,y_teste, f="datasets/lowmf/"+dataset+"/"+use+"/teste"+str(f)+"_"+use, zero_based=False)

	print("Time")
	print("avg:",np.mean(tim))
	print('std:',np.std(tim))
	exit()


