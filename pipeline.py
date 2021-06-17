


from inout.arguments import arguments
from inout.general import get_data, dump_svmlight_file_gz, print_in_file, load_svmlight_file
from mf.metafeatures import MetaFeatures
from instanceselection.cover import Cover
import numpy as np
import gc
from mf.bestk import findBestK, findBestKsaved
from mf.errbased import hyperplane_errors, hyperplane_acima, ajusta_treino
from classifiers.traditionalClassifiers import TraditionalClassifier
from classifiers.config import getClassifierInfo
from sklearn.metrics import f1_score
import copy
from collections import Counter
import socket
import json
import time

def main():
	gc.collect()
	args = arguments()

	#for f in range(1): #args.folds
	for f in range(args.folds):
		print("Fold {}".format(f))

		X_train, y_train, X_test, y_test, _ = get_data(args.inputdir, f)
		X_train_tfidf = copy.copy(X_train) 
		X_test_tfidf = copy.copy(X_test)

		#(metric, approach)
		#groups = [("cosine", "knn"), ("cosine", "cent"), ("l2", "knn"), ("l2", "cent")]
		if args.mfgroups:
			try:
				k = findBestKsaved(args.dataset)
			except:
				k = findBestK(X_train, y_train)

			t_init = time.time()
			mf = MetaFeatures(groups=args.mfgroups, k=k)
			mf.fit(X_train, y_train)
			X_train = mf.transform(X_train)
			t_train = time.time() - t_init 

			t_init = time.time()
			X_test  = mf.transform(X_test)
			t_test = time.time() - t_init 
			#TODO: Remover apos refatorar codigo mf err
			if 'err' in args.MFapproach:

				ninstances = X_train_tfidf.shape[0]
				n_classes = len(sorted(map(int, list(set(y_train)))))
				X_train2, y_train2 = ajusta_treino(X_train_tfidf, y_train, n_classes, args.folds*2)
				mf_errors_treino, mf_errors_teste = hyperplane_errors(X_train2, y_train2, X_test_tfidf, k)
				mf_acima_treino, mf_acima_teste   =  hyperplane_acima(X_train2, y_train2, X_test_tfidf, k)
				X_train = np.hstack((X_train, mf_errors_treino[:ninstances], mf_acima_treino[:ninstances]))
				X_test = np.hstack((X_test, mf_errors_teste, mf_acima_teste))

		if args.cover:
			cover = Cover(percentsel=args.cover)
			cover.fit(X_train, y_train)
			X_train, y_train = cover.transform(X_train)

		if args.savefinalrep:
			dump_svmlight_file_gz(X_train,y_train,f,output_dir=f"{args.outputdir}/",train_or_test="train")
			dump_svmlight_file_gz(X_test,y_test,f,output_dir=f"{args.outputdir}/",train_or_test="test")

		if args.doclf:
			info = getClassifierInfo(args.classifier)
			info['dataset'] = args.dataset
			classifier = TraditionalClassifier(info)
			classifier.fit(X_train, y_train)
			y_pred = classifier.predict(X_test)
			micro = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
			macro = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
			print(f"F1-Score - fold={f}")
			print_in_file(f"F1-Score - fold={f}", args.filename)
			print("\tMicro: ", micro)
			print_in_file(f"\tMicro: {micro}", args.filename)
			print("\tMacro: ", macro)
			print_in_file(f"\tMacro: {macro}", args.filename)


		data = {
			"hiperparams": str(args),
			"machine": socket.gethostname(),
			#"micro": micro,
			#"macro": macro,
			"mf_train_time": t_train,
			"mf_test_time": t_test,
			"mf_total_time": t_train + t_test,
			"mf_k":k,
		}

		filename = f"{args.outputdir}/out"
		with open(f"{filename}.fold={f}.json", 'w') as outfile:
			json.dump(data, outfile, indent=4)


if __name__ == '__main__':
	main()
