


from inout.arguments import arguments
from inout.general import get_data, dump_svmlight_file_gz, print_in_file, load_svmlight_file
from mf.metafeatures import MetaFeatures
from instanceselection.cover import Cover
import numpy as np
import gc
from mf.bestk import findBestK, findBestKsaved
from classifiers.traditionalClassifiers import TraditionalClassifier
from classifiers.config import getClassifierInfo
from sklearn.metrics import f1_score


def main():
	gc.collect()
	args = arguments()

	for f in range(1): #args.folds
		print("Fold {}".format(f))

		X_train, y_train, X_test, y_test, _ = get_data(args.inputdir, f)

		#(metric, approach)
		#groups = [("cosine", "knn"), ("cosine", "cent"), ("l2", "knn"), ("l2", "cent")]
		if args.mfgroups:
			#k = findBestK(X_train, y_train)
			k = findBestKsaved(args.dataset)
			mf = MetaFeatures(groups=args.mfgroups, k=k)
			mf.fit(X_train, y_train)
			X_train = mf.transform(X_train)
			X_test  = mf.transform(X_test)
		
		if args.cover:
			cover = Cover(percentsel=0.65)
			cover.fit(X_train, y_train)
			X_train, y_train = cover.transform(X_train)

		if args.savefinalrep:
			dump_svmlight_file_gz(X_train,y_train,f,output_dir=f"{args.output_dir}/",train_or_test="train")
			dump_svmlight_file_gz(X_test,y_test,f,output_dir=f"{args.output_dir}/",train_or_test="test")

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


if __name__ == '__main__':
	main()
