


from inout.arguments import arguments
from inout.general import get_data, dump_svmlight_file_gz, load_svmlight_file
from mf.metafeatures import MetaFeatures
from instanceselection.cover import Cover
import numpy as np
import gc


def main():
	gc.collect()
	args = arguments()

	for f in range(1): #args.folds
		print("Fold {}".format(f))

		X_train, y_train, X_test, y_test, _ = get_data(args.inputdir, f)

		#(metric, approach)
		#groups = [("cosine", "knn"), ("cosine", "cent"), ("l2", "knn"), ("l2", "cent")]
		if args.mfgroups:
			mf = MetaFeatures(groups=args.mfgroups, k=20)
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


if __name__ == '__main__':
	main()
