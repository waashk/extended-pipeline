
from inout.general import str2bool
from datetime import datetime
import argparse
import os
import random


def arguments():
	# datasets/webkb/tfidf/ --splitdir datasets/webkb/ --outputdir output/webkb/cnn/
	parser = argparse.ArgumentParser(description='Generate baseline splits.')
	parser.add_argument("--datain", required=True)
	parser.add_argument('-d', '--dataset', type=str)
	parser.add_argument('--folds', type=int, default=10)

	#Metafeatures groups
	parser.add_argument('--MFmetric', nargs='+', default=[], type=str, choices=['l1', 'l2', 'cosine'])
	parser.add_argument('--MFapproach', nargs='+', default=[], type=str, choices=['knn', 'cent'])

	#Cover selection
	parser.add_argument('--cover', type=float, default=0.)
	
	parser.add_argument("--out", required=True)
	parser.add_argument('--savefinalrep', type=int, default=0)

	args = parser.parse_args()


	args.mfgroups = [(metric, approach) for approach in args.MFapproach for metric in args.MFmetric]
	
	repname = ["tfidf"]
	
	if args.mfgroups:
		repname += ["mf"]
		for (metric, approach) in args.mfgroups:
			repname += [metric, approach]

	if args.cover:
		repname += ["cover", "{:.2f}".format(args.cover)]

	repname = "_".join(repname)

	args.inputdir = f'{args.datain}/{args.dataset}/tfidf/'
	args.outputdir = f'{args.out}/{args.dataset}/{repname}/'

	if not os.path.exists(args.outputdir):
		print(f"Criando saida {args.outputdir}")
		os.system("mkdir -p {}".format(args.outputdir))


	random.seed(1608637542)

	return args 