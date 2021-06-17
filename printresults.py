import argparse
import json
import io
from scipy.stats import t as qt
import numpy as np

def stats(micro_list, macro_list):
	nfolds = len(micro_list)
	med_mic = np.mean(micro_list)*100
	error_mic = abs(qt.isf(0.975,df=(nfolds-1)))*np.std(micro_list,ddof=1)/np.sqrt(nfolds)*100
	med_mac = np.mean(macro_list)*100
	error_mac = abs(qt.isf(0.975,df=(nfolds-1)))*np.std(macro_list,ddof=1)/np.sqrt(nfolds)*100

	print("{:.1f}({:.1f})\t{:.1f}({:.1f})".format(med_mic,error_mic,med_mac,error_mac))#,end="")



def arguments():
	parser = argparse.ArgumentParser(description='Bert.')
	#parser.add_argument("--out", type=str)
	#parser.add_argument("--method", type=str)
	parser.add_argument("--nfolds", type=int, default=10)
	args = parser.parse_args()

	


	return args



args = arguments()

datasets = ["20ng", "acm", "reut90", "webkb", "ohsumed", "wos5736", "wos11967", "books", "dblp", "trec", "pang_movie_2L", "vader_movie_2L", "yelp_reviews_2L", "vader_nyt_2L", "mr", "sst1", "sst2", "subj", "mpqa"]

for dataset in datasets:

	#print(dataset)
	if dataset in ["sst2", "subj", "mpqa"]:
		args.path = f"out/{dataset}/tfidf_mf_cosine_knn_l2_knn_cosine_cent_l2_cent_cosine_err/"
	else:
		args.path = f"out/{dataset}/tfidf_mf_cosine_knn_l2_knn_cosine_cent_l2_cent/"

	micro_list = []
	macro_list = []

	for f in range(args.nfolds):
		try:
			with open(f"{args.path}/out.fold={f}.json", "r") as fil:
				data = json.load(fil)

			micro_list.append(data['micro'])
			macro_list.append(data['macro'])
			#print(f"{data['micro']*100:.2f} {data['macro']*100:.2f}")
		except:
			#print("erro: " + ismethod)
			pass

	if len(micro_list) != 10:
		#print("erro: " + ismethod + " " + str(len(micro_list)))
		print("\t")
	else:
		#print(f"Stats for {len(micro_list)} folds")
		stats(micro_list, macro_list)
		#print()