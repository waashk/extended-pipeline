

#for d in trec; # ohsumed wos5736 wos11967 cade12; # mr sst1 sst2 subj mpqa; 
#for d in mr sst1 sst2 trec wos5736 wos11967; # ohsumed wos5736 wos11967 cade12; # mr sst1 sst2 subj mpqa; 
#for d in cade12; # ohsumed wos5736 wos11967 cade12; # mr sst1 sst2 subj mpqa; 
#do
#    python pipeline.py --folds 10 --out out --datain ../newdatasets/ --MFmetric cosine l2 --MFapproach knn cent err --classifier svm -d $d
#    #python pipeline.py --folds 10 --out out --datain ../newdatasets/ --MFmetric cosine l2 --MFapproach err --classifier svm -d $d
#    #python pipeline.py --folds 10 --out outerr --datain datasets/ --MFmetric cosine l2 --MFapproach knn cent err --classifier svm -d $d --savefinalrep 1
#done;

for d in ohsumed wos5736 wos11967 books dblp cade12 trec sst1;
do
	python pipeline.py --folds 10 --out out --datain datasets/ --MFmetric cosine l2 --MFapproach knn cent --classifier svm -d $d
done;

for d in pang_movie_2L vader_movie_2L yelp_reviews_2L vader_nyt_2L mr sst1 sst2 subj mpqa;
do
	python pipeline.py --folds 10 --out out --datain datasets/ --MFmetric cosine l2 --MFapproach knn cent err --classifier svm -d $d
done;

#ohsumed wos5736 wos11967 books dblp cade12 trec
#pang_movie_2L vader_movie_2L yelp_reviews_2L vader_nyt_2L mr sst1 sst2 subj mpqa

#for d in ohsumed wos5736 wos11967 books dblp cade12 trec pang_movie_2L vader_movie_2L yelp_reviews_2L vader_nyt_2L mr sst1 sst2 subj mpqa; do mkdir datasets/$d ; done; 

aws ec2 stop-instances --instance-ids i-008218d959297ac74