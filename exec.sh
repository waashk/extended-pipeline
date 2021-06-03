

#for d in trec; # ohsumed wos5736 wos11967 cade12; # mr sst1 sst2 subj mpqa; 
#for d in mr sst1 sst2 trec wos5736 wos11967; # ohsumed wos5736 wos11967 cade12; # mr sst1 sst2 subj mpqa; 
for d in cade12; # ohsumed wos5736 wos11967 cade12; # mr sst1 sst2 subj mpqa; 
do
    python pipeline.py --folds 10 --out out --datain ../newdatasets/ --MFmetric cosine l2 --MFapproach knn cent err --classifier svm -d $d
    #python pipeline.py --folds 10 --out out --datain ../newdatasets/ --MFmetric cosine l2 --MFapproach err --classifier svm -d $d
    #python pipeline.py --folds 10 --out outerr --datain datasets/ --MFmetric cosine l2 --MFapproach knn cent err --classifier svm -d $d --savefinalrep 1
done;