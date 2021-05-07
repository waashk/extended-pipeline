
import numpy as np
from scipy.stats import t as qt
import os
import io
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import argparse
import gzip


def print_stats(folds, micro_list, macro_list):
    #print(micro_list)
    med_mic = np.mean(micro_list)*100
    error_mic = abs(qt.isf(0.975, df=(folds-1))) * \
        np.std(micro_list, ddof=1)/np.sqrt(len(micro_list))*100
    med_mac = np.mean(macro_list)*100
    error_mac = abs(qt.isf(0.975, df=(folds-1))) * \
        np.std(macro_list, ddof=1)/np.sqrt(len(macro_list))*100
    print("Micro\tMacro")
    print("{:.2f}({:.2f})\t{:.2f}({:.2f})".format(
        med_mic, error_mic, med_mac, error_mac))
    return med_mic, error_mic, med_mac, error_mac


def print_in_file(msg, filename):
    with open(filename, 'a') as arq:
        arq.write(msg+"\n")


def get_data(inputdir, f):

    X_train, y_train = load_svmlight_file(
        inputdir+"train"+str(f)+".gz", dtype=np.float64)
    X_test, y_test = load_svmlight_file(
        inputdir+"test"+str(f)+".gz", dtype=np.float64)

    # Same vector size
    if (X_train.shape[1] > X_test.shape[1]):
        X_test, y_test = load_svmlight_file(
            inputdir+"test"+str(f)+".gz", dtype=np.float64, n_features=X_train.shape[1])
    elif (X_train.shape[1] < X_test.shape[1]):
        X_train, y_train = load_svmlight_file(
            inputdir+"train"+str(f)+".gz", dtype=np.float64, n_features=X_test.shape[1])

    n_classes = int(max(np.max(y_train), np.max(y_test)))+1

    return X_train, y_train, X_test, y_test, n_classes


def get_y_train(args, train_idx):
    with open(os.path.join(args.splitdir, 'score.txt'), 'r') as arq:
        y = np.array(list(map(str.rstrip, arq.readlines())))
    y_train = y[train_idx]
    return y_train


def get_array(X, idxs):
    return [X[idx] for idx in idxs]


def readfile(filename):
    with io.open(filename, 'rt', newline='\n', encoding='utf8', errors='ignore') as filein:
        return filein.readlines()


def load_splits_ids(folddir):
    splits = []
    with open(folddir, encoding='utf8', errors='ignore') as fileout:
        for line in fileout.readlines():
            train_index, test_index = line.split(';')
            train_index = list(map(int, train_index.split()))
            test_index = list(map(int, test_index.split()))
            splits.append((train_index, test_index))
    return splits


def save_splits_ids(splits, folddir):
    with open(folddir, 'w', encoding='utf8', errors='ignore') as fileout:
        for train_index, test_index in splits:
            line = ' '.join(list(map(str, train_index))) + ';' + \
                ' '.join(list(map(str, test_index))) + '\n'
            fileout.write(line)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dump_svmlight_file_gz(X,y,f,output_dir,train_or_test=""):
    #output_dir = "newdatasets/topics/{}/representations/{}-folds/{}/".format(args.dataset,args.folds,args.name)
    filename = "{}{}.gz".format(train_or_test,f)
    if not os.path.exists(output_dir):
        os.system("mkdir {}".format(output_dir))
    with gzip.open(output_dir+filename, 'w') as filout:
        dump_svmlight_file(X, y, filout, zero_based=False)

#def dump_svmlight_file_gz(X,y,f,args,train_or_test=""):
#    output_dir = "newdatasets/topics/{}/representations/{}-folds/{}/".format(args.dataset,args.folds,args.name)
#    filename = "{}{}.gz".format(train_or_test,f)
#    if not os.path.exists(output_dir):
#        os.system("mkdir {}".format(output_dir))
#    with gzip.open(output_dir+filename, 'w') as filout:
#        dump_svmlight_file(X, y, filout, zero_based=False)
