#!/usr/bin/env python

### Utils for logistic regression.

import numpy as np
from scipy.sparse import lil_matrix
import random
import sys
import pandas
from tqdm import tqdm

tqdm.pandas(desc="splitter")

# Load features.
def read_features(ifile, n_feats):
    
    labels = []
    
    with open(ifile) as ifh:
        lines = [l.rstrip() for l in ifh]
    df = pandas.read_csv(ifile, sep = '\t', names = ['label', 'semantics', 'features'])

    df_feats = df['features'].str.split()

    def splitter(ls):
        res = {}
        for item in ls:
            [f,v] = item.split(':')
            res[int(f)] = float(v)
        return res
    
    sp = lil_matrix((df.shape[0], n_feats))

    def orderer(d, i):
        for k,v in d.items():
            sp[i, k] = float(v)
    
    df_feats = df_feats.progress_apply(splitter)


    i = 0
    for item in tqdm(df_feats, "orderer"):
        orderer(item, i)
        i+=1

    return sp, df['label']

# Evaluate logistic regression
def evaluate(test,labels):
    n_correct = 0
    
    for i in range(len(labels)):
        n_correct += (labels[i]==test[i])
    
    v = round(n_correct*1./len(labels), 3)
    
    print(n_correct,"/",len(labels),"=",str(v))

    return v

