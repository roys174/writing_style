#!/usr/bin/env python

### Utils for logistic regression.

import numpy as np
from scipy.sparse import lil_matrix
import random
import sys

# Load features.
def read_features(ifile, n_feats):
    labels = []
    
    with open(ifile) as ifh:
        lines = [l.rstrip() for l in ifh]
    
    all_features = lil_matrix((len(lines), n_feats), dtype=np.int8)
    
    # print("Created sparse matrix of size",len(lines), n_feats, all_features.shape)
    
    i = 0
    for line in lines:
        d = line.rstrip().split("\t")
        
        if (len(d) < 3):
            print("Bad line", line)
            continue
        
        labels.append(d[0])
        
        features = d[2].split(" ")
        data = dict()
        
        for feature in features:
            [f,v] = feature.split(":")
            
            f = int(f)

            data[f] = v
                
            
        
        for k in sorted(data.keys()):
            all_features[i, k] = float(data[k])
        
        i += 1
              
    return [all_features, labels]
    
# Evaluate logistic regression
def evaluate(test,labels):
    n_correct = 0
    
    for i in range(len(labels)):
        n_correct += (labels[i]==test[i])
    
    v = round(n_correct*1./len(labels), 3)
    
    print n_correct,"/",len(labels),"=",str(v)

    return v

