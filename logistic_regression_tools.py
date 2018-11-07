#!/usr/bin/env python

### Utils for logistic regression.

import numpy as np
import random
import sys

# Load features.
def read_features(ifile, n_feats):
    all_features = []
    labels = []
    
    with open(ifile) as ifh:
        for line in ifh:
            d = line.rstrip().split("\t")
        
            if (len(d) < 3):
                continue
        
            labels.append(d[0])
        
            features = d[2].split(" ")
        
            local_features = [0 for i in range(n_feats)]

            for feature in features:
                [f,v] = feature.split(":")
            
                f = int(f)
                
                local_features[f] = float(v)
        
            all_features.append(local_features)
              
    return [all_features, labels]
    
# Evaluate logistic regression
def evaluate(test,labels,confidence = [], ids = None, gold = None, ofile = None):
    n_correct = 0
    
    if (ofile != None):
        ofh = open(ofile, 'w')
        ofh.write("InputStoryid,AnswerRightEnding\n")
        
    
    for i in range(len(labels)):
        n_correct += (labels[i]==test[i])
    
    print n_correct,"/",len(labels),"=",round(n_correct*1./len(labels), 3)

    if (len(confidence)):        
        n_pairs_correct = 0

        replace = {'2':'1', '1':'2'}
        
        # If model predicts the same label for both laternatives, take the one with the highest confidence.
        for i in range(len(labels)/2):
            if (test[2*i] == test[2*i+1]):
                if (abs(confidence[2*i]) > abs(confidence[2*i+1])):
                    test[2*i+1] = replace[test[2*i+1]]
                else:
                    test[2*i] = replace[test[2*i]]
            
            is_correct = (labels[2*i]==test[2*i])
            
            n_pairs_correct += is_correct
            n_pairs_correct += (labels[2*i+1]==test[2*i+1])

            if (ofile != None):
                v = gold[i]
                if (not is_correct):
                    v = 3-v
                    
                ofh.write(ids[i]+","+str(v)+"\n")
    
        # Print results.
        print n_pairs_correct,"/",len(labels),"=",round(n_pairs_correct*1./len(labels), 3)

    if (ofile != None):
        ofh.write("\n")
        ofh.close()

