#!/usr/bin/env python

import sys
sys.path.insert(0, '/Users/roysch/aristo/entailment/scripts/')
import logistic_regression_tools
import pickle


def main():
    ids=None
    gold=None
    out_file = None
    if (len(sys.argv) < 3):
        print "Usage:",sys.argv[0],"<if> <model if> <is ranking> <mapping file> <out file>" 
        return -1
    elif (len(sys.argv) > 3):
        mapping_file = sys.argv[3]
        ids,gold = read_mapping_file(mapping_file)
        if (len(sys.argv) > 4):
            out_file = sys.argv[4]

    clf = pickle.load(open(sys.argv[2], 'r'))
    n_feats = len(clf.coef_[0])
    
    features = []
    labels = []

    [features, labels] = logistic_regression_tools.read_features(sys.argv[1], n_feats)
    
    test = clf.predict(features)
    
    confidence = clf.decision_function(features)
    
    
    logistic_regression_tools.evaluate(test,labels,confidence,ids,gold, out_file)
    

def read_mapping_file(ifile):
    ids = []
    gold = []
    with open(ifile) as ifh:
        # Ignore first line
        ifh.readline()

        for line in ifh:
            e = line.rstrip().split("\t");
            id = e[0]
            ids.append(id)
            gold.append(int(e[-1]))
	
        ifh.close()

	return ids,gold


sys.exit(main())