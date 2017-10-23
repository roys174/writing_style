#!/usr/bin/env python

from __future__ import print_function
import sys
import logistic_regression_tools
import pickle


def main():
    if (len(sys.argv) < 3):
        print ("Usage:",sys.argv[0],"<if> <model if>")
        return -1

    clf = pickle.load(open(sys.argv[2], 'r'))
    n_feats = len(clf.coef_[0])
    
    [features, labels] = logistic_regression_tools.read_features(sys.argv[1], n_feats)
    
    test = clf.predict(features)
    
    logistic_regression_tools.evaluate(test,labels)
    


sys.exit(main())