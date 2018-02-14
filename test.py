#!/usr/bin/env python

from __future__ import print_function
import sys
import logistic_regression_tools
import pickle


def main():
    ofile = None
    text_ifile = None
    if (len(sys.argv) < 3):
        print ("Usage:",sys.argv[0],"<if> <model if> <text ifile - optional> <ofile - optional>")
        return -1
    elif (len(sys.argv) > 4):
        text_ifile = sys.argv[3]
        ofile = sys.argv[4]

    clf = pickle.load(open(sys.argv[2], 'r'))
    n_feats = len(clf.coef_[0])
    
    [features, labels] = logistic_regression_tools.read_features(sys.argv[1], n_feats)
    
    test = clf.predict(features)
    
    logistic_regression_tools.evaluate(test,labels)
    
    if ofile is not None:
        with open(ofile, 'w') as ofh, open(text_ifile) as ifh:
            for l, test, label in zip(ifh, test, labels):
                ofh.write("{}   {}  {}".format(test, label, l))

sys.exit(main())