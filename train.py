#!/usr/bin/env python

from __future__ import print_function

import sys
import logistic_regression_tools
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import pickle
import numpy as np
from optparse import OptionParser,OptionValueError
from tqdm import tqdm


def main():
    options = usage()

    ifile = options.ifile
    
    with open(options.features_file) as ifh:
        n_feats = len(ifh.readlines())

    if options.verbose:
        print("Training with", n_feats, "features")

        
    model_of = options.model_of
    Cs = [float(x) for x in options.Cs.split(",")]

    # Load features.
    if options.verbose:
        print("Reading", ifile)

    [train_features, train_labels] = logistic_regression_tools.read_features(ifile, n_feats)

    dev_features = None

    cv = None
    if options.dev_file is not None:
        if options.verbose:
            print("Reading", options.dev_file)
        [dev_features, dev_labels] = logistic_regression_tools.read_features(options.dev_file, n_feats)  

        max_r = -1
        argmax = None

        for C in tqdm(Cs):
            print ("Testing",C)
            clf = linear_model.LogisticRegression(C=C, verbose=True)

            clf.fit(train_features, train_labels)

            if options.verbose:
                print("Done, now predicting")
            train_predicted_labels = clf.predict(train_features)
            if options.verbose:
                print("Done, now evaluating")
    
            v = logistic_regression_tools.evaluate(train_predicted_labels,train_labels)
        
            if dev_features is not None:
                dev_predicted_labels = clf.predict(dev_features)
                v = logistic_regression_tools.evaluate(dev_predicted_labels,dev_labels)

            if v > max_r:
                max_r = v
                argmax = C
    elif len(Cs) == 1:
        argmax = float(options.Cs[0])        
    elif options.n_folds is not None:
        max_r = -1
        argmax = None
        cv = int(options.n_folds)
        
        for C in tqdm(Cs):
            clf = linear_model.LogisticRegression(C=C, verbose=True)

            scores = cross_val_score(clf, train_features, train_labels, cv=cv)
            v = np.mean(scores)

            print(C,v)
            if v > max_r:
                max_r = v
                argmax = C
    else:
        print("More than one C value provided but no dev set to use for selection")
        return -2

    clf = linear_model.LogisticRegression(C=argmax)
    clf.fit(train_features, train_labels)
    train_predicted_labels = clf.predict(train_features)
    logistic_regression_tools.evaluate(train_predicted_labels,train_labels)

    pickle.dump(clf, open(model_of, 'wb'))
    
    return 0

def usage():
    C = "1,0.5,0.1,0.05,0.01"

    parser = OptionParser()

    parser.add_option("-i", dest="ifile",
                    help="Input file", metavar="FILE")
    parser.add_option("-d", dest="dev_file",
                    help="Input development file", metavar="FILE")
    parser.add_option("-e", dest="features_file",
                    help="Feature file", metavar="FILE")
    parser.add_option("-o", dest="model_of",
                    help="Model output file", metavar="FILE")
    parser.add_option("-v", dest="verbose",
                    help="Verbose mode", metavar="bool", default=False)
    parser.add_option("-f", dest="n_folds",
                    help="Run n-fold cross validation", metavar="INT", default=5)
    parser.add_option("-c", metavar="string",
                            dest="Cs", 
                            help="Regularization parameters (comma separated)",
                            default=C)

    (options, args) = parser.parse_args()

    if (options.ifile == None or options.model_of == None):
            raise OptionValueError("input file or model file missing")
    
    return options


    
if __name__ == '__main__':
    sys.exit(main())
