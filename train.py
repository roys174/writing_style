#!/usr/bin/env python

import sys
import logistic_regression_tools
from sklearn import linear_model
import pickle
from optparse import OptionParser,OptionValueError

def main():
    options = usage()

    ifile = options.ifile
    C = float(options.C)
    n_feats = int(options.n_feats)
    model_of = options.model_of

    # Load features.
    [features, labels] = logistic_regression_tools.read_features(ifile, n_feats)

    clf = linear_model.LogisticRegression(C=C)

    clf.fit(features, labels)
    pickle.dump(clf, open(model_of, 'wb'))

    test = clf.predict(features)
    
    logistic_regression_tools.evaluate(test,labels)
    
    return 0

def usage():
    C=0.025

    parser = OptionParser()

    parser.add_option("-i", dest="ifile",
                    help="Input file", metavar="FILE")
    parser.add_option("-o", dest="model_of",
                    help="Model output file", metavar="FILE")
    parser.add_option("-n", dest="n_feats",
                    help="Number of features", metavar="INT")
    parser.add_option("-c", metavar="FLOAT",
                            dest="C", 
                            help="Regularization parameter",
                            default=C)

    (options, args) = parser.parse_args()

    if (options.ifile == None or options.model_of == None):
            raise OptionValueError("input file or model file missing")
    
    return options


    
if __name__ == '__main__':
    sys.exit(main())
