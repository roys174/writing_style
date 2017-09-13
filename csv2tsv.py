#!/usr/bin/env python

import sys
import csv

def main(args):
    argc = len(args)
    
    if argc < 3: 
        print "Usage:",args[0],"<input file> <output file>"
        return -1
    
    ifile = args[1]
    ofile = args[2]
    
    with open(ifile) as ifh, open(ofile, 'w') as ofh:
        ifh = csv.reader(ifh)
        
        for l in ifh:
            ofh.write("\t".join(l)+"\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))