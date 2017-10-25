#!/usr/bin/env python

import spacy
import sys


def main(args):
    
    if (len(args) < 3):
        print("Usage: "+args[0]+" <input files> <output files>")
        sys.exit(-1)


    en_nlp = spacy.load('en')


    ifile = args[1]
    ofile = args[2]

    print("Reading "+ifile+" and writing "+ofile)
    n_lines = 0

    with open(ifile) as ifile, open(ofile, "wb") as ofile: 
        for line in ifile:
            n_lines = n_lines + 1

            if (n_lines % 1000 == 0):
                print (str(n_lines/1000)+"K    \r"),
                sys.stdout.flush()
    
            uline = unicode(line.rstrip(), errors='ignore')
            en_doc=en_nlp(uline)

            tagged_sentence = []

            for tok in en_doc:
                if tok.tag_ == "SP":
                    continue
                
                tagged_sentence.append(str(tok)+"_"+tok.tag_)

            ofile.write(" ".join(tagged_sentence)+"\n")

if __name__ == '__main__':
    sys.exit(main(sys.argv))
