#!/usr/bin/env python

from __future__ import print_function

import sys
import re

dic = {'neutral': '0', 'entailment': '1', 'contradiction': '2'}
#dic = {'entailment': '1', 'contradiction': '2'}

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print("Usage:",sys.argv[0],"<if> <of>")
		sys.exit(-1)

	with open(sys.argv[1]) as ifh, open(sys.argv[2], 'w') as ofh:
		ifh.readline()
	
		for l in ifh:
			e = l.split("\t")
	
			label = e[0]

			parsed = e[4]

			ds = re.findall("\([^ ]+ [^ \)]+\)", parsed)


			r = [x[1:-1].split() for x in ds]
			r2 = [x[1]+"_"+x[0] for x in r]
		
			if label in dic:
				ofh.write(dic[label]+"\t"+" ".join(r2)+"\n")
			#else:
			#	print("Label",label,"not found")

	sys.exit(0)		
