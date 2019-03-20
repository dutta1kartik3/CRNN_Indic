# -*- coding: utf-8 -*-
#argument expected - a list of words or phrases used 
# what this does - would write the unique unicode points to arg[2]
import os,sys

textFile=sys.argv[1]
lookupFile=sys.argv[2]
with open(lookupFile) as f:
    labels =  frozenset(f.read().splitlines())
lookupFile = open(lookupFile,'w')
#print labels
#labels=[]
#print textFile
#print lookupFile
with open(textFile) as f :
	for line in f:
		for uc in unicode(line,'utf-8'):
			decimal_uc=ord(uc)
			if str(decimal_uc) not in labels:
				labels =  labels.union(frozenset([decimal_uc]))
				#print decimal_uc
#print labels
for item in labels:
  lookupFile.write("%s\n" % item)
