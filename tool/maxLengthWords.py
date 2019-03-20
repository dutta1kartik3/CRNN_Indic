# -*- coding: utf-8 -*-
#argument expected - a list of words or phrases used 
# what this does - would write the words which are longer than the limit and would cause issue in training / testing
import os,sys

textFile=sys.argv[1]

#print textFile
#print lookupFile
with open(textFile) as f :
	for line in f:
		count=0	
		for uc in unicode(line,'utf-8'):
			count=count+1
		if count > 26:
			print line
