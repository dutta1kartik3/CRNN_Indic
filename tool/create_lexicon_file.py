import os
import lmdb # install lmdb by "pip install lmdb"
import numpy as np
offset = 1

def createLex(outputPath, imagePathListFile,lookupFile):

	s = frozenset([])
	fp = open(outputPath,'w')	

	with open(imagePathListFile) as f:
        	imagePathList = f.read().splitlines()
	    	nSamples = len(imagePathList)

	with open(lookupFile) as f:
        	lookupTableList = f.read().splitlines()

	for i in xrange(nSamples):
        #print imagePathList[i]
	        labelString = imagePathList[i].split(' ')[1]
        	label=""
		count = 0
		for uc in unicode(labelString,'utf-8'):
                	decimal_uc=str(ord(uc))
	                label_index=str(lookupTableList.index(decimal_uc)+ offset)
        	        label+=label_index + ' '
			count+=1

		while(count < 32):
			label+= '0' + ' ' ##All labels need to have same length
			count+=1
        	label=label.strip()
		s = s.union([label])
		fp.write(label)
		fp.write('\n')
	
	fp.close()
	fp2 = open('lexicon-'+ outputPath,'w')
	for i in s:
		fp2.write(i)
		fp2.write('\n')
	fp2.close()
createLex(outputPath='temp-test-IH.txt', imagePathListFile='test.txt', lookupFile='hindi_ml.txt')
