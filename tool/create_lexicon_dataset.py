import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np

def createLexicon(wordListFile, lookupFile, outputFile):
    lexicon = frozenset([])
    with open(wordListFile) as f:
    	wordListFile = f.read().splitlines()
    nSamples = len(wordListFile)
    
    cache = {}
    cnt = 1
    with open(lookupFile) as f:
        lookupTableList = f.read().splitlines()
    for i in xrange(nSamples):
        labelString = wordListFile[i].split(' ')[0]
	label=""
	#the string is spliited into unicode code points
	#now each of the unicode points decimal value is found and
	#labels is a concatenated sequence of such values seprated by space
	#then the index of this value in the lookupfile is found so finally this can be used as the
	#class label for the network ( index begin from zero)

	for uc in unicode(labelString,'utf-8'):
		decimal_uc=str(ord(uc))
		label_index=str(lookupTableList.index(decimal_uc))
		label+=label_index + ' '
	label=label.strip()
        lexicon = lexicon.union([label])
		
    f= open(outputFile,"w")
    for i in lexicon:
       f.write(i)
       f.write('\n')
    f.close()
    
createLexicon(wordListFile='hindi_vocab.txt', lookupFile='hindi_final_lookup.txt', outputFile='lexicon-hindi.txt')
